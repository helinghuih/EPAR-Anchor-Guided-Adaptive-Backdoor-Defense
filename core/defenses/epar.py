import os
import os.path as osp
import time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .base import Base
from ..utils.log import Log
import core.defenses.analysis_utils as utils
from .official_cul import train_official_cul_model


def kmeans_pytorch(X, num_clusters=2, max_iter=100, tol=1e-4, device='cuda', init_anchor=None):
    if X.shape[0] < num_clusters:
        return torch.zeros(X.shape[0], dtype=torch.long, device=device), X

    X = X.to(device)
    if init_anchor is not None and num_clusters == 2:
        c0 = F.normalize(init_anchor.to(device).view(1, -1), p=2, dim=1).squeeze(0)
        dists_to_c0 = torch.cdist(X, c0.unsqueeze(0)).squeeze(1)
        far_k = max(1, int(0.10 * X.size(0)))
        far_idx = torch.topk(dists_to_c0, k=far_k, largest=True).indices
        c1 = F.normalize(X[far_idx].mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
        centroids = torch.stack([c0, c1])
    else:
        indices = torch.randperm(X.size(0), device=device)[:num_clusters]
        centroids = X[indices].clone()

    for _ in range(max_iter):
        dists = torch.cdist(X, centroids)
        labels = torch.argmin(dists, dim=1)

        new_centroids = []
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                c = F.normalize(X[mask].mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
                new_centroids.append(c)
            else:
                idx = torch.randint(0, X.size(0), (1,), device=device).item()
                new_centroids.append(X[idx])

        new_centroids = torch.stack(new_centroids)
        if torch.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return labels, centroids


def kmeans_1d_signed(scores_np, max_iter=100, tol=1e-6):
    scores = np.asarray(scores_np, dtype=np.float64)
    if scores.size < 2:
        return np.zeros(scores.size, dtype=np.int64)

    c0 = float(np.percentile(scores, 25))
    c1 = float(np.percentile(scores, 75))
    if abs(c1 - c0) < tol:
        c0, c1 = float(scores.min()), float(scores.max())

    labels = np.zeros(scores.size, dtype=np.int64)
    for _ in range(max_iter):
        d0 = np.abs(scores - c0)
        d1 = np.abs(scores - c1)
        new_labels = (d1 < d0).astype(np.int64)

        new_c0 = float(scores[new_labels == 0].mean()) if (new_labels == 0).any() else c0
        new_c1 = float(scores[new_labels == 1].mean()) if (new_labels == 1).any() else c1
        labels = new_labels
        if abs(new_c0 - c0) + abs(new_c1 - c1) < tol:
            break
        c0, c1 = new_c0, new_c1

    return labels


def anchor_subspace_clustering(g2_feats_norm, anchor_feats_norm, anchor_feat_center):
    diff = g2_feats_norm - anchor_feat_center
    anchor_std = anchor_feats_norm.std(dim=0, keepdim=True) + 1e-4
    z_diff = torch.abs(diff) / anchor_std
    topk_vals, _ = torch.topk(z_diff, k=min(32, z_diff.size(1)), dim=1)
    anomaly_score = topk_vals.mean(dim=1).numpy()

    labels = kmeans_1d_signed(anomaly_score)
    c0_mean = anomaly_score[labels == 0].mean() if (labels == 0).any() else 0.0
    c1_mean = anomaly_score[labels == 1].mean() if (labels == 1).any() else 0.0
    if c0_mean <= c1_mean:
        return (labels == 0), (labels == 1), float(c0_mean), float(c1_mean)
    else:
        return (labels == 1), (labels == 0), float(c1_mean), float(c0_mean)


class EPAR(Base):

    def __init__(self, model, loss, poisoned_trainset, poisoned_testset, clean_testset,
                 num_classes, target_class, poisoned_retrainset=None, trusted_clean_set=None,
                 seed=0, deterministic=False):
        super(EPAR, self).__init__(seed, deterministic)
        self.model = model
        self.loss = loss
        self.poisoned_trainset = poisoned_trainset
        self.poisoned_retrainset = poisoned_retrainset if poisoned_retrainset is not None else poisoned_trainset
        self.poisoned_testset = poisoned_testset
        self.clean_testset = clean_testset
        self.trusted_clean_set = trusted_clean_set
        self.num_classes = num_classes
        self.target_class = target_class
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, schedule, model_factory):
        work_dir = osp.join(schedule['save_dir'],
                            schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        self.log = Log(osp.join(work_dir, 'log.txt'))
        self.work_dir = work_dir

        d_name = schedule.get('dataset_name', 'Dataset')
        a_name = schedule.get('attack_name', 'Attack')

        self.log(f"========== EPAR Defense ({d_name} - {a_name}, Seed {self.seed}) ==========")

        if 'device' in schedule and schedule['device'] == 'GPU':
            self.device = torch.device("cuda")
        self.model = self.model.to(self.device)

        if 'shots' not in schedule['cul_schedule']:
            raise ValueError("Config Error: 'shots' missing!")
        shots = schedule['cul_schedule']['shots']

        self.log("[Phase 1] Exposure and Partition...")
        if self.trusted_clean_set is None:
            self.trusted_clean_set = utils.get_few_shot_clean_subset(
                self.poisoned_trainset, num_classes=self.num_classes, shots=shots, seed=self.seed
            )
        cul_subset = self.trusted_clean_set

        m_expo = train_official_cul_model(
            self.model, cul_subset, self.poisoned_testset, self.clean_testset,
            schedule['cul_schedule'], self.log, device=self.device
        )

        def get_preds_full(model, ds):
            model.eval()
            loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
            preds, labels = [], []
            with torch.no_grad():
                for img, label in loader:
                    preds.append(model(img.to(self.device)).argmax(1).cpu())
                    labels.append(label.cpu())
            return torch.cat(preds), torch.cat(labels)

        seed_preds, seed_labels = get_preds_full(m_expo, self.trusted_clean_set)
        p_expo, given_labels = get_preds_full(m_expo, self.poisoned_trainset)

        seed_keep_ratio = np.zeros(self.num_classes, dtype=np.float64)
        train_keep_ratio = np.zeros(self.num_classes, dtype=np.float64)
        for c in range(self.num_classes):
            s_mask = (seed_labels == c)
            if s_mask.sum() > 0:
                seed_keep_ratio[c] = (seed_preds[s_mask] == c).float().mean().item()
            t_mask = (given_labels == c)
            if t_mask.sum() > 0:
                train_keep_ratio[c] = (p_expo[t_mask] == c).float().mean().item()

        excess_retention = train_keep_ratio - seed_keep_ratio
        dom_class = Counter(seed_preds.tolist()).most_common(1)[0][0]
        if float(np.max(excess_retention)) > 0.10 and seed_keep_ratio[dom_class] < 0.50:
            inferred_target = int(np.argmax(excess_retention))
        else:
            inferred_target = int(dom_class)

        self.log(f"Inferred Target Class: {inferred_target}")

        anchor_indices = [
            i for i in range(len(self.trusted_clean_set))
            if int(self.trusted_clean_set[i][1]) == inferred_target
        ]
        anchor_ds = Subset(self.trusted_clean_set, anchor_indices)

        anchor_loader = DataLoader(anchor_ds, batch_size=32, shuffle=False, num_workers=4)
        anchor_ddn_res = utils.analyze_batch_with_ddn(
            self.model, anchor_loader, self.device, num_classes=self.num_classes, ddn_steps=100
        )
        anchor_hfe_values = [res['freq_diff_pre_post_energy'] for res in anchor_ddn_res]
        hfe_anchor_val = float(np.mean(anchor_hfe_values)) if anchor_hfe_values else 0.0

        self.log("[Phase 2] Anchor-Guided Identification...")

        g2_indices = [
            i for i in range(len(self.poisoned_trainset))
            if given_labels[i].item() == inferred_target
        ]

        features_buffer = {}
        target_layer = next(m for _, m in reversed(list(self.model.named_modules())) if isinstance(m, nn.Linear))
        handle = target_layer.register_forward_hook(
            lambda m, inp, out: features_buffer.__setitem__('feat', inp[0].detach().flatten(1))
        )

        anchor_feats_list = []
        self.model.eval()
        with torch.no_grad():
            for imgs, _ in anchor_loader:
                _ = self.model(imgs.to(self.device))
                anchor_feats_list.append(features_buffer['feat'].cpu())
        anchor_feats_norm = F.normalize(torch.cat(anchor_feats_list, dim=0), p=2, dim=1)
        anchor_feat_center = F.normalize(anchor_feats_norm.mean(dim=0, keepdim=True), p=2, dim=1)

        indices_c0, indices_c1 = [], []
        c0_center, c1_center = None, None

        if len(g2_indices) > 0:
            ds_g2 = utils.MatchSampleDataset(self.poisoned_trainset, g2_indices)
            loader_g2 = DataLoader(ds_g2, batch_size=128, shuffle=False, num_workers=4)
            g2_feats_list, g2_indices_map = [], []

            with torch.no_grad():
                for imgs, _, indices in loader_g2:
                    _ = self.model(imgs.to(self.device))
                    g2_feats_list.append(features_buffer['feat'].cpu())
                    g2_indices_map.extend(indices.tolist())

            g2_feats_norm = F.normalize(torch.cat(g2_feats_list, dim=0), p=2, dim=1)

            labels, _ = kmeans_pytorch(
                g2_feats_norm, num_clusters=2, device=self.device, init_anchor=anchor_feat_center
            )
            mask_0, mask_1 = (labels == 0).cpu(), (labels == 1).cpu()
            c0_center = F.normalize(g2_feats_norm[mask_0].mean(dim=0, keepdim=True), p=2, dim=1)
            c1_center = F.normalize(g2_feats_norm[mask_1].mean(dim=0, keepdim=True), p=2, dim=1)

            feat_dist0_init = float(1.0 - F.cosine_similarity(anchor_feat_center, c0_center).item())
            feat_dist1_init = float(1.0 - F.cosine_similarity(anchor_feat_center, c1_center).item())

            if abs(feat_dist0_init - feat_dist1_init) < 0.005:
                mask_0_np, mask_1_np, _, _ = anchor_subspace_clustering(
                    g2_feats_norm, anchor_feats_norm, anchor_feat_center
                )
                mask_0 = torch.from_numpy(mask_0_np)
                mask_1 = torch.from_numpy(mask_1_np)
                c0_center = F.normalize(g2_feats_norm[mask_0].mean(dim=0, keepdim=True), p=2, dim=1)
                c1_center = F.normalize(g2_feats_norm[mask_1].mean(dim=0, keepdim=True), p=2, dim=1)

            indices_c0 = [g2_indices_map[i] for i in mask_0.nonzero(as_tuple=True)[0].tolist()]
            indices_c1 = [g2_indices_map[i] for i in mask_1.nonzero(as_tuple=True)[0].tolist()]

        handle.remove()

        cache_dir = './cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_tag = schedule.get('cache_tag', f"{d_name}_{a_name}_seed{self.seed}")
        ddn_path = os.path.join(cache_dir, f"{cache_tag}_ddn_results_v2.pth")

        hfe_map = torch.load(ddn_path) if os.path.exists(ddn_path) else {}
        missing_ddn = [idx for idx in g2_indices if idx not in hfe_map]
        if missing_ddn:
            ds_calc = utils.MatchSampleDataset(self.poisoned_trainset, missing_ddn)
            l_calc = DataLoader(ds_calc, batch_size=64, shuffle=False, num_workers=4)
            ddn_results = utils.analyze_batch_with_ddn(
                self.model, l_calc, self.device, num_classes=self.num_classes, ddn_steps=100
            )
            for i, res in enumerate(ddn_results):
                hfe_map[missing_ddn[i]] = res['freq_diff_pre_post_energy']
            torch.save(hfe_map, ddn_path)

        def get_avg_hfe(indices):
            vals = [hfe_map[i] if not isinstance(hfe_map[i], dict) else hfe_map[i]['freq_diff_pre_post_energy']
                    for i in indices if i in hfe_map]
            return float(np.mean(vals)) if vals else 0.0

        hfe_c0, hfe_c1 = get_avg_hfe(indices_c0), get_avg_hfe(indices_c1)
        dist0, dist1 = abs(hfe_c0 - hfe_anchor_val), abs(hfe_c1 - hfe_anchor_val)

        feat_dist0 = float(1.0 - F.cosine_similarity(anchor_feat_center, c0_center).item())
        feat_dist1 = float(1.0 - F.cosine_similarity(anchor_feat_center, c1_center).item())

        joint_dist0 = dist0 * max(feat_dist0, 1e-4)
        joint_dist1 = dist1 * max(feat_dist1, 1e-4)

        if joint_dist0 <= joint_dist1:
            candidate_indices, poison_cluster = indices_c0, indices_c1
            mean_clean, mean_poison = hfe_c0, hfe_c1
        else:
            candidate_indices, poison_cluster = indices_c1, indices_c0
            mean_clean, mean_poison = hfe_c1, hfe_c0

        self.log("[Phase 3] Adaptive Refinement...")

        poison_bias = mean_poison - mean_clean
        pgd_path = os.path.join(cache_dir, f"{cache_tag}_pgd_g2_results_v2.pth")
        pgd_map = torch.load(pgd_path) if os.path.exists(pgd_path) else {}

        missing = [idx for idx in candidate_indices if idx not in pgd_map]
        if missing:
            ds_pgd = utils.MatchSampleDataset(self.poisoned_trainset, missing)
            l_pgd = DataLoader(ds_pgd, batch_size=64, shuffle=False, num_workers=4)
            for img, lbl, idxs in tqdm(l_pgd, desc="PGD Probing", leave=False):
                steps_list = utils.pgd_attack_steps_batch(self.model, img, lbl, device=self.device)
                for i, oid in enumerate(idxs):
                    pgd_map[oid.item()] = steps_list[i]
            torch.save(pgd_map, pgd_path)

        if len(candidate_indices) > 0:
            hfe_arr = np.asarray([
                hfe_map[idx] if not isinstance(hfe_map[idx], dict) else hfe_map[idx]['freq_diff_pre_post_energy']
                for idx in candidate_indices
            ], dtype=np.float64)
            p_arr = np.asarray([pgd_map.get(idx, 0) for idx in candidate_indices], dtype=np.float64)
            epsilon = 1e-6

            lo_h, hi_h = np.percentile(hfe_arr, 2), np.percentile(hfe_arr, 98)
            if hi_h - lo_h < epsilon:
                lo_h, hi_h = hfe_arr.min(), hfe_arr.max()
            norm_h = np.clip((hfe_arr - lo_h) / (hi_h - lo_h + epsilon), 0.0, 1.0)

            lo_p, hi_p = np.percentile(p_arr, 2), np.percentile(p_arr, 98)
            if hi_p - lo_p < epsilon:
                lo_p, hi_p = p_arr.min(), p_arr.max()
            norm_p = np.clip((p_arr - lo_p) / (hi_p - lo_p + epsilon), 1e-3, 1.0)

            clean_h = norm_h if poison_bias < 0 else (1.0 - norm_h)
            r_score = norm_p

            scores = clean_h * r_score
            metrics = [
                (candidate_indices[i], float(scores[i]))
                for i in range(len(candidate_indices))
            ]
            metrics.sort(key=lambda x: x[1], reverse=True)

            drop_ratio = schedule.get('drop_ratio', 0.40)
            keep_count = max(1, int(len(metrics) * (1.0 - drop_ratio)))
            g2_refined = [x[0] for x in metrics[:keep_count]]
        else:
            g2_refined = candidate_indices

        all_train_indices = [i for i in range(len(self.poisoned_trainset)) if i not in g2_indices]
        trusted_indices = list(self.trusted_clean_set.indices) if hasattr(self.trusted_clean_set, 'indices') else []
        final_indices = list(dict.fromkeys(all_train_indices + g2_refined + trusted_indices))

        self.log("[Phase 4] Retraining Purified Model...")

        clean_ds = Subset(self.poisoned_retrainset, final_indices)

        self.model = model_factory().to(self.device)
        self.model = utils.train_clean_model(
            self.model, clean_ds, self.poisoned_testset, self.clean_testset,
            schedule['retrain_schedule'], "Retraining", self.log, self.device
        )
        torch.save(self.model.state_dict(), osp.join(self.work_dir, 'final_model.pth'))
        self.log("Defense Complete.")