import os
import os.path as osp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip
from tqdm import tqdm
from collections import Counter

from .base import Base
from ..utils.log import Log
import core.defenses.analysis_utils as utils
from .official_cul import train_official_cul_model

def kmeans_pytorch(X, num_clusters=2, max_iter=100, tol=1e-4, device='cuda'):
    if X.shape[0] < num_clusters:
        return torch.zeros(X.shape[0], dtype=torch.long, device=device), X

    indices = torch.randperm(X.size(0))[:num_clusters]
    centroids = X[indices].clone().to(device)
    X = X.to(device)

    for i in range(max_iter):
        dists = torch.cdist(X, centroids)
        labels = torch.argmin(dists, dim=1)

        new_centroids = []
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                c = X[mask].mean(dim=0)
                new_centroids.append(c)
            else:
                idx = torch.randint(0, X.size(0), (1,)).item()
                new_centroids.append(X[idx])

        new_centroids = torch.stack(new_centroids)
        if torch.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return labels, centroids


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, schedule, model_factory):
        work_dir = osp.join(schedule['save_dir'],
                            schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        self.log = Log(osp.join(work_dir, 'log.txt'))
        self.work_dir = work_dir

        d_name = schedule.get('dataset_name', 'Dataset')
        a_name = schedule.get('attack_name', 'Attack')

        self.log(f"========== EPAR Defense (Split Strategy: Blind CUL + Clean Anchor) ==========")
        self.log(f"Attack: {a_name} | Real Target: {self.target_class}")

        if 'device' in schedule and schedule['device'] == 'GPU':
            self.device = torch.device("cuda")
        self.model = self.model.to(self.device)

        if 'shots' not in schedule['cul_schedule']:
            raise ValueError("Config Error: 'shots' missing!")
        shots = schedule['cul_schedule']['shots']

        self.log(f"\n[Step 1] Selecting {shots} Blind Samples/Class & Training Official CUL...")
        if self.trusted_clean_set is not None:
            cul_subset = self.trusted_clean_set
            self.log(f"Exposure Data Size: {len(cul_subset)} (Trusted Clean Set)")
        else:
            cul_subset = utils.get_few_shot_clean_subset(
                self.poisoned_trainset,
                num_classes=self.num_classes,
                shots=shots,
                seed=666
            )
            self.log(f"Exposure Data Size: {len(cul_subset)} (Mixed Clean/Poison)")

        m_expo = train_official_cul_model(
            self.model, cul_subset, self.poisoned_testset, self.clean_testset,
            schedule['cul_schedule'], self.log, device=self.device
        )
        self.log("\n[Step 2] Inferring Target & Building Anchor...")

        m_expo.eval()
        infer_dataset = self.trusted_clean_set if self.trusted_clean_set is not None else self.clean_testset
        infer_loader = DataLoader(infer_dataset, batch_size=128, shuffle=False, num_workers=4)
        all_preds = []

        with torch.no_grad():
            for img, _ in infer_loader:
                img = img.to(self.device)
                preds = m_expo(img).argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)

        counts = Counter(all_preds)
        inferred_target = counts.most_common(1)[0][0]
        self.log(f"Inferred Target Class: {inferred_target}")

        if self.trusted_clean_set is not None:
            anchor_indices = []

            for i in range(len(self.trusted_clean_set)):
                _, label = self.trusted_clean_set[i]

                if int(label) == inferred_target:
                    anchor_indices.append(i)

            anchor_ds = Subset(self.trusted_clean_set, anchor_indices)
        else:
            anchor_indices = utils.get_clean_samples_of_target(
                self.poisoned_trainset,
                target_class=inferred_target,
                count=50
            )

            if len(anchor_indices) == 0:
                self.log("[Error] Could not find any clean samples for anchor! (Check logic)")
                anchor_indices = [0, 1, 2]

            anchor_ds = utils.MatchSampleDataset(self.poisoned_trainset, anchor_indices)
        anchor_loader = DataLoader(anchor_ds, batch_size=32, shuffle=False, num_workers=4)

        self.log("Calculating DDN HFE for Anchor...")
        anchor_ddn_res = utils.analyze_batch_with_ddn(
            self.model, anchor_loader, self.device, num_classes=self.num_classes, ddn_steps=100
        )

        anchor_hfe_values = [res['freq_diff_pre_post_energy'] for res in anchor_ddn_res]
        hfe_anchor_val = np.mean(anchor_hfe_values)
        self.log(f">>> Clean Anchor HFE Diff: {hfe_anchor_val:.4f}")

        self.log("\n[Step 3] Splitting & Clustering...")

        def get_preds_full(model, ds):
            model.eval()
            l = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
            preds, labels = [], []
            with torch.no_grad():
                for img, label in tqdm(l, desc="Splitting"):
                    img = img.to(self.device)
                    preds.append(model(img).argmax(1).cpu())
                    labels.append(label.cpu())
            return torch.cat(preds), torch.cat(labels)

        p_expo, true_labels = get_preds_full(m_expo, self.poisoned_trainset)
        g2_indices = []

        for i in range(len(self.poisoned_trainset)):
            if p_expo[i].item() == true_labels[i].item():
                g2_indices.append(i)

        self.log(f"G2 Size: {len(g2_indices)}")

        label_counts = torch.bincount(true_labels, minlength=self.num_classes)
        normal_class_counts = [int(label_counts[c]) for c in range(self.num_classes) if c != inferred_target]
        estimated_class_size = int(np.median(normal_class_counts))
        self.log(f"Estimated Normal Class Size: {estimated_class_size}")

        features_buffer = {}

        def get_hook(name):
            def hook(model, input, output):
                features_buffer[name] = input[0].detach()

            return hook

        target_layer = None
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Linear):
                target_layer = module
                break
        handle = target_layer.register_forward_hook(get_hook('feat'))

        indices_c0, indices_c1 = [], []
        if len(g2_indices) > 0:
            ds_g2 = utils.MatchSampleDataset(self.poisoned_trainset, g2_indices)
            loader_g2 = DataLoader(ds_g2, batch_size=128, shuffle=False, num_workers=4)

            g2_feats_list = []
            g2_indices_map = []

            self.model.eval()
            with torch.no_grad():
                for imgs, _, indices in tqdm(loader_g2, desc="Extracting Features"):
                    imgs = imgs.to(self.device)
                    _ = self.model(imgs)
                    g2_feats_list.append(features_buffer['feat'].cpu())
                    g2_indices_map.extend(indices.tolist())

            g2_feats = torch.cat(g2_feats_list)
            g2_feats_norm = F.normalize(g2_feats, p=2, dim=1)

            labels, _ = kmeans_pytorch(g2_feats_norm, num_clusters=2, device=self.device)

            mask_0 = (labels == 0)
            mask_1 = (labels == 1)

            indices_c0 = [g2_indices_map[i] for i in mask_0.nonzero(as_tuple=True)[0].tolist()]
            indices_c1 = [g2_indices_map[i] for i in mask_1.nonzero(as_tuple=True)[0].tolist()]

        handle.remove()
        self.log(f"Cluster 0: {len(indices_c0)} | Cluster 1: {len(indices_c1)}")

        self.log("\n[Step 4] Identifying Clusters (Anchor vs C0/C1)...")

        cache_dir = './cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_tag = schedule.get('cache_tag', f"{d_name}_{a_name}")
        ddn_cache_name = f"{cache_tag}_ddn_results.pth"
        ddn_path = os.path.join(cache_dir, ddn_cache_name)

        hfe_map = {}
        if os.path.exists(ddn_path):
            self.log(f"Loading HFE Cache: {ddn_path}")
            hfe_map = torch.load(ddn_path)
        else:
            self.log("Computing HFE for G2...")
            ds_calc = utils.MatchSampleDataset(self.poisoned_trainset, g2_indices)
            l_calc = DataLoader(ds_calc, batch_size=64, shuffle=False, num_workers=4)

            ddn_results = utils.analyze_batch_with_ddn(
                self.model, l_calc, self.device, num_classes=self.num_classes, ddn_steps=100
            )
            for i, res in enumerate(ddn_results):
                hfe_map[g2_indices[i]] = res['freq_diff_pre_post_energy']
            torch.save(hfe_map, ddn_path)

        def get_avg_hfe(indices):
            vals = [hfe_map[i] if not isinstance(hfe_map[i], dict) else hfe_map[i]['freq_diff_pre_post_energy']
                    for i in indices if i in hfe_map]
            return np.mean(vals) if vals else 0

        hfe_c0 = get_avg_hfe(indices_c0)
        hfe_c1 = get_avg_hfe(indices_c1)

        self.log(f"Anchor: {hfe_anchor_val:.2f} | C0: {hfe_c0:.2f} | C1: {hfe_c1:.2f}")
        dist0 = abs(hfe_c0 - hfe_anchor_val)
        dist1 = abs(hfe_c1 - hfe_anchor_val)

        if dist0 < dist1:
            candidate_indices = indices_c0
            poison_cluster = indices_c1
            mean_poison = hfe_c1
            mean_clean = hfe_c0
            self.log("Cluster 0 is closer to Anchor -> CLEAN.")
        else:
            candidate_indices = indices_c1
            poison_cluster = indices_c0
            mean_poison = hfe_c0
            mean_clean = hfe_c1
            self.log("Cluster 1 is closer to Anchor -> CLEAN.")

        self.log("\n[Step 5] Refinement (Directional)...")

        poison_bias = mean_poison - mean_clean
        self.log(f"Poison Bias (Poison - Clean): {poison_bias:.4f}")

        pgd_cache_name = f"{cache_tag}_pgd_g2_results.pth"
        pgd_path = os.path.join(cache_dir, pgd_cache_name)
        pgd_map = {}
        if os.path.exists(pgd_path):
            pgd_map = torch.load(pgd_path)

        missing = [idx for idx in candidate_indices if idx not in pgd_map]
        if missing:
            self.log(f"Computing PGD for {len(missing)} samples...")
            ds_pgd = utils.MatchSampleDataset(self.poisoned_trainset, missing)
            l_pgd = DataLoader(ds_pgd, batch_size=64, shuffle=False, num_workers=4)
            for img, lbl, idxs in tqdm(l_pgd, desc="PGD"):
                steps_list = utils.pgd_attack_steps_batch(self.model, img, lbl, device=self.device)
                for i, oid in enumerate(idxs):
                    pgd_map[oid.item()] = steps_list[i]
            torch.save(pgd_map, pgd_path)

        metrics = []
        hfe_vals = []
        for idx in candidate_indices:
            h = hfe_map[idx] if not isinstance(hfe_map[idx], dict) else hfe_map[idx]['freq_diff_pre_post_energy']
            hfe_vals.append(h)

        if len(hfe_vals) > 0:
            min_h, max_h = min(hfe_vals), max(hfe_vals)
            epsilon = 1e-6

            for idx in candidate_indices:
                h = hfe_map[idx] if not isinstance(hfe_map[idx], dict) else hfe_map[idx]['freq_diff_pre_post_energy']
                p = pgd_map.get(idx, 0)
                norm_h = (h - min_h) / (max_h - min_h + epsilon)
                metric = (1.0 - norm_h) * p
                metrics.append((idx, metric))

            metrics.sort(key=lambda x: x[1], reverse=True)

            DROP_RATIO = 0.40
            CLASS_SIZE_FACTOR = schedule.get('class_size_factor', 0.8)

            base_keep_count = int(len(metrics) * (1.0 - DROP_RATIO))
            class_prior_keep = int(estimated_class_size * CLASS_SIZE_FACTOR)
            keep_count = min(base_keep_count, class_prior_keep)
            keep_count = max(1, keep_count)
            drop_count = len(metrics) - keep_count

            self.log(
                f"Base Keep: {base_keep_count} | "
                f"Class-Prior Keep: {class_prior_keep} | "
                f"Final Keep: {keep_count}"
            )

            if poison_bias < 0:
                kept_items = metrics[drop_count:]
                self.log(f"Bias < 0 (Poison Low). Drop TOP (High Metric).")
            else:
                kept_items = metrics[:keep_count]
                self.log(f"Bias > 0 (Poison High). Drop BOTTOM (Low Metric).")

            g2_refined = [x[0] for x in kept_items]
        else:
            g2_refined = candidate_indices

        self.log(f"Final Kept G2: {len(g2_refined)} / {len(candidate_indices)}")

        if hasattr(self.poisoned_trainset, 'poisoned_set'):
            p_set = self.poisoned_trainset.poisoned_set
            g2_kept_p = sum(1 for i in g2_refined if i in p_set)
            self.log(f"[God's View] Poison Remaining in G2 Refined: {g2_kept_p}")
        all_train_indices = [i for i in range(len(self.poisoned_trainset)) if i not in g2_indices]  # G1
        final_indices = all_train_indices + g2_refined

        self.log(f"\n[Step 6] Retraining with {len(final_indices)} samples...")

        clean_ds = Subset(self.poisoned_retrainset, final_indices)

        self.model = model_factory().to(self.device)
        self.model = utils.train_clean_model(
            self.model, clean_ds, self.poisoned_testset, self.clean_testset,
            schedule['retrain_schedule'], "Retraining", self.log, self.device
        )
        torch.save(self.model.state_dict(), osp.join(self.work_dir, 'final_model.pth'))
        self.log("Defense Complete.")