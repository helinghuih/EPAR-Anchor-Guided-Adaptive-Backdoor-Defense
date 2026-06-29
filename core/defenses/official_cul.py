import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import copy
import logging


# 简单的日志包装，兼容您现有的 logger
def get_logger():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)


class OfficialCUL:
    """
    Direct port of the official CUL implementation from 'Expose Before You Defend'.
    Source: exposes/unlearn.py
    """

    def __init__(self, model, defense_loader, test_loader, p_test_loader, device='cuda'):
        self.net = model
        self.defense_loader = defense_loader
        self.clean_test_loader = test_loader
        self.bad_test_loader = p_test_loader
        self.device = device

        # 官方默认参数 (Hardcoded from official arguments method)
        self.lr = 0.00001  # 1e-5
        self.epochs = 60  # Default unlearn_epochs
        self.sched_ms = [20, 20]  # Milestones
        self.sched_gamma = 0.1
        self.weight_decay = 5e-4

        # 记录器
        self.logger = get_logger()

    def init_defense_utils(self):
        # [关键差异 1] 官方使用 Adam 而不是 SGD
        optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        sched = lr_scheduler.MultiStepLR(optimizer, self.sched_ms, gamma=self.sched_gamma)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        return optimizer, sched, criterion

    def acc_test(self, loader):
        self.net.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.net(images)
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum().item()
                total_samples += len(labels)
        return total_correct / total_samples if total_samples > 0 else 0

    def train(self):
        print(f"\n[Official CUL] Starting training with Adam(lr={self.lr}) for {self.epochs} epochs...")

        optimizer, sched, criterion = self.init_defense_utils()
        self.net.train()

        for epoch in range(1, self.epochs + 1):
            total_loss = 0

            for i, (images, labels) in enumerate(self.defense_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                output = self.net(images)
                loss = criterion(output, labels)

                # [关键差异 2] 梯度裁剪 max_norm=20
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)

                # [核心逻辑] 梯度上升 (Gradient Ascent)
                (-loss).backward()

                optimizer.step()
                total_loss += loss.item()

            sched.step()

            # Periodic Evaluation
            if epoch % 5 == 0 or epoch == self.epochs:
                acc = self.acc_test(self.clean_test_loader)
                asr = self.acc_test(self.bad_test_loader)
                print(
                    f"CUL Epoch {epoch}/{self.epochs} | Loss: {total_loss / len(self.defense_loader):.4f} | ASR: {asr * 100:.2f}% | Acc: {acc * 100:.2f}%")

        return self.net


# 对外接口函数，适配您的 epar.py 调用习惯
def train_official_cul_model(model_init, dataset, p_test, clean_test, schedule, log=None, device='cuda'):
    """
    Wrapper function to fit into EPAR workflow.
    Ignores 'schedule' dictionary for hyperparameters to strictly follow official defaults.
    """
    # 构造 DataLoader
    defense_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    clean_test_loader = DataLoader(clean_test, batch_size=128, shuffle=False, num_workers=4)
    bad_test_loader = DataLoader(p_test, batch_size=128, shuffle=False, num_workers=4)

    # 深度拷贝模型
    model = copy.deepcopy(model_init).to(device)

    # 初始化官方实现类
    cul_solver = OfficialCUL(model, defense_loader, clean_test_loader, bad_test_loader, device)

    # 覆盖 Epoch 设置 (如果在 schedule 里指定了的话，否则用默认 20)
    if 'epochs' in schedule:
        cul_solver.epochs = schedule['epochs']

    # 开始训练
    exposed_model = cul_solver.train()

    return exposed_model