import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


class OfficialCUL:

    def __init__(self, model, defense_loader, test_loader=None, p_test_loader=None, log=None, device='cuda'):
        self.net = model
        self.defense_loader = defense_loader
        self.clean_test_loader = test_loader
        self.bad_test_loader = p_test_loader
        self.log = log
        self.device = device

        self.lr = 1e-4
        self.epochs = 20
        self.sched_ms = [20, 20]
        self.sched_gamma = 0.1
        self.weight_decay = 5e-4
        self.max_unlearn_loss = 7.0

    def _log(self, msg):
        if self.log is not None:
            self.log(msg)
        else:
            print(msg)

    def init_defense_utils(self):
        optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        sched = lr_scheduler.MultiStepLR(optimizer, self.sched_ms, gamma=self.sched_gamma)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        return optimizer, sched, criterion

    def train(self):
        optimizer, sched, criterion = self.init_defense_utils()

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            total_loss = 0.0

            for images, labels in self.defense_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                output = self.net(images)
                loss = criterion(output, labels)
                (-loss).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
                total_loss += loss.item()

            sched.step()
            avg_loss = total_loss / len(self.defense_loader)
            if epoch >= 3 and avg_loss >= self.max_unlearn_loss:
                break

        return self.net


def train_official_cul_model(model_init, dataset, p_test, clean_test, schedule, log=None, device='cuda'):
    batch_size = schedule.get('batch_size', 128)
    num_workers = schedule.get('num_workers', 4)

    defense_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = copy.deepcopy(model_init).to(device)
    cul_solver = OfficialCUL(model, defense_loader, log=log, device=device)

    if 'lr' in schedule:
        cul_solver.lr = schedule['lr']
    if 'weight_decay' in schedule:
        cul_solver.weight_decay = schedule['weight_decay']
    if 'epochs' in schedule:
        cul_solver.epochs = schedule['epochs']
    if 'schedule' in schedule:
        cul_solver.sched_ms = schedule['schedule']
    if 'gamma' in schedule:
        cul_solver.sched_gamma = schedule['gamma']
    if 'max_unlearn_loss' in schedule:
        cul_solver.max_unlearn_loss = schedule['max_unlearn_loss']

    return cul_solver.train()