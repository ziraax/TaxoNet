import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLoss(nn.Module):
    def __init__(self, class_weights, device):
        super(WeightedLoss, self).__init__()
        self.device = device
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        else:
            self.class_weights = None

    def forward(self, outputs, targets):
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        return criterion(outputs, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, device=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, outputs, targets):
        eps = 1e-8  # for numerical stability
        probs = F.softmax(outputs, dim=-1)  # [B, C]
        probs_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        probs_true = probs_true.clamp(min=eps, max=1.0)

        # alpha can be scalar or class-specific
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]  # [B]
            else:
                alpha_t = self.alpha  # scalar
        else:
            alpha_t = 1.0

        loss = -alpha_t * (1 - probs_true) ** self.gamma * torch.log(probs_true)  # [B]
        return loss.mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        n_class = outputs.size(1)
        one_hot = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1)
        smooth_labels = one_hot * (1 - self.epsilon) + (1 - one_hot) * self.epsilon / (n_class - 1)

        log_probs = F.log_softmax(outputs, dim=1)
        loss = -torch.sum(smooth_labels * log_probs, dim=1)
        return loss.mean()





