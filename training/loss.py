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
    def __init__(self, alpha=0.25, gamma=2, class_weights=None, device=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.device = device

        if self.class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    def forward(self, outputs, targets):
        """
        Calculate Focal Loss
        :param outputs: predicted class scores, shape [batch_size, num_classes]
        :param targets: true class labels, shape [batch_size]
        """
        # Apply softmax to get the probability distributions
        probs = F.softmax(outputs, dim=-1)
        
        # Gather probabilities for the true class
        probs_true_class = probs.gather(1, targets.unsqueeze(1))
        
        # Compute the focal loss component
        loss = -self.alpha * (1 - probs_true_class) ** self.gamma * torch.log(probs_true_class)
        
        if self.class_weights is not None:
            # Apply class weights if specified
            loss = loss * self.class_weights[targets].unsqueeze(1)
        
        return loss.mean()  # Return the average loss for the batch


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





