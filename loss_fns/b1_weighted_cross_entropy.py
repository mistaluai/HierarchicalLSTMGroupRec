import torch
from collections import Counter

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, dataset, device, num_classes=8):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss(weight=self.__compute_weights(dataset).to(device))

    def __compute_weights(self, dataset):
        labels = torch.tensor(dataset.get_labels(), dtype=torch.long)
        class_counts = torch.bincount(labels, minlength=self.num_classes)
        total_samples = len(labels)
        weights = total_samples / (class_counts + 1e-6)  # Avoid div by zero
        return weights.float()

    def forward(self, logit, target):
        return self.loss(logit, target)