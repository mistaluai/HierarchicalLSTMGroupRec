import torch
from collections import Counter

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, dataset, device):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.dataset = dataset
        self.device = device
        weight = self.__compute_weights()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def __compute_weights(self):
        print('Computing class weights...')
        label_counts = Counter([label.item() for _, label in self.dataset])
        total_samples = len(self.dataset)

        class_weights = [total_samples / label_counts[i] if i in label_counts else 0 for i in range(8)]

        weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        print('Class weights computed.')
        return weights

    def forward(self, logit, target):
        return self.loss(logit, target)