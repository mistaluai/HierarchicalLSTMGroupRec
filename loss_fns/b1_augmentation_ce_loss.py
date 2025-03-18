from loss_fns.b1_weighted_cross_entropy import WeightedCrossEntropyLoss
import torch.nn.functional as F

class AugmentationCELoss(WeightedCrossEntropyLoss):
    def __init__(self, dataset, device, num_classes):
        super().__init__(dataset, device, num_classes)

    def forward(self, logit, target):
        return F.cross_entropy(logit, target, weight=self.weights)