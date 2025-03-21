import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import v2


class AugmentationDataLoader():
    def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory, num_classes, mixup_alpha=1.0, cutmix_alpha=1.0):
        mixup_alpha = float(mixup_alpha)
        cutmix_alpha = float(cutmix_alpha)

        cutmix = v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha)
        mixup = v2.MixUp(num_classes=num_classes, alpha=mixup_alpha)
        self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        self.loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, collate_fn=self.collate_fn)

    def get_loader(self):
        return self.loader

    def collate_fn(self, batch):
        inputs, labels = default_collate(batch)
        return self.cutmix_or_mixup(inputs, labels)