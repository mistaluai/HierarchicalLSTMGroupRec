from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

class b1_ModelTrainer:
    def __init__(self, model, optimizer, criterion, epochs, dataloaders, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.DEVICE = device

    def train_model(self):
        model, optimizer, criterion, epochs, dataloaders = self.model, self.optimizer, self.criterion, self.epochs, self.dataloaders
        for epoch in range(epochs):
            ## change model mode depending on the phase
            for phase in ['train', 'val']:
                dataloader = dataloaders[phase]
                epoch_loss = 0  # Track total loss for the epoch
                if phase == 'train':
                    model.train()
                    for inputs, labels in tqdm(dataloader, desc=phase):
                        inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
                        # zero grads of he optim
                        optimizer.zero_grad()
                        # freeze the non-learnable weights
                        self.__handle_transfer_learning(phase, epoch // epochs)
                        # forward pass
                        logit = model(inputs)
                        loss = criterion(logit, labels)
                        loss.backward()
                        # update weights
                        optimizer.step()
                        epoch_loss += loss.item()  # Accumulate loss
                else:
                    # skip evaluation if no suitable dataloader
                    if dataloaders[phase] is None:
                        continue
                    model.eval()
                    self.__eval_model(dataloader)
                print(f"Epoch {epoch + 1}/{epochs}, {phase} Loss: {epoch_loss / len(dataloader)}")  # Print loss

    def __handle_transfer_learning(self, phase, ratio_epochs, tl_coeff=0.8):
        if phase == "train":
            if ratio_epochs >= tl_coeff:
                # Unfreeze all layers for fine-tuning
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                # Freeze the CNN part
                for param in self.model.parameters():
                    param.requires_grad = False
                # Unfreeze the classification layer
                for param in self.model.fc.parameters():
                    param.requires_grad = True
        elif phase == "val":
            for param in self.model.parameters():
                param.requires_grad = False
    def __eval_model(self, dataloader):
        raise NotImplementedError('Not implemented yet')