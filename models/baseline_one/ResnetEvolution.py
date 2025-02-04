import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torchvision
from tqdm import tqdm

## criterion, optim, lr, GPU/TPU
class ResnetEvolution(nn.Module):
    def __init__(self, optim, lr, device, is_tpu=False, hidden_layers=[128, 64, 32]):
        super(ResnetEvolution, self).__init__()
        self.optimizer = optim(self.parameters(), lr=lr)
        self.hidden_layers = hidden_layers
        self.model = self.__init_backbone(torchvision.models.resnet50(pretrained=True))
        self.is_tpu = is_tpu
        self.DEVICE = device

    def __init_backbone(self, backbone):
        num_features = backbone.fc.in_features

        layers = []
        input_size = num_features  # Start with backbone output size
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # Activation function
            input_size = hidden_size  # Update input for next layer

        layers.append(nn.Linear(input_size, 8))  # Final output layer

        backbone.fc = nn.Sequential(*layers)  # Output layer for binary classification

        return backbone

    def train_model(self, criterion, epochs, dataloaders):
        model = self.model
        optimizer = self.optimizer
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
                        self.__optimizer_step(optimizer)
                        epoch_loss += loss.item()  # Accumulate loss

                else:
                # skip evaluation if no suitable dataloader
                    if dataloaders[phase] is None:
                        continue
                    model.eval()
                    self.__eval_model(dataloader)

                print(f"Epoch {epoch + 1}/{epochs}, {phase} Loss: {epoch_loss / len(dataloader)}")  # Print loss

    def __optimizer_step(self, optimizer):
        if self.is_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

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
    def forward(self, images):
        return self.model(images)