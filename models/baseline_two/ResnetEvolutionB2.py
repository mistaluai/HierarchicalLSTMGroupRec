import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torchvision.models as models

import torch_xla.core.xla_model as xm
import torchvision
from tqdm import tqdm
from typing import Optional, Type
from dataclasses import dataclass


@dataclass
class ModelConfig:
    optim: Type[torch.optim.Optimizer]  # Expected optimizer class
    lr: float
    hidden_layers: int
    device: Optional[torch.device] = None
    is_tpu: bool = False
    pretrained: bool = True
    
    
## criterion, optim, lr, GPU/TPU
class ResnetEvolution(nn.Module):
    def __init__(self, config: ModelConfig):
        super(ResnetEvolution, self).__init__()
        self.optimizer = config.optim(self.parameters(), lr=config.lr)
        self.hidden_layers = config.hidden_layers
        self.backbone = models.resnet50(pretrained=config.pretrained)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Adaptive pooling to keep feature size consistent
        self.classifer = self.__init_classifier()

        self.is_tpu = config.is_tpu
        self.DEVICE = config.device or (xm.xla_device() if config.is_tpu else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __init_classifier(self):

        layers = []
        input_size = self.backbone.fc.in_features  # Start with backbone output size
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # Activation function
            input_size = hidden_size  # Update input for next layer

        layers.append(nn.Linear(input_size, 8))  # Final output layer

        self.classifer = nn.Sequential(*layers)  # Output layer for binary classification


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

        """
        Process a single frame containing multiple person images.
        
        Args:
            images (List[torch.Tensor]): List of cropped person images in a frame (each shape: [3, 224, 224])

        Returns:
            torch.Tensor: Softmax output for group activity classification
        """
        person_features = []
        for img in images:
            
            img = img.unsqueeze(0).to(self.device)  # Add batch dimension   #this preprocessing need to be transfered to the 
            features = self.feature_extractor(img)  # Extract deep features
            person_features.append(features.squeeze(0))  # Remove batch dimension

        person_features = torch.stack(person_features)  # Shape: (num_people, 4096)
        pooled_features = self.pooling(person_features.unsqueeze(0).transpose(1, 2)).squeeze()  # Apply average pooling

        return self.classifier(pooled_features)  # Classify group activity

            
        
