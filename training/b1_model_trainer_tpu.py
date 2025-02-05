import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class b1_ModelTrainer:
    def __init__(self, model, optimizer,lr, criterion, epochs, datasets, batch_size, device, is_tpu=False, save_path='/kaggle/working/model.pt'):
        self.is_tpu = is_tpu
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.datasets = datasets
        self.batch_size = batch_size
        self.DEVICE = device
        self.lr = lr
        self.save_path = save_path

    # Start training processes
    def _mp_fn(self, rank, flags):
        torch.set_default_tensor_type('torch.FloatTensor')
        self.trainer_TPU()


    def train_on_TPU(self):
        FLAGS = {}
        xmp.spawn(self._mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

    def trainer_TPU(self):
        model, criterion, lr, epochs, datasets, batch_size = self.model, self.criterion,self.lr, self.epochs, self.datasets, self.batch_size

        device = xm.xla_device()
        model = model.to(device)
        optimizer = self.optimizer(model.parameters(), lr= lr * xm.xrt_world_size())

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets['train'],
            num_replicas=xm.xrt_world_size(), #number of cores
            rank=xm.get_ordinal(), #current core
            shuffle=True,
            drop_last=True
        )

        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets['val'],
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False,
            drop_last=False
        )

        train_dataloader = DataLoader(datasets['train'], batch_size=batch_size, sampler=train_sampler, num_workers=1)
        valid_dataloader = DataLoader(datasets['val'], batch_size=batch_size, sampler=valid_sampler, num_workers=1)


        num_train_steps = int(len(datasets['train']) / batch_size / xm.xrt_world_size() * epochs)
        xm.master_print(f'num_train_steps per TPU core = {num_train_steps}, world_size={xm.xrt_world_size()}')

        for epoch in range(epochs):
            parallel_loader = pl.ParallelLoader(train_dataloader, [device])
            self.train_model(optimizer, parallel_loader.per_device_loader(device))

            # Evaluate model after training each epoch
            parallel_loader = pl.ParallelLoader(valid_dataloader, [device])
            val_loss, val_acc = self.__eval_model(parallel_loader.per_device_loader(device))
            xm.master_print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.2f}%")

            # Save best model based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                xm.master_print(f"Saving best model with Validation Loss: {best_loss:.4f}")
                xm.save(model.state_dict(), self.save_path)

    def train_model(self, optimizer, dataloader):
        model, criterion, epochs = self.model, self.criterion, self.epochs
        for epoch in range(epochs):
            ## change model mode depending on the phase
            epoch_loss = 0  # Track total loss for the epoch
            model.train()
            for inputs, labels in tqdm(dataloader, desc='training'):
                inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)

                # zero grads of he optim
                optimizer.zero_grad()
                # freeze the non-learnable weights
                self.__handle_transfer_learning('train', epoch / epochs)
                # forward pass
                logit = model(inputs)
                loss = criterion(logit, labels)
                loss.backward()
                # update weights
                self.__optimizer_step(optimizer)
                epoch_loss += loss.item()  # Accumulate loss

            print(f"Epoch {epoch + 1}/{epochs}, training Loss: {epoch_loss / len(dataloader)}")  # Print loss


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
        model, criterion = self.model, self.criterion
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Compute accuracy
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = (correct / total) * 100
        return avg_loss, accuracy