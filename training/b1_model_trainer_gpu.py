class b1_ModelTrainer:
    def __init__(self, model, optimizer, criterion, epochs, dataloaders, device, save_folder, is_continue=False, checkpoint=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.DEVICE = device
        self.save_folder = save_folder
        self.is_continue = is_continue
        self.checkpoint = checkpoint

    def train_model(self):
        model, optimizer, criterion, epochs, dataloaders = self.model, self.optimizer, self.criterion, self.epochs, self.dataloaders

        epoch = 0
        if self.is_continue:
            epoch, model, optimizer = self.__load_checkpoint(model, optimizer, self.checkpoint)

        for training_epoch in range(epoch, epochs):
            print(f"\nTraining epoch {training_epoch}: training {'full model' if self.__check_transfer_learning(training_epoch/epochs) else 'only fc model'}")
            ## change model mode depending on the phase
            for phase in ['train', 'val']:
                dataloader = dataloaders[phase]
                epoch_loss = 0  # Track total loss for the epoch
                if phase == 'train':
                    model.train()
                    for inputs, labels in tqdm(dataloader, desc=phase):
                        inputs = inputs.to(self.DEVICE)
                        labels = labels.to(self.DEVICE)
                        # zero grads of he optim
                        optimizer.zero_grad()
                        # freeze the non-learnable weights
                        self.__handle_transfer_learning(phase, training_epoch / epochs)
                        # forward pass
                        logit = model(inputs)
                        loss = criterion(logit, labels)
                        loss.backward()
                        # update weights
                        optimizer.step()
                        epoch_loss += loss.item()  # Accumulate loss
                    print(
                        f"Epoch {training_epoch + 1}/{epochs}, {phase} Loss: {epoch_loss / len(dataloader)}")  # Print loss
                else:
                    # skip evaluation if no suitable dataloader
                    if dataloaders[phase] is None:
                        continue
                    model.eval()
                    loss, acc = self.__eval_model(dataloader)
                    print(f"Epoch {training_epoch + 1}/{epochs}, ({phase}) Loss: {loss} | Accuracy: {acc}")  # Print loss


            self.__save_checkpoint(training_epoch, model.state_dict(), optimizer.state_dict())

        self.__save_model()

    def __handle_transfer_learning(self, phase, ratio_epochs, tl_coeff=0):
        if phase == "train":
            if self.__check_transfer_learning(ratio_epochs, tl_coeff):
                # Unfreeze all layers for fine-tuning
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                # Freeze the CNN part
                for param in self.model.parameters():
                    param.requires_grad = False
                # Unfreeze the classification layer
                for param in self.model.get_fc().parameters():
                    param.requires_grad = True
        elif phase == "val":
            for param in self.model.parameters():
                param.requires_grad = False

    def __check_transfer_learning(self, ratio_epochs, tl_coeff=0):
        return ratio_epochs >= tl_coeff

    def __eval_model(self, dataloader):
        model = self.model
        criterion = self.criterion
        model.eval()
        val_loss = 0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs = inputs.to(self.DEVICE)
                labels = labels.to(self.DEVICE)

                # Forward pass
                logits = model(inputs)
                probs = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
                loss = criterion(logits, labels)
                val_loss += loss.item()  # Accumulate loss

                # Compute accuracy
                predicted = torch.argmax(probs, dim=1)  # Get the class with the highest probability
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        # Calculate average loss and accuracy
        avg_loss = val_loss / len(dataloader)
        accuracy = correct_preds / total_preds
        return avg_loss, accuracy

    def __save_model(self):
        torch.save(self.model.state_dict(), self.save_folder + "/b1_model.pth")

    def __save_checkpoint(self, epoch, model_state_dict, optimizer_state_dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
        }
        torch.save(checkpoint, self.save_folder + f'/checkpoint-epoch{epoch}.pth')

    def __load_checkpoint(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        model = model.load_state_dict(model_state_dict)
        optimizer = optimizer.load_state_dict(optimizer_state_dict)
        return epoch, model, optimizer