import torch
from Utilities.Utilities import Utilities


class TrainerClassifier:

    def __init__(self, hyperparameter):
        self.hyperparameters = hyperparameter

    def set_model(self, model, device):
        self.model = model
        self.device = device

    def set_scope(self, scope):
        self.scope = scope

    def set_data(self, x_train, y_train, x_valid, y_valid):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def run(self):
        train_accuracy_dict = {}
        valid_accuracy_dict = {}

        for epoch in range(self.hyperparameters["max_epoch"]):
            # Train
            self.model.train()
            total_loss = 0.0
            total_accuracy = 0.0
            n_batch = len(self.x_train)

            for n in range(n_batch):
                x = self.x_train[n].to(self.device)
                y = self.y_train[n].to(self.device)
                # Forward pass
                y_hat = self.model(x)
                loss = self.scope.criterion(y_hat, y)
                # Backward propagation
                self.scope.optimizer.zero_grad()
                loss.backward()
                self.scope.optimizer.step()

                total_loss += loss.item()
                # Compute accuracy
                batch_accuracy = Utilities.compute_accuracy(y, y_hat)
                total_accuracy += batch_accuracy

            train_loss = total_loss / n_batch
            training_accuracy = total_accuracy / n_batch

            print("Epoch: " + str(epoch + 1) + "/" + str(self.hyperparameters["max_epoch"]))
            print("Training loss: " + str(train_loss) + " - Training accuracy: " + str(training_accuracy))

            train_accuracy_dict[epoch] = training_accuracy

            # Validation
            self.model.eval()

            total_loss = 0.0
            total_accuracy = 0.0

            n_batch = len(self.x_valid)
            for n in range(n_batch):
                x = self.x_valid[n].to(self.device)
                y = self.y_valid[n].to(self.device)
                # Forward pass
                y_hat = self.model(x)
                loss = self.scope.criterion(y_hat, y)
                total_loss += loss.item()
                # Compute accuracy
                batch_accuracy = Utilities.compute_accuracy(y, y_hat)
                total_accuracy += batch_accuracy

            valid_loss = total_loss / n_batch
            valid_accuracy = total_accuracy / n_batch

            print("Epoch: " + str(epoch + 1) + "/" + str(self.hyperparameters["max_epoch"]))
            print("Validation loss: " + str(valid_loss) + " - Validation accuracy: " + str(valid_accuracy))

            valid_accuracy_dict[epoch] = valid_accuracy

            if self.scope.scheduler:
                validation_metric = valid_accuracy
                old_lr = self.scope.optimizer.param_groups[0]['lr']
                self.scope.scheduler.step(validation_metric)
                new_lr = self.scope.optimizer.param_groups[0]['lr']

                if old_lr != new_lr:
                    print(
                        f'Learning rate changed from {old_lr} to {new_lr} at epoch {epoch}'
                    )

            # Early stopper
            if self.scope.early_stopper:
                validation_metric = valid_accuracy
                self.model, keep_training = self.scope.early_stopper.set(
                    model=self.model,
                    epoch=epoch,
                    metric_epoch=validation_metric
                )

                if not keep_training:
                    break

        train_accuracy_list = [train_accuracy_dict[e] for e in train_accuracy_dict.keys()]
        valid_accuracy_list = [valid_accuracy_dict[e] for e in valid_accuracy_dict.keys()]

        return train_accuracy_list, valid_accuracy_list
