import torch
from torch import nn
from torch.utils.data import Dataset
import os

class Trainer():
    def __init__(
        self,
        model: nn.Module,
        loss_function,
        optimizer,
        label_vocab: dict,
        device,
        log_steps:int=500,
        log_level:int=2):
        
        """
        Args:
            - model: the model we want to train.
            - loss_function: the loss_function to minimize.
            - optimizer: the optimizer used to minimize the loss_function.
            - label_vocab (dictionary): vocabulary for the labels
            - log_steps (int): Number of iterations that we use to observe the loss function trend.
            - log_level (int): Always use to observe the loss function trend
        """
        
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device

        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level
        self.label_vocab = label_vocab

    def train(self, train_dataset:Dataset, 
              valid_dataset:Dataset, 
              epochs:int=1):
        
        """
        Args:
            - train_dataset: a Dataset or DatasetLoader instance containing
                             the training instances.
            - valid_dataset: a Dataset or DatasetLoader instance used to evaluate
                             learning progress.
            - epochs: the number of times to iterate over train_dataset.

        Returns:
            - avg_train_loss: the average training loss on train_dataset over epochs.
        """
        
        assert epochs > 1 and isinstance(epochs, int)
        train_loss = 0.0
        patience = 0
        best_loss = 0.0
        for epoch in range(epochs):
            if self.log_level > 0:
                print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train()

            for step, sample in enumerate(train_dataset):
                inputs = sample['inputs'].to(self.device)
                labels = sample['outputs'].to(self.device)
                chars = sample['char'].to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(inputs,chars)
               
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
               
                sample_loss = self.loss_function(predictions, labels)
                sample_loss.backward()
                self.optimizer.step()

                epoch_loss += sample_loss.tolist()

                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step, epoch_loss / (step + 1)))
            
            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            if self.log_level > 0:
                print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))

            valid_loss, patience, best_loss = self.evaluate(valid_dataset, patience, best_loss)
            if patience > 5:
                print(f"\033[No Improvement for 5 steps \033[0m")
                break
                

            if self.log_level > 0:
                print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))

        if self.log_level > 0:
            print('... Done!')
        
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss
    

    def evaluate(self, valid_dataset, patience ,best_loss):
        
        """
        Args:
            - valid_dataset: the dataset to use to evaluate the model.

        Returns:
            - avg_valid_loss: the average validation loss over valid_dataset.
        """
        
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in valid_dataset:
                inputs = sample['inputs'].to(self.device)
                labels = sample['outputs'].to(self.device)
                chars = sample['char'].to(self.device)

                predictions = self.model(inputs,chars)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                sample_loss = self.loss_function(predictions, labels)
                valid_loss += sample_loss.tolist()
        
        if (valid_loss / len(valid_dataset)) < best_loss:
            if not os.path.exists("models"):
                os.makedirs("models")
            best_loss = valid_loss / len(valid_dataset)
            patience = 0
            torch.save(self.model.state_dict(), 'models/best_model.pt')
            print(f"\033[Improvement perfomances, model saved\033[0m")
        else:
            patience += 1 

        return valid_loss / len(valid_dataset), patience, best_loss

    def predict(self, x):
        
        """
        Args:
            - x: a tensor of indices.
        
        Returns: 
            - A list containing the predicted NER tag for each token in the
            input sentences.
        """
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            predictions = torch.argmax(logits, -1)
            return logits, predictions