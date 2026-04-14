# src/model/test_lstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam    # Adam is lot like SGD, tends to find values faster

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

class LSTMbyHand(L.LightningModule):

    def __init__(self): # initialisation
        super().__init__()
        # Use Normal Distribution to generate random numbers
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # % long term memory to remember (forget gate)
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # % potential memory to remember (input gate)
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # potential long term memory (input gate)
        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # % potential memory to remember (output gate)
        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def lstm_unit(self, input_value, long_memory, short_memory):    # LSTM math
        
        # Forget Gate (calculate % long memory to remember)
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) +
                                              (input_value * self.wlr2) +
                                              self.blr1)
        
        # Input Gate (calculate % to remember + new potential long memory) 
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) +
                                                   (input_value * self.wpr2) +
                                                   self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) +
                                      (input_value * self.wp2) +
                                      self.bp1)
        
        # Update the long term memory
        updated_long_memory = ((long_memory * long_remember_percent) +
                               (potential_remember_percent * potential_memory))
        
        # Output Gate (calculate % to remember + new short memory)
        output_percent = torch.sigmoid((short_memory * self.wo1) +
                                       (input_value * self.wo2) +
                                       self.bo1)        
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        # Return updated memories
        return([updated_long_memory, updated_short_memory])
    
    def forward(self, input):   # forward pass
        
        long_memory = 0
        short_memory = 0
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)

        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)

        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)

        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory

    def configure_optimizers(self): # configure Adam optimiser
        
        return Adam(self.parameters())
    
    def training_step(self, batch, batch_idx):  # calculate loss and log training progress
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i)**2
        
        self.log("train_loss", loss)    # log the loss so it can be reviewed later
        # log method from lightning creates lightning_logs folder to store values to keep track
        # here the loss is being stored under the label train_loss

        # keep track of predictions for each input sequence type
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss
    
model = LSTMbyHand()

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

print("Compary B: Observed = 1, Predicted =",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

trainer = L.Trainer(max_epochs=2000)
trainer.fit(model, train_dataloaders=dataloader)

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

print("Compary B: Observed = 1, Predicted =",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

# use TensorBoard to draw graphs which tell what happened during training
# to start TensorBoard server past in terminal: tensorboard --logdir=lightning_logs/

path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path

trainer = L.Trainer(max_epochs=10000)
trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_best_checkpoint)

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

print("Compary B: Observed = 1, Predicted =",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

class LightningLSTM(L.LightningModule):
    
    def __init__(self):
        
        super().__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=1)

    def forward(self, input):
        
        input_trans = input.view(len(input), 1)

        lstm_out, temp = self.lstm(input_trans)

        prediction = lstm_out[-1]
        return prediction

    def configure_optimizers(self):
        
        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i)**2
        
        self.log("train_loss", loss)

        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss

model = LightningLSTM()

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

print("Compary B: Observed = 1, Predicted =",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)
trainer.fit(model, train_dataloaders=dataloader)

