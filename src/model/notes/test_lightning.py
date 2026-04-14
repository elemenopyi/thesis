# src/model/test_lightning.py

import torch    # Create tensors to store all numerical values (raw data + values for each weight, bias)
import torch.nn as nn   # make weights and biases part of the neural network
import torch.nn.functional as F # gives activation functions
from torch.optim import SGD     # Stochastic Gradient Descent, to fit neural network with data

import lightning as L   # to make training easier to code
from torch.utils.data import TensorDataset, DataLoader  # makes it easier when working with larger datasets (how?)

# Draw Graphs
import matplotlib.pyplot as plt
import seaborn as sns


# creating a new neural network means creating a new class
# class BasicLightning(L.LightningModule):   # the class BasicLightning will inherit from a lightning module instead of PyTorch module
    
#     def __init__(self):   # initialisation method for the new class

#         super().__init__()    # call initialisation method for the parent class LightningModule

#         # Initialise weights and biases in the neural network
#         self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False) # requires_grad=True when weight needs to be optimised
#         self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
#         self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
#         self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
#         self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
#         self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

#         self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

#     # The neural network parameters for each weight and bias have been created
#     # Now these need to be connected to the input, the activation functions and the output

#     def forward(self, input):   # second method: forward pass the initialised parameters

#         input_to_top_relu = input * self.w00 + self.b00     # y = mx + c
#         top_relu_output = F.relu(input_to_top_relu)     # pass input_to_top_relu with ReLU activation function
#         scaled_top_relu_output = top_relu_output * self.w01

#         input_to_bottom_relu = input * self.w10 + self.b10
#         bottom_relu_output = F.relu(input_to_bottom_relu)
#         scaled_bottom_relu_output = bottom_relu_output * self.w11

#         input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

#         output = F.relu(input_to_final_relu)

#         return output

# # copy of first model but now for training the parameters
# class BasicLightningTrain(L.LightningModule):   # the class BasicLightning will inherit from a lightning module instead of PyTorch module
    
#     def __init__(self):   # initialisation method for the new class

#         super().__init__()    # call initialisation method for the parent class BasicLightning

#         self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
#         self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
#         self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
#         self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
#         self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
#         self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

#         self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True) ### DIFFERENCE ### this tells PyTorch to optimise the parameter

#         ### NEW ###
#         # so that Lightning function improves the learning rate
#         # variable to store learning rate is created
#         self.learning_rate = 0.01

#     def forward(self, input):

#         input_to_top_relu = input * self.w00 + self.b00
#         top_relu_output = F.relu(input_to_top_relu)
#         scaled_top_relu_output = top_relu_output * self.w01

#         input_to_bottom_relu = input * self.w10 + self.b10
#         bottom_relu_output = F.relu(input_to_bottom_relu)
#         scaled_bottom_relu_output = bottom_relu_output * self.w11

#         input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

#         output = F.relu(input_to_final_relu)

#         return output

# ////////////////////////////// revise BasicLightning class to include new methods ////////////////////
class BasicLightningTrain(L.LightningModule):   # the class BasicLightning will inherit from a lightning module instead of PyTorch module
    
    def __init__(self):   # initialisation method for the new class

        super().__init__()    # call initialisation method for the parent class BasicLightning

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True) ### DIFFERENCE ### this tells PyTorch to optimise the parameter

        ### NEW ###
        # so that Lightning function improves the learning rate
        # variable to store learning rate is created
        self.learning_rate = 0.01

    def forward(self, input):

        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output
    
    # Set up method for optimisation of the neural network
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate)    # parameters with requires_grad is passed through SGD
        # with new learning_rate variable
    
    # Set up method for training step: 
    def training_step(self, batch, batch_idx):
        
        # take a batch of training data from the dataloader and the index for that batch
        input_i, label_i = batch
        output_i = self.forward(input_i)

        # calculate the loss (sum of the squared residuals)
        loss = (output_i - label_i)**2

        return loss


# tensor with 11 sequences (values between 0 and 1) is created using PyTorch function linspace
input_doses = torch.linspace(start=0, end=1, steps=11)
input_doses


#\/\/\/\/\/\/\/\/\/\/\/\/\/ Run model BasicLightning \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#
# model = BasicLightning()   # Make the neural network 'model' from the class BasicLightning which was created
# # model is the standard variable name when using PyTorch

# output_values = model(input_doses)  # pass the input values to the model

# sns.set_style('whitegrid')

# sns.lineplot(
#     x=input_doses,
#     y=output_values,
#     color='green',
#     linewidth=2.5
# )

# plt.ylabel('Effectiveness')
# plt.xlabel('Dose')
# plt.show(block=True)
#\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#


#\/\/\/\/\/\/\/\/\/\/\/\/\/ Run model BasicLightning_train \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#
# model = BasicLightningTrain()

# output_values = model(input_doses)

# sns.set(style='whitegrid')

# sns.lineplot(
#     x=input_doses,

#     # because the final_bias is now a gradient > detach() is called on output_values 
#     # to create new tensor which only has the values
#     y=output_values.detach(),
#     color='green',
#     linewidth=2.5
# )

# plt.ylabel('Effectiveness')
# plt.xlabel('Dose')
# plt.show(block=True)
#\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#


model = BasicLightningTrain()
# Training data
inputs = torch.tensor([0., 0.5, 1.] * 100)
labels = torch.tensor([0., 1., 0.] * 100)    # observed output

optimiser = SGD(model.parameters(), lr=0.1)     # every parameter with requires_grad=True is passed through SGD

print("Final bias, before optimisation: " + str(model.final_bias.data) + "\n")  # str func converts tensor value to string so its printable

# \\\\\\\\\\\\\\\\\\ NEW: Wrap training data in a DataLoader as Lightning is used \\\\\\\\\\\\\\\\
# DataLoaders are helpful when there is a lot of data:
    # 1. makes it easy to access data in batches
    # 2. makes it easy to shuffle the data each epoch
    # 3. makes it easy to use relatively small fraction of data for quick training (for debugging)

dataset = TensorDataset(inputs, labels)  # combine inputs and labels in a TensorDataset
dataloader = DataLoader(dataset)    # the tensor dataset is used to create a data loader
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    
# ***************** Run revised BasicLightningTrain model *************************************

# ===== model & trainer =======

trainer = L.Trainer(max_epochs=20)  # use lightning trainer to find good learning rate value then use it to optimise
# 34 because results from previous runs
# let lightning determine if GPU is avalaiable and how many by setting accelerator, devices to auto


tuner = L.pytorch.tuner.Tuner(trainer)
# find improved learning by calling tuner.lr_find and store in lr_find_results
    # feed lr_find with model, training data, minimum and maximum learning rate
    # tell lr_find to not stop early
lr_find_results = tuner.lr_find(model,
                                        train_dataloaders=dataloader,
                                        min_lr=0.001,
                                        max_lr=1.0,
                                        early_stop_threshold=None)
# access an improved learning rate by called suggestion()
new_lr = lr_find_results.suggestion() ## suggestion() returns the best guess for the optimal learning rate

print(f"lr_find() suggests {new_lr:.5f} for the learning rate.")

model.learning_rate = new_lr    # set learning rate variable in the model to new learning rate

# optimise using trainer and call the fit function (it requires the model and training data)
trainer.fit(model, train_dataloaders=dataloader)
# when fit is called, the trainer will call the model's configure_optimisers function
    # > then trainer calls the model's training_step function
    # > trainer also 
        # handles zero-ing the derivative after each epoch :: optimiser.zero_grad() 
        # calculates the new gradient :: loss.backward()
        # and takes step towards optimal parameter values :: optimiser.step()
    # then it calls training_step again for each epoch

print(model.final_bias.data)

output_values = model(input_doses)

sns.set(style="whitegrid")

sns.lineplot(x=input_doses,
             y=output_values.detach(),
             color='green',
             linewidth=2.5)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()










####\/\/\/\/\/\/ Old: Optimise Parameters form Training Data \/\/\/\/\/\/####

# Training data
# inputs = torch.tensor([0., 0.5, 1.])
# labels = torch.tensor([0., 1., 0.])    # observed output

# optimiser = SGD(model.parameters(), lr=0.1)     # every parameter with requires_grad=True is passed through SGD

# print("Final bias, before optimisation: " + str(model.final_bias.data) + "\n")  # str func converts tensor value to string so its printable

# # Every time all the data points (all 3) from training data are optimised it is called an Epoch
# for epoch in range(100):    # all 3 data points are run through 100 times
    
#     total_loss = 0  # for each epoch total_loss keeps track of how well the model fits the data

#     # nested forloop which runs each datapoint from training data through the model and calculates the total loss 
#     for iteration in range(len(inputs)):    # range is same length as number of inputs
        
#         # determine the input and effectiveness (label) for each datapoint
#         input_i = inputs[iteration]
#         label_i = labels[iteration]

#         # runs the datapoint's input through the model
#         output_i = model(input_i)

#         # calculate loss between known and predicted value with loss function
#         # squared residual (residual = output - known)
#         # can use any other loss func as well; MSELoss(), CrossEntropyLoss() come with PyTorch
#         loss = (output_i - label_i)**2

#         # Backprop
#         # calculate the Derivative of the loss func with respect to the parameter(s) which need to be optimised
#         loss.backward()  # adds new derivatives on top of derivative after each point > as model keeps track of the derivative

#         # add squared residual to total loss
#         total_loss += float(loss)

#     # check if total loss is really small
#     # print out num of epoch and break out of the optimisation loop (stop training)
#     if (total_loss < 0.0001):
#         print("Num steps: " + str(epoch))
#         break
#     # otherwise take small step towards better value for final_bias
#     optimiser.step()    # like loss.backwards, it also has access to the derivatives stored in the model
#     optimiser.zero_grad()   # zero out derivatives stored in the model, otherwise next nested loop will accumulate derivatives from previous

#     # print currect epoch and current value for final_bias
#     print("Step: " + str(epoch) + " Final bias: " + str(model.final_bias.data) + "\n")

#     output_values = model(input_doses)

#     sns.set_style('whitegrid')

#     sns.lineplot(
#         x=input_doses,
#         y=output_values.detach(),
#         color='green',
#         linewidth=2.5
#     )

#     plt.ylabel('Effectiveness')
#     plt.xlabel('Dose')
#     plt.show()

# # print out the final value for the final bias
# print("Final bias, after optimisation: " + str(model.final_bias.data))

