# src/model/test.py

import torch    # Create tensors to store all numerical values (raw data + values for each weight, bias)
import torch.nn as nn   # make weights and biases part of the neural network
import torch.nn.functional as F # gives activation functions
from torch.optim import SGD     # Stochastic Gradient Descent, to fit neural network with data

# Draw Graphs
import matplotlib.pyplot as plt
import seaborn as sns


# creating a new neural network means creating a new class
class BasicNN(nn.Module):   # the class BasicNN will inherit from a PyTorch class called Module
    
    def __init__(self):   # initialisation method for the new class

        super().__init__()    # call initialisation method for the parent class nn.Module

        # Initialise weights and biases in the neural network
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False) # requires_grad=True when weight needs to be optimised
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    # The neural network parameters for each weight and bias have been created
    # Now these need to be connected to the input, the activation functions and the output

    def forward(self, input):   # second method: forward pass the initialised parameters

        input_to_top_relu = input * self.w00 + self.b00     # y = mx + c
        top_relu_output = F.relu(input_to_top_relu)     # pass input_to_top_relu with ReLU activation function
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output

# copy of first model but now for training the parameters
class BasicNN_train(nn.Module):   # the class BasicNN will inherit from a PyTorch class called Module
    
    def __init__(self):   # initialisation method for the new class

        super().__init__()    # call initialisation method for the parent class nn.Module

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True) ### DIFFERENCE ### this tells PyTorch to optimise the parameter


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


# tensor with 11 sequences (values between 0 and 1) is created using PyTorch function linspace
input_doses = torch.linspace(start=0, end=1, steps=11)

input_doses

model = BasicNN()   # Make the neural network 'model' from the class BasicNN which was created
# model is the standard variable name when using PyTorch

output_values = model(input_doses)  # pass the input values to the model

sns.set_style('whitegrid')

sns.lineplot(
    x=input_doses,
    y=output_values,
    color='green',
    linewidth=2.5
)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show(block=True)

model = BasicNN_train()

output_values = model(input_doses)

sns.set_style('whitegrid')

sns.lineplot(
    x=input_doses,

    # because the final_bias is now a gradient > detach() is called on output_values 
    # to create new tensor which only has the values
    y=output_values.detach(),
    color='green',
    linewidth=2.5
)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show(block=True)


#### Optimise Parameters form Training Data ####

# Training data
inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])    # observed output

optimiser = SGD(model.parameters(), lr=0.1)     # every parameter with requires_grad=True is passed through SGD

print("Final bias, before optimisation: " + str(model.final_bias.data) + "\n")  # str func converts tensor value to string so its printable

# Every time all the data points (all 3) from training data are optimised it is called an Epoch
for epoch in range(100):    # all 3 data points are run through 100 times
    
    total_loss = 0  # for each epoch total_loss keeps track of how well the model fits the data

    # nested forloop which runs each datapoint from training data through the model and calculates the total loss 
    for iteration in range(len(inputs)):    # range is same length as number of inputs
        
        # determine the input and effectiveness (label) for each datapoint
        input_i = inputs[iteration]
        label_i = labels[iteration]

        # runs the datapoint's input through the model
        output_i = model(input_i)

        # calculate loss between known and predicted value with loss function
        # squared residual (residual = output - known)
        # can use any other loss func as well; MSELoss(), CrossEntropyLoss() come with PyTorch
        loss = (output_i - label_i)**2

        # Backprop
        # calculate the Derivative of the loss func with respect to the parameter(s) which need to be optimised
        loss.backward()  # adds new derivatives on top of derivative after each point > as model keeps track of the derivative

        # add squared residual to total loss
        total_loss += float(loss)

    # check if total loss is really small
    # print out num of epoch and break out of the optimisation loop (stop training)
    if (total_loss < 0.0001):
        print("Num steps: " + str(epoch))
        break
    # otherwise take small step towards better value for final_bias
    optimiser.step()    # like loss.backwards, it also has access to the derivatives stored in the model
    optimiser.zero_grad()   # zero out derivatives stored in the model, otherwise next nested loop will accumulate derivatives from previous

    # print currect epoch and current value for final_bias
    print("Step: " + str(epoch) + " Final bias: " + str(model.final_bias.data) + "\n")

    output_values = model(input_doses)

    sns.set_style('whitegrid')

    sns.lineplot(
        x=input_doses,
        y=output_values.detach(),
        color='green',
        linewidth=2.5
    )

    plt.ylabel('Effectiveness')
    plt.xlabel('Dose')
    plt.show()

# print out the final value for the final bias
print("Final bias, after optimisation: " + str(model.final_bias.data))

