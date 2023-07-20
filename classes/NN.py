import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class NN(nn.Module):
    
    def __init__(self, layer_sizes, activation_functions = None):

        super(NN,self).__init__()
        
        # Initialize Variables
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.depth = len(self.layer_sizes)

        if self.activation_functions == None: 
            self.activation_functions = [None] * (self.depth - 1)
        
        # Create model
        layers = []
        for i in range(self.depth - 1):
            layers.append(nn.Linear(in_features = self.layer_sizes[i], out_features = self.layer_sizes[i+1]))
            if i != self.depth - 2:
                if self.activation_functions[i] == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif self.activation_functions[i] == 'relu':
                    layers.append(nn.ReLU())
        # Final model
        self.model = nn.Sequential(*layers)


    def forward(self, input):
        return self.model(input)

    # ONLY for LogReg
    def accuracy(self, test_set):
        x_test = test_set.tensors[0]
        y_test = test_set.tensors[1]

        probs = F.softmax(self.model(x_test), dim = -1)
        y_pred = torch.argmax(probs, axis = 1)
        y_test = torch.argmax(y_test, axis = 1)

        return round(torch.mean(torch.eq(y_pred, y_test).float()).item()*100,2)
    
    def train(self, dataset, regularization = None, criterion = 'mse', optimizer = 'bgd', epochs = 5, lr = 0.05):

        # Optimizer
        if optimizer == 'bgd': # Batch Gradient Descent
            optimizer = optim.SGD(self.model.parameters(), lr = lr)

        # Criterion
        if criterion == 'mse': # Mean Square Error
            criterion = nn.MSELoss()
        elif criterion == 'cross_entropy': # Cross Entropy Loss
            criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for data in dataset:

                # Reset Gradients
                optimizer.zero_grad()

                # Get feature and target
                x_data, y_data = data

                # Forward Pass
                y_pred =  self.model(x_data)
                
                # Regularization
                if regularization == 'l2': 
                    # Weight of reg
                    C = 0.01 
                    # Regularization term
                    reg = sum(torch.norm(param, p = 2) ** 2 for param in self.model.parameters())
                elif regularization == 'l1':
                    C = 0.01
                    reg = sum(torch.norm(param, p = 2) ** 2 for param in self.model.parameters())
                else:
                    C = 0
                    reg = torch.tensor(0)

                # Loss Function
                loss = criterion(y_pred, y_data) + C * reg
                loss.backward()

                # Upgrade Weights and Biases
                optimizer.step()

