import numpy as np
from scipy.special import softmax

class LogisticRegression:
    def __init__(self, inp_size, out_size):
        # Initialize size of input and output which will be the number of neurons in layer
        # Iniaialize randomly the weights and bias
        self.inp_size = inp_size
        self.out_size = out_size
        self.W = np.random.rand(inp_size,out_size)
        self.b = np.random.rand(1,out_size)

    def forward_pass(self, input):
        '''Calculate forward pass'''

        # Apply the linear transformation with bias and apply non-linear softmax along lines
        return softmax(np.dot(input, self.W) + self.b, axis = 1)

    def back_prop(self, input, output):
        '''Calculate gradients'''

        # When selecting batch from original, if batch size is 1, it returns a vector (784,) which does not allow for matrix multiplication
        if input.ndim == 1:
            input = input.reshape(1,784)
            output = output.reshape(1,1)


        # Forward pass of input
        z_tilde = self.forward_pass(input)

        # Initialize the gradients
        grad_W =  np.zeros((self.inp_size,self.out_size))
        grad_b = np.zeros((1,self.out_size))

        # Batch size
        batch_size = input.shape[0]

        # Go over each input and calculate its contribution to the gradients
        for i in range(batch_size):
            # I is the expected output
            I = np.array([0 if j != output[i] else 1 for j in range(self.out_size)])
            # Update gradients using the formulas
            grad_W += np.outer(input[i], z_tilde[i] - I)
            grad_b += z_tilde[i] - I

        # Finish the mean by dividing over the batch_size
        grad_W /= batch_size
        grad_b /= batch_size

        return grad_W, grad_b

    
    def train(self, input, output, epochs, batch_size):

        # Perform the number of epochs
        for _ in range(epochs):
            # Choose batch_size indexes from the input data
            n_inputs = input.shape[0]
            indexes = np.random.choice(n_inputs, batch_size, replace=False)

            # Select them from the input and ouput
            batch = input[indexes]  
            batch_out = output[indexes]     

            # Preform gradient descent
            grad_W, grad_b = self.back_prop(batch, batch_out)

            self.W -= grad_W
            self.b -= grad_b

    def classify(self, input):

        # Perform forward pass for prediction and then choose the one with maximum probability
        fp = self.forward_pass(input)
        return np.argmax(fp, axis = 1)
    
