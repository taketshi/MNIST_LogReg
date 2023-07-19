# MNIST_LogReg
Simple implementation of Logistic Regression to the MNIST dataset. Attempted input generation.

## Description
On the jupyter notebook __main.ipynb__ I try a simple implementation of Logistic Regression on the MNIST dataset. It is also tested a slightly more complex NN with regularization. 

Finally, it is (unsuccessfuly) attempted to generate an input of a specific number which minimizes the loss function wrt the target number. I believe it failed because the NN is not learning what we consider to be the crucial parts of the numbers shape. Perhaps a sort of dimensionality reduction would be advantageous because it could remedy this.

## Files
__main.ipynb__: the main body\
__classes__: contains the .py files with the classes\
  &nbsp; &nbsp; __NN.py__: contains the general NN using pytorch. Check taketshi/NN_torch\
  &nbsp; &nbsp; __LogisticRegression.py__: contains a numpy implementation of a logistic regression with gradient descent\

## Dependencies 
- main.ipynb\
&nbsp; from classes.NN import NN\
&nbsp; import torch\
&nbsp; import torch.nn as nn\
&nbsp; import torch.nn.functional as F\
&nbsp; import matplotlib.pyplot as plt\
&nbsp; from keras.datasets import mnist


  
  
