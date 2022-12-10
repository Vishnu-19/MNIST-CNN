import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
       #Fully Connected Layers
      
       #Model 1
        if mode == 1:
            self.fc1=nn.Linear(28*28,100)  #input size = 28*28
            self.fc2=nn.Linear(100,10)    #Final Output Layer
       
       #Model 2     
        elif mode == 2 or mode == 3:          # Same FC layer for mode 3
            self.fc1=nn.Linear(60*4*4,100)  # in = 60 out channels with 4*4 size , out = 100
            self.fc2=nn.Linear(100,10)      #Final Output Layer

        #Model 4
        elif mode == 4 :
            self.fc1=nn.Linear(60*4*4,100)  #in = 60 out channels with 4*4 size, out = 100
            self.fc2=nn.Linear(100,100)     # Additional layer with 100 neurons
            self.fc3=nn.Linear(100,10)      #Final Output Layer

        #Model 5
        elif mode == 5:
            self.fc1=nn.Linear(60*4*4,1000)     #in = 60 out channels with 4*4 size, out = 1000
            self.fc2=nn.Linear(1000,1000)       # Additional layer with 100 neurons
            self.fc3=nn.Linear(1000,10)         #Final Output Layer
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        #convolution layers
        if mode != 1:
            self.conv1=nn.Conv2d(1, 40, 5) # Conv layer 1 with 1 input channel and 40 out channels with 5*5 kernel
            self.conv2=nn.Conv2d(40, 60, 5) # Conv layer 2 with 40 input channel and 60 out channels with 5*5 kernel
        
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        X = X.view(X.shape[0], -1)
        X=self.fc1(X)
        X=torch.sigmoid(X)      # Applying sigmoid activation
        X=self.fc2(X)        
        X=F.softmax(X,dim=1)
        return X
        

    # Use two convolutional layers.
    def model_2(self, X):
        X=F.max_pool2d(torch.sigmoid(self.conv1(X)),(2,2))    # max pooling for conv1 layer
        X=F.max_pool2d(torch.sigmoid(self.conv2(X)),(2,2))   # max pooling for conv2 layer
        X = X.view(X.shape[0], -1)
        X=self.fc1(X)
        X=self.fc2(torch.sigmoid(X))            # Applying sigmoid activation
        X=F.softmax(X,dim=1)
        return X

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        X=F.max_pool2d(F.relu(self.conv1(X)),(2,2))     # Replacing sigmoid with relu activation  
        X=F.max_pool2d(F.relu(self.conv2(X)),(2,2))     # Replacing sigmoid with relu activation 
        X = X.view(X.shape[0], -1)
        X=F.relu(self.fc1(X))
        X=F.relu(self.fc2(X))
        X=F.softmax(X,dim=1)
        return X
        

    # Add one extra fully connected layer.
    def model_4(self, X):
        X=F.max_pool2d(F.relu(self.conv1(X)),(2,2))
        X=F.max_pool2d(F.relu(self.conv2(X)),(2,2))
        X = X.view(X.shape[0], -1)
        X=F.relu(self.fc1(X))
        X=F.relu(self.fc2(X))    # Adding another FC (100,100)
        X=F.relu(self.fc3(X))
        X=F.softmax(X,dim=1)
        return X

    # Use Dropout now.
    def model_5(self, X):
        X=F.max_pool2d(F.relu(self.conv1(X)),(2,2))
        X=F.max_pool2d(F.relu(self.conv2(X)),(2,2))
        X = X.view(X.shape[0], -1)
        X=F.relu(self.fc1(X))   # FC layers with 1000 neurons
        X=F.dropout(X, p=0.5)    # Dropout with rate of 0.5
        X=F.relu(self.fc2(X))
        X=F.dropout(X, p=0.5)
        X=F.relu(self.fc3(X)) 
        X=F.softmax(X,dim=1)
        return X
    
    
