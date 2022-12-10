from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet 
import argparse
import numpy as np 
import matplotlib.pyplot as plt


f=open("output.txt","a")  #For traces 

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # ======================================================================
        # Compute loss based on criterion
        loss = criterion(output,target)
       
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
       
        # ======================================================================
        # Count correct predictions overall 
        
        correct += (pred.T == target).sum()
        
    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx+1) * batch_size)

    
    f.write('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
       float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
        100. * correct / ((batch_idx+1) * batch_size)))
    return train_loss, train_acc
    


def test(model, device, test_loader):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    criterion=nn.CrossEntropyLoss()

    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            output = model(data)
            
            # ======================================================================
            # Compute loss based on same criterion as training
            loss = criterion(output,target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # ======================================================================
            # Count correct predictions overall 
            correct += (pred.T == target).sum()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    
    f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy
    

def run_main(FLAGS):

        
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)
    
    # ======================================================================
    # Define loss function.
    criterion = nn.CrossEntropyLoss()     #Using Cross Entropy for loss
    
   
    # ======================================================================
    # Define optimizer function.
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)    #Using Stochastic gradient descent for optimization

    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)
    
    best_accuracy = 0.0
    
   
    testaccuracy,testloss=[],[]       # For plotting the results 
    trainaccuracy,trainloss=[],[]
    f1=open("Accuracy_loss_data.txt","a")    # For saving Traces into output.txt
    f1.write("Epoch\t\tTrain Loss\t\tTrain Accuracy\t\tTest Loss\t\tTest Accuracy\t\t\n")
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss, test_accuracy = test(model, device, test_loader)
        f1.write(str(epoch)+"\t\t"+str(train_loss)+"\t\t"+str(round(train_accuracy.item()*100,2))+"\t\t"+str(test_loss)+"\t"+str(round(test_accuracy.item(),2))+"\n") # Saving to output.txt in a form of tabular representation
        testaccuracy.append(test_accuracy)
        testloss.append(test_loss)
        trainaccuracy.append(train_accuracy*100)
        trainloss.append(train_loss)
        print('Epoch '+str(epoch)+' done..')  
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
        
    
   #Plotting Train Accuracy and Train Loss
    epochs = range(1,FLAGS.num_epochs+1) 
    plt.plot(epochs,trainaccuracy)
    plt.title('Train Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('Train_accuracy-'+str(FLAGS.mode)+'.png')
    plt.show()
    
    plt.plot(epochs,trainloss)
    plt.title('Train Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('Train_loss-'+str(FLAGS.mode)+'.png')
    plt.show()
    
    
    #Plotting Test Accuracy and Test Loss
    plt.plot(epochs,testaccuracy)
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('Test_accuracy'+str(FLAGS.mode)+'.png')
    plt.show()
    
    plt.plot(epochs,testloss)
    plt.title('Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('Test_loss'+str(FLAGS.mode)+'.png')
    plt.show()
  
    f.write("\naccuracy is {:2.2f}".format(best_accuracy))    
    f1.write('\nBest Accuracy is '+str(round(best_accuracy.item(),2)))    #Saved to output.txt 
    print("Training and evaluation finished") #Confirmation on Output screen
    f1.close()
    
   
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
