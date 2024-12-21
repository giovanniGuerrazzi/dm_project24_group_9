import os
import time
import gzip
import torch
import pickle
import matplotlib
matplotlib.use('Agg') # Used to disable the display of the graph
import Classifier
import statistics
import numpy as np
import seaborn as sns 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import IPython.display as display
from sklearn.metrics import confusion_matrix, classification_report

from itertools import product

BATCH_SIZE = 15000


print("GRID SEARCH")
print("Loading data...")
# PATH
with gzip.open('dataset/train_set.pkl', 'r') as file:
    train_set = pickle.load(file)
with gzip.open('dataset/test_set.pkl', 'r') as file:
    test_set = pickle.load(file)
print("Data loaded!")
seed = int(time.time()%150)
print("Seed: ", seed)
print("Dataset processing...")
train_set = train_set.dropna()
test_set = test_set.dropna()
# Features
features = ['avg_position','avg_position_1','avg_position_2','avg_position_3','avg_position_4']
print("Features: ", features)
# Split
x_train = train_set[features]
y_train = train_set['top20']
# Test
x_test = test_set[features]
y_test = test_set['top20']
print("Dataset processed!")
print("Dataset shape: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# wait input from user for continue the training
input("Press Enter to continue...")
# Select device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device: ", device)
print("Converting to tensor...")
# Convert to tensor
x_train = torch.tensor(x_train.values).float()
y_train = torch.tensor(y_train.values).float()
x_test = torch.tensor(x_test.values).float()
y_test = torch.tensor(y_test.values).float()
print("Converted to tensor!")
print("Moving tensor to GPU...")
# Move to GPU
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)
# HYPERPARAMETER
num_epochs = 10
# Network configurations
network_configs = [
    {"layers": [5, 10, 25, 1], "activation": "tanh", "optimizer": "sgd", "penality": 0.0002, "momentum": 0.9, "learning_rate": 0.05},
    {"layers": [5, 8, 10, 15, 1], "activation": "tanh", "optimizer": "sgd", "penality": 0.0002, "momentum": 0.9, "learning_rate": 0.05},
    #{"layers": [10, 256, 256, 300, 3], "activation": "tanh", "optimizer": "sgd", "penality": 0.0005, "momentum": 0.8, "learning_rate": 0.003},
    # Add here others config
]
#
numberTest = len(network_configs)
bestResults = []
# CREATE DATA LOADER
print("Creating data loader...")
data_loader_train = torch.utils.data.DataLoader(list(zip(x_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(list(zip(x_test, y_test)), batch_size=BATCH_SIZE, shuffle=True)
print("Data loader created!")
###
#We will use this list for save all data for all training epoch and then calculate the mean
history_train_loss = []

print("Start training...")
# MODELS CONFIG
for number, config in enumerate(network_configs):
    print(f"Test[{number+1}/{numberTest}] --> Start training...")
    #Net settings
    layers = config['layers']
    activation = config['activation']
    optimizerName = config['optimizer']
    penality = config['penality']
    momentum = config['momentum']
    lr = config['learning_rate']
    print(f"Test[{number+1}/{numberTest}] --> Layers: {layers}, Activation: {activation}, Optimizer: {optimizerName}, Penality: {penality}, Momentum: {momentum}, Learning-rate: {lr}")
    # PATH
    testName = f"{layers}-{optimizerName}-{activation}-{penality}-{momentum}-{lr}"
    pathName = f'models/Cup-{testName}'
    bestResults.append(testName + "\n")

    # CREATE DIR
    print(f"Test[{number+1}/{numberTest}] --> Creating directory: {pathName}...")
    os.makedirs(pathName, exist_ok=True)
    print(f"Test[{number+1}/{numberTest}] --> Directory created!")
    
    # CREATE NET
    print(f"Test[{number+1}/{numberTest}] --> Creating net...")
    structureNet = []
    net = Classifier.Classifier(layers, structureNet, activation)
    print(f"Test[{number+1}/{numberTest}] --> Net created!")
    # MOVE NET TO GPU
    net = net.to(device)
    # SET TYPE NET
    net = net.float()
    # OPTIMIZER AND CRITERION
    print(f"Test[{number+1}/{numberTest}] --> Creating optimizer and criterion...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = None
    if optimizerName == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=penality)
    elif optimizerName == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=penality)
    else:
        print("OPTIMIZER NOT FOUND!")
        exit(1)
    
    #Values used for graphs
    loss_values_train = []
    accuracy_testues_train = []
    loss_values_test = []
    accuracy_testues_test = []
    # BEST
    best_accuracy_train = 0.0
    best_accuracy_test = 0.0
    best_loss_train = 100.0
    best_loss_test = 100.0
    #
    results = []
    net.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total = 0
        correct = 0
        
        for batch_input, batch_output in data_loader_train:
            #Forward pass
            outputs = net(batch_input)
            #Training loss
            loss = criterion(outputs, batch_output.view(-1, 1))
            #Calculate total loss
            total_loss += loss.item()
            #Backward and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss_train = total_loss / len(data_loader_train)
        loss_values_train.append(avg_loss_train)
        print(f"Test[{number+1}/{numberTest}] --> Epoch[{epoch+1}/{num_epochs}] --> Loss: {avg_loss_train}")    
        
    # Create predictions
    predictions_train = torch.sigmoid(net(x_train)).cpu().detach().numpy()
    predictions_test = torch.sigmoid(net(x_test)).cpu().detach().numpy()
    binary_predictions_train = (predictions_train >= 0.5).astype(int)
    binary_predictions_test = (predictions_test >= 0.5).astype(int)        
        
    confusion_matrix_train = confusion_matrix(y_train.cpu(), binary_predictions_train)
    scores = classification_report(y_train.cpu(), binary_predictions_train)
    
    # Save the confusion matrix to png
    plt.figure(figsize=(10, 7))
    plt.title('Confusion Matrix - Train')
    sns.heatmap(confusion_matrix_train, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{pathName}/Confusion-Matrix-Train.png')
    plt.clf()
    # Save all scores to txt file
    with open(f"{pathName}/scores-train.txt", "w") as file:
        file.write(scores)
    
    # TESTING
    confusion_matrix_test = confusion_matrix(y_test.cpu(), binary_predictions_test)
    scores = classification_report(y_test.cpu(), binary_predictions_test)
    
    # Generate confusion matrix png
    plt.figure(figsize=(10, 7))
    plt.title('Confusion Matrix - Test')
    sns.heatmap(confusion_matrix_test, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{pathName}/Confusion-Matrix-Test.png')
    plt.clf()
    
    # Save all scores to txt file
    with open(f"{pathName}/scores-test.txt", "w") as file:
        file.write(scores)
    print(f"Test[{number+1}/{numberTest}] --> End training!")