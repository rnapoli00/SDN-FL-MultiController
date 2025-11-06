#!/usr/bin/python


import flwr as fl
import numpy as np
from sklearn.linear_model import SGDClassifier
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.utils import Bunch
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import sys
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchmetrics import Recall, ConfusionMatrix
import datetime
import torch.optim
import random

import sys
import os
'''
# --- Gestione ID Client da argomento ---
if len(sys.argv) > 1:
    client_name = sys.argv[1]
else:
    client_name = "client1"  
'''


# Class to create own customized dataset
class build_torch_dataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx, :]
        current_target = self.targets[idx]
        return (torch.tensor(current_sample, dtype=torch.float), torch.tensor(current_target, dtype=torch.long))

# Convert dataframe to the torch dataset
def convert_df_to_torch_dataset(df):

    # Extract the features and the targets
    df_data = df.iloc[:, 0: len(df.columns) - 1]
    df_target= df.iloc[:, len(df.columns) - 1: len(df.columns)]

    # Convert the dataframe to numpy array first
    ds_torch_data = df_data.to_numpy()
    ds_torch_target = df_target.to_numpy()
    
    # Convert labels from 2D to 1D
    ds_torch_target_list = ds_torch_target.tolist()
    ds_torch_target_1D = []
    for i in range(len(ds_torch_target_list)):
        ds_torch_target_1D = np.append(ds_torch_target_1D, ds_torch_target_list[i][0])

    ds_torch = build_torch_dataset(ds_torch_data, ds_torch_target_1D)
    return ds_torch


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(115, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 5),
            nn.Softmax(dim=1)
        )
    def forward(self, features):
        x = self.flatten(features)
        logits = self.linear_relu_stack(x)
        return logits

#print(NeuralNetwork().to('cpu'))

def get_parameters(net):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

               
# Function to perform the evaluation of each model based on the confusion matrix
def evaluation(confmat_glb):

    # Display the confusion matrix
    print(confmat_glb)

    # Achieve the TP, FN, FP for benign
    tp_benign = confmat_glb[0, 0].item()
    fn_benign = confmat_glb[0, 1].item() + confmat_glb[0, 2].item() + confmat_glb[0, 3].item() + confmat_glb[0, 4].item()
    fp_benign = confmat_glb[1, 0].item() + confmat_glb[2, 0].item() + confmat_glb[3, 0].item() + confmat_glb[4, 0].item()

    # Achieve the TP, FN, FP for ACK
    tp_ack = confmat_glb[1, 1].item()
    fn_ack = confmat_glb[1, 0].item() + confmat_glb[1, 2].item() + confmat_glb[1, 3].item() + confmat_glb[1, 4].item()
    fp_ack = confmat_glb[0, 1].item() + confmat_glb[2, 1].item() + confmat_glb[3, 1].item() + confmat_glb[4, 1].item()


    # calcualte recall, precision and f1 score for each label respective
    recall_benign, precision_benign, f1_score_benign = evaluation_helper(tp_benign, fn_benign, fp_benign)
    recall_ack, precision_ack, f1_score_ack = evaluation_helper(tp_ack, fn_ack, fp_ack)


    # Add them to a 2D list
    return [[recall_benign, precision_benign, f1_score_benign], [ recall_ack, precision_ack, f1_score_ack]]

# Helper function to calculate recall precision and f1 score
def evaluation_helper(tp, fn, fp):
    if tp == 0:
        recall = 0
        precision = 0
        f1_score = 0
    else:
        recall = round((tp)/(tp + fn), 4)
        precision = round((tp)/(tp + fp), 4)
        f1_score = round(2 * ((precision * recall)/(precision + recall)), 4)


    return recall, precision, f1_score

def display_evaluation(eval_list):
    print()
    print("The display will followed by format: Type: [Recall, precision, f1_score]")
    for i in range(len(eval_list)):
        if i == 0:
            print('benign:', end = ' ')
        if i == 1:
            print('ack:', end = ' ')
                
        print(eval_list[i])


def train(dataloader, model, loss_fn, optimizer, epoch):
    for i in range(epoch):
        model.train()
        for tup in dataloader:

            X = tup[0]
            y = tup[1]

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, total = 0, 0
    recall_glb = 0.0
    recall_model = Recall(task="multiclass", average='macro', num_classes=5)
    confmat_glb = torch.zeros(5, 5, dtype=torch.int64)
    with torch.no_grad():
        for tup in dataloader:
            X = tup[0]
            y = tup[1]

            # calculate y_pred
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Find the specific target
            pred_int = pred.argmax(1)
            recall_local = recall_model(pred_int, y)

            recall_glb += recall_local

            total += y.size(0)

            # Generate the confusion matrix
            confmat = ConfusionMatrix(task="multiclass", num_classes=5)

             
            confmat_local = confmat(pred_int, y)
            confmat_glb += confmat_local


    recall_glb /= size
    recall_glb = recall_glb * 12
    test_loss /= size

    eval_list = evaluation(confmat_glb)
    display_evaluation(eval_list)
            
    return test_loss, recall_glb

def train_test_itr(epochs, train_loader, test_loader):
    loss_fn = nn.CrossEntropyLoss()
    model_dnn = NeuralNetwork()
    optimizer = torch.optim.SGD(model_dnn.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n----------------------------------------------")
        train(train_loader, model_dnn, loss_fn, optimizer, epoch=5)
        test(test_loader, model_dnn, loss_fn)
        
        
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, net, trainloader, valloader, loss_func, optimizer, epoch):
        self.client_id = client_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.epoch = epoch
        print("avvio client")
    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.trainloader, self.net, self.loss_func, self.optimizer, self.epoch)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        torch.save(self.net.state_dict(), 'mode2_new.pt')
        loss, accuracy = test(self.valloader, self.net, self.loss_func)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}    



# ----------------------------
# --- CHANGED FOR NEW FLOWER API ---
# Creiamo una factory function client_fn(context) che restituisce un oggetto di tipo flwr.client.Client.
# Questo viene realizzato convertendo il NumPyClient con `.to_client()`.
# NOTA: start_client accetta `client_fn` (consente l'avvio compatibile con le API nuove).
# ----------------------------

def make_client_fn(net, trainloader, valloader, loss_fun, optimizer, epoch):
    """
    Ritorna una funzione client_fn(context) che istanzia e ritorna un flwr.client.Client.
    Usiamo una closure per passare le risorse locali (modello, loader, ecc.).
    """

    # Funzione client_fn (nuova API Flower)
    def client_fn(context):
        client_id = random.randint(1, 100)
        print(f"[Client {client_id}] Avvio client dummy...")
        numpy_client = FlowerClient(client_id, net, trainloader, valloader, loss_fun, optimizer, epoch)
        return numpy_client.to_client()
    return client_fn




def read_csv_files(path_name):
    df_ori = pd.read_csv(path_name)
    return df_ori

print(os.getcwd())
df_processed = read_csv_files("/home/tesimagistrale1/Desktop/progetto tesi/project/new_dataset/new_client1.csv")


# 80% for training and 20% for testing
df_client_train_ori, df_client_test_ori = train_test_split(df_processed, train_size=0.8, random_state=42, stratify=df_processed['target'])

df_client_test = df_client_test_ori.reset_index(drop=True)


# For traning dataset drop the data that belongs to the specific target to simulate unknown attack
 # (Traning does not know the target but testing we will test it)

df_client_train = df_client_train_ori[df_client_train_ori['target'] != 0].reset_index(drop=True)

plt.subplot(2, 2, 1)
plt.title("Train label distribution client1", fontsize=10)
df_client_train.groupby('target').size().plot(kind='pie', autopct='%.2f', figsize=(10,10))
plt.subplots_adjust(left=0.1, right=1.0, top=0.9, bottom=0.1)

plt.subplot(2, 2, 2)
plt.title("Test label distribution client1", fontsize=10)
df_client_test.groupby('target').size().plot(kind='pie', autopct='%.2f', figsize=(5,5))

ds_torch_train_client = convert_df_to_torch_dataset(df=df_client_train)
ds_torch_test_client = convert_df_to_torch_dataset(df=df_client_test)


train_loader_client = torch.utils.data.DataLoader(ds_torch_train_client, batch_size = 12, drop_last=True)
test_loader_client = torch.utils.data.DataLoader(ds_torch_test_client, batch_size = 12, drop_last=True)
        

def LDL_rst(e):
    print()
    train_test_itr(epochs=e, train_loader=train_loader_client, test_loader=test_loader_client)

starttime_LDL = datetime.datetime.now()
                
LDL_rst(e = 2)

endtime_LDL = datetime.datetime.now()

time_LDL = (endtime_LDL - starttime_LDL).seconds

time_LDL


trainloader = train_loader_client
valloader = test_loader_client
loss_fun = nn.CrossEntropyLoss()
model_dnn = NeuralNetwork()
optimizer = torch.optim.SGD(model_dnn.parameters(), lr=1e-3)
                


client_name = make_client_fn(model_dnn, trainloader, valloader, loss_fun, optimizer, epoch=3)
         
                
fl.client.start_client(
server_address = "127.0.0.1:8080",
client_fn=client_name,
)
