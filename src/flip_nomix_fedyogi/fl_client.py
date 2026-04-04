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
import csv
import time
import torch.optim
import random

import sys
import os

from sklearn.preprocessing import StandardScaler

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
def convert_df_to_torch_dataset(df, scaler= None):

    # Extract the features and the targets
    df_data = df.iloc[:, 0: len(df.columns) - 1]
    df_target= df.iloc[:, len(df.columns) - 1: len(df.columns)]
    
    if scaler is None:        
        scaler = StandardScaler()
        df_data = scaler.fit_transform(df_data)
    else:
        df_data = scaler.transform(df_data)

    # Convert the dataframe to numpy array first
    #ds_torch_data = df_data.to_numpy()
    ds_torch_target = df_target.to_numpy()
    ds_torch_data = df_data
    
    # Convert labels from 2D to 1D
    ds_torch_target_list = ds_torch_target.tolist()
    ds_torch_target_1D = []
    for i in range(len(ds_torch_target_list)):
        ds_torch_target_1D = np.append(ds_torch_target_1D, ds_torch_target_list[i][0])

    ds_torch = build_torch_dataset(ds_torch_data, ds_torch_target_1D)
    return ds_torch, scaler


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        #self.linear_relu_stack = nn.Sequential(
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            #nn.Softmax(dim=1)
        )
    #def forward(self, features):
    #    x = self.flatten(features)
    #    logits = self.linear_relu_stack(x)
    #    return logits
    def forward(self, x):
        return self.net(x)
        

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
    print("matrice di confusione, righe= classe vera, colonne classe predetta:")
    print(confmat_glb)

    # Achieve the TP, FN, FP for benign
    tp_benign = confmat_glb[0, 0].item()
    fn_benign = confmat_glb[0, 1].item() + confmat_glb[0, 2].item() + confmat_glb[0, 3].item() + confmat_glb[0, 4].item()
    fp_benign = confmat_glb[1, 0].item() + confmat_glb[2, 0].item() + confmat_glb[3, 0].item() + confmat_glb[4, 0].item()

    # Achieve the TP, FN, FP for ACK
    tp_ack = confmat_glb[1, 1].item()
    fn_ack = confmat_glb[1, 0].item() + confmat_glb[1, 2].item() + confmat_glb[1, 3].item() + confmat_glb[1, 4].item()
    fp_ack = confmat_glb[0, 1].item() + confmat_glb[2, 1].item() + confmat_glb[3, 1].item() + confmat_glb[4, 1].item()

    # Achieve the TP, FN, FP for SYN
    tp_syn = confmat_glb[2, 2].item()
    fn_syn= confmat_glb[2, 0].item() + confmat_glb[2, 1].item() + confmat_glb[2, 3].item() + confmat_glb[2, 4].item()
    fp_syn = confmat_glb[0, 2].item() + confmat_glb[1, 2].item() + confmat_glb[3, 2].item() + confmat_glb[4, 2].item()
    
    # Achieve the TP, FN, FP for FIN
    tp_fin = confmat_glb[3, 3].item()
    fn_fin= confmat_glb[3, 0].item() + confmat_glb[3, 1].item() + confmat_glb[3, 2].item() + confmat_glb[3, 4].item()
    fp_fin = confmat_glb[0, 3].item() + confmat_glb[1, 3].item() + confmat_glb[2, 3].item() + confmat_glb[4, 3].item()

    # Achieve the TP, FN, FP for UDP
    tp_udp = confmat_glb[4, 4].item()
    fn_udp= confmat_glb[4, 0].item() + confmat_glb[4, 1].item() + confmat_glb[4, 2].item() + confmat_glb[4, 3].item()
    fp_udp = confmat_glb[0, 4].item() + confmat_glb[1, 4].item() + confmat_glb[2, 4].item() + confmat_glb[3, 4].item()


    # calcualte recall, precision and f1 score for each label respective
    recall_benign, precision_benign, f1_score_benign = evaluation_helper(tp_benign, fn_benign, fp_benign)
    recall_ack, precision_ack, f1_score_ack = evaluation_helper(tp_ack, fn_ack, fp_ack)
    recall_syn, precision_syn, f1_score_syn = evaluation_helper(tp_syn, fn_syn, fp_syn)
    recall_fin, precision_fin, f1_score_fin = evaluation_helper(tp_fin, fn_fin, fp_fin)
    recall_udp, precision_udp, f1_score_udp = evaluation_helper(tp_udp, fn_udp, fp_udp)


    # Add them to a 2D list
    return [[recall_benign, precision_benign, f1_score_benign], [ recall_ack, precision_ack, f1_score_ack],
            [recall_syn, precision_syn, f1_score_syn], [recall_fin, precision_fin, f1_score_fin], [recall_udp, precision_udp, f1_score_udp]]


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
        if i == 2:
            print('syn:', end = ' ')
        if i == 3:
            print('fin:', end = ' ')
        if i == 4:
            print('udp:', end = ' ')
                
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
    num_batches = len(dataloader)
    model.eval()
    
    test_loss, correct = 0, 0
    confmat_glb = torch.zeros(5, 5, dtype=torch.int64)
    
    with torch.no_grad():
        for tup in dataloader:
            X = tup[0]
            y = tup[1]

            # Calcola le predizioni
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Estrapola la classe predetta
            pred_int = pred.argmax(1)
            
            # Calcola il numero di predizioni corrette per la vera Accuracy
            correct += (pred_int == y).type(torch.float).sum().item()

            # Aggiorna la matrice di confusione
            confmat = ConfusionMatrix(task="multiclass", num_classes=5)
            confmat_local = confmat(pred_int, y)
            confmat_glb += confmat_local

    # Calcolo metriche globali corrette
    test_loss /= num_batches
    accuracy = correct / size

    # Valutazione e visualizzazione
    eval_list = evaluation(confmat_glb)
    display_evaluation(eval_list)
            
    return test_loss, accuracy, eval_list, confmat_glb

def train_test_itr(epochs, train_loader, test_loader):
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    model_dnn = NeuralNetwork()
    optimizer = torch.optim.SGD(model_dnn.parameters(), lr=1e-4, momentum=0.9)
    #optimizer = torch.optim.SGD(model_dnn.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n----------------------------------------------")
        print(f"prova fl_client\n----------------------------------------------")
        train(train_loader, model_dnn, loss_fn, optimizer, epoch=5) #5 epoch ogni iterazione del for
        test(test_loader, model_dnn, loss_fn)
        
        
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, loss_func, optimizer, epoch,
                 controllerid, base_path):          # ← aggiunto
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.epoch = epoch
        self.controllerid = controllerid            # ← salvato sull'istanza
        self.base_path = base_path                  # ← salvato sull'istanza
        
    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.trainloader, self.net, self.loss_func, self.optimizer, self.epoch)
        #riga sottostante per fedAvg
        return get_parameters(self.net), len(self.trainloader), {}
        #return get_parameters(config={}), 1000, {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        torch.save(self.net.state_dict(), f'model_client_{self.controllerid}.pt')
        
        loss, accuracy, eval_list, confmat_glb = test(self.valloader, self.net, self.loss_func)
        
        current_round = config.get("server_round", 0)
        
        class_names = ["benign", "ack", "syn", "fin", "udp"]
        
        per_class_metrics = {}
        for i, name in enumerate(class_names):
            per_class_metrics[f"recall_{name}"]    = eval_list[i][0]
            per_class_metrics[f"precision_{name}"] = eval_list[i][1]
            per_class_metrics[f"f1_{name}"]        = eval_list[i][2]
        
        confmat_list = confmat_glb.tolist()
        cm_header = [f"cm_{r}_{c}" for r in range(5) for c in range(5)]
        cm_values  = [confmat_list[r][c] for r in range(5) for c in range(5)]
        
        # usa self.controllerid e self.base_path invece delle variabili globali
        results_file = os.path.join(self.base_path, f"metrics_client_{self.controllerid}.csv")
        file_exists = os.path.isfile(results_file)
        
        with open(results_file, mode='a', newline='') as f:
            print(f"scrivo su: {results_file}")
            writer = csv.writer(f)
            if not file_exists:
                header = ["round", "timestamp", "loss", "accuracy"]
                header += [f"recall_{n}"    for n in class_names]
                header += [f"precision_{n}" for n in class_names]
                header += [f"f1_{n}"        for n in class_names]
                header += cm_header
                writer.writerow(header)
            
            row = [
                current_round,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                float(loss),
                float(accuracy)
            ]
            row += [per_class_metrics[f"recall_{n}"]    for n in class_names]
            row += [per_class_metrics[f"precision_{n}"] for n in class_names]
            row += [per_class_metrics[f"f1_{n}"]        for n in class_names]
            row += cm_values
            writer.writerow(row)
            
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# ----------------------------
# --- CHANGED FOR NEW FLOWER API ---
# Creiamo una factory function client_fn(context) che restituisce un oggetto di tipo flwr.client.Client.
# Questo viene realizzato convertendo il NumPyClient con `.to_client()`.
# NOTA: start_client accetta `client_fn` (consente l'avvio compatibile con le API nuove).
# ----------------------------

def make_client_fn(net, trainloader, valloader, loss_fun, optimizer, epoch,
                   controllerid, base_path):        # ← aggiunto
    """
    Ritorna una funzione client_fn(context) che istanzia e ritorna un flwr.client.Client.
    """
    def client_fn(context):
        numpy_client = FlowerClient(
            net, trainloader, valloader, loss_fun, optimizer, epoch,
            controllerid, base_path                # ← passato alla classe
        )
        return numpy_client.to_client()
    return client_fn



def read_csv_files(path_name):
    df_ori = pd.read_csv(path_name)
    return df_ori

def main():

    controllerid = int(sys.argv[1])
    print(f"Working Directory: {os.getcwd()}")
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 1. Definiamo i percorsi di tutti i dataset
    csv_files = {
        1: os.path.join(base_path, "networkdatasetcontroller1.csv"),
        2: os.path.join(base_path, "networkdatasetcontroller2.csv"),
        3: os.path.join(base_path, "networkdatasetcontroller3.csv")
    }
    ROW_THRESHOLD = 13000

    print(f"[Client {controllerid}] In attesa che tutti i 3 dataset raggiungano almeno {ROW_THRESHOLD} righe...")

    while True:
        all_ready = True
        status_report = []

        for f in csv_files.values():
            if os.path.exists(f):
                # Leggiamo solo il numero di righe per essere veloci (senza caricare tutto il DF)
                try:
                    row_count = sum(1 for _ in open(f)) - 1 # -1 per l'header
                except Exception:
                    row_count = 0
                
                status_report.append(f"{os.path.basename(f)}: {row_count}/{ROW_THRESHOLD}")
                
                if row_count < ROW_THRESHOLD:
                    all_ready = False
            else:
                status_report.append(f"{os.path.basename(f)}: NON TROVATO")
                all_ready = False

        if all_ready:
            print(f"\n[Client {controllerid}] Condizione soddisfatta!")
            break
        else:
            # Stampa lo stato attuale su una riga per non intasare il terminale
            print(f"\r[Client {controllerid}] Stato: {' | '.join(status_report)}", end="", flush=True)
            time.sleep(5) # Controlla ogni 5 secondi

    print(f"\n[Client {controllerid}] Tutti i dataset sono pronti. Inizio crosstesting...")

    train_parts = []
    test_parts = []

    # 2. Carichiamo e splittiamo TUTTI i dataset
    for cid, path in csv_files.items():
        df_temp = pd.read_csv(path)
        
        # Usiamo lo stesso random_state=42 per garantire che il 20% di test 
        # sia sempre lo stesso per ogni esecuzione e non si mescoli mai col training
        train_part, test_part = train_test_split(
            df_temp, 
            train_size=0.8, 
            random_state=42, 
            stratify=df_temp['target']
        )
        
        # Salviamo la parte di train se corrisponde al client corrente
        if cid == controllerid:
            df_client_train = train_part.reset_index(drop=True)
        
        # Aggiungiamo la parte di test alla lista globale
        test_parts.append(test_part)

    # 3. Creiamo il Test Set Globale (unione dei 20% di tutti e 3 i controller)
    df_client_test = pd.concat(test_parts).reset_index(drop=True)

    print(f"--- Client {controllerid} ---")
    print(f"Training locale su: {len(df_client_train)} campioni")
    print(f"Test globale (unseen) su: {len(df_client_test)} campioni")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Train Dist (Client {controllerid})")
    df_client_train.groupby('target').size().plot(kind='pie', autopct='%.1f%%')
    plt.subplot(1, 2, 2)
    plt.title("Global Test Dist (All Clients)")
    df_client_test.groupby('target').size().plot(kind='pie', autopct='%.1f%%')
    plot_path = os.path.join(base_path, f"dataset_distribution_client_{controllerid}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Client {controllerid}] Distribuzione salvata in: {plot_path}")

    ds_torch_train_client, scaler = convert_df_to_torch_dataset(df_client_train)
    ds_torch_test_client,  _     = convert_df_to_torch_dataset(df_client_test, scaler=scaler)

    train_loader_client = torch.utils.data.DataLoader(ds_torch_train_client, batch_size=64, shuffle=True)
    test_loader_client  = torch.utils.data.DataLoader(ds_torch_test_client,  batch_size=64, shuffle=False)
            


    def LDL_rst(e):
        print()
        train_test_itr(epochs=e, train_loader=train_loader_client, test_loader=test_loader_client)

    #starttime_LDL = datetime.datetime.now()
                    
    #LDL_rst(e = 2)

    #endtime_LDL = datetime.datetime.now()

    #time_LDL = (endtime_LDL - starttime_LDL).seconds

    #time_LDL


    trainloader = train_loader_client
    valloader = test_loader_client
    loss_fun = nn.CrossEntropyLoss()
    model_dnn = NeuralNetwork()
    #optimizer = torch.optim.SGD(model_dnn.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model_dnn.parameters(), lr=0.001)        


    client_fn = make_client_fn(
        model_dnn, trainloader, valloader, loss_fun, optimizer, epoch=5,
        controllerid=controllerid, base_path=base_path   # ← passato esplicitamente
    )
            
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client_fn=client_fn,
    )

if __name__ == "__main__":
    main()