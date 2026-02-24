# Federated Learning with SDN Controllers using Mininet and Ryu

This project simulates a federated learning environment where SDN controllers act as FL clients and communicate through an aggregator node. The entire setup is built using Mininet for network emulation and Ryu as the SDN controller framework.

---

## 📁 Dataset

The dataset used by the controllers is located at:

   ```
   your/path/to/project/src/dataset
   ```
If you need to modify or replace the dataset, ensure you update the corresponding file paths in the fl_client.py script accordingly.

---

## 💪 Environment Setup

- Python version: 3.9 (Conda environment recommended)

### 🔧 Python Dependencies

Install the following Python packages:
- Mininet;
- Ryu;
- Flower version: `1.17`;
- hping3

```bash
pip install pandas matplotlib torch torchmetrics torchvision
```

---

## ▶️ Running the Demo

Open **four terminal windows**.

### 1. Start FL Server 

```bash
cd your/path/to/project/src
python3 server.py
```

### 2. Start Ryu Controllers

#### **Terminal 2 - Controller 1**

```bash
cd your/path/to/ryu/bin
python3 ryu-manager --observe-links your/path/to/project/src/controller1.py
```

#### **Terminal 3 - Controller 2 (separate instance)**

```bash
cd your/path/to/ryu/bin
python3 ryu-manager --observe-links --ofp-tcp-listen-port 6634 your/path/to/project/src/controller2.py
```

> ⚠️ Make sure `controller2.py` listens on a different port than `controller1.py`.

---

### 2. Terminal 4 - Launch Mininet Topology

In another terminal window, run:

```bash
cd your/path/to/project/src
sudo -E mn --custom myTopo.py --topo create_topo --switch ovs --controller=remote,ip=127.0.0.1,port=6633 --controller=remote,ip=127.0.0.1,port=6634 --arp --mac --test none --post=nameofattack.txt
```
Replace "nameofattack.txt" with one of the proposed attacks (synflood.txt, udpflood.txt, etc).

## 📌 Notes

- IP address and ports used in `server.py` should match the client IPs depending on the setup. Localhost is used.

- **Controllers act as clients** for federated training.
- **An attack on the hosts** is performed to create a dataset on which the controller will begin federated training.
- **Aggregator node (server.py)** performs the model aggregation.


The processes remain in the background even after the terminal is closed. These commands help to clear the memory from other controller instances and other FL instances respectively:
pkill -f '^python3 ryu-manager --observe-links'
pkill -f '^python3 serverTest.py'


