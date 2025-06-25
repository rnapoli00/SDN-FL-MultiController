# Federated Learning with SDN Controllers using Mininet and Ryu

This project simulates a federated learning environment where SDN controllers act as clients and communicate through an aggregator node. The entire setup is built using Mininet for network emulation and Ryu as the SDN controller framework.

---

## 📁 Dataset

The dataset used by the controllers is located at:

   ```
   your/path/to/project/src/dataset
   ```
If you need to modify or replace the dataset, ensure you update the corresponding file paths in the controller scripts (controller1.py, controller2.py, etc.) accordingly.

---

## 💪 Environment Setup

- Python version: 3.9 (Conda environment recommended)

### 🔧 Python Dependencies

Install the following Python packages:
- Mininet;
- Ryu;
- Flower version: `1.17`

```bash
pip install pandas matplotlib torch torchmetrics torchvision
```

---

## ▶️ Running the Demo

### 1. Start FL Server 

```bash
cd your/path/to/project/src
python3 server.py
```

### 1. Start Ryu Controllers

Open **two terminal windows**.

#### **Terminal 1 - Controller 1**

```bash
cd your/path/to/ryu/bin
python3 ryu-manager --observe-links your/path/to/project/src/controller1.py
```

#### **Terminal 1 - Controller 2 (separate instance)**

```bash
cd your/path/to/ryu/bin
python3 ryu-manager --observe-links --ofp-tcp-listen-port 6634 your/path/to/project/src/controller2.py
```

> ⚠️ Make sure `controller2.py` listens on a different port than `controller1.py`.

---

### 2. Launch Mininet Topology

In another terminal window, run:

```bash
cd cd your/path/to/project/src
sudo -E mn --custom myTopo.py --topo create_topo --switch ovs --controller=remote,ip=127.0.0.1,port=6633 --controller=remote,ip=127.0.0.1,port=6634 --arp --mac
```

---

### (Optional) Open Host Terminals

From the Mininet CLI:

```bash
xterm h1
xterm h2
xterm h3
```

From these terminal windows you could run host training.
---

## 📌 Notes

- Use `dump` in Mininet to inspect the current network state.
- IP address used in `server.py` should match the client IPs (e.g., `10.0.0.1` or `192.168.56.105` depending on the setup).

---

## 📌 Summary

- **Controllers act as clients** for federated training.
- **Aggregator node (server.py)** performs the model aggregation.
