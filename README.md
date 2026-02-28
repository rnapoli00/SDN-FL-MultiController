# Federated Learning with SDN Controllers using Mininet and Ryu

This project simulates a federated learning environment where SDN controllers act as FL clients. The controllers capture network packets in real-time to generate local datasets to train their models, then they communicate the parameters to an aggregator node. The entire infrastructure is built using Mininet for network emulation and Ryu as the SDN controller framework."

---

## Environment Setup

- Python version: 3.9 (Conda environment recommended)

### Python Dependencies

Install the following Python packages:
- Mininet;
- Ryu;
- Flower version: `1.17`;
- hping3

```bash
pip install pandas matplotlib torch torchmetrics torchvision
```

---

## Running the Demo

Open **four terminal windows**.

### 1. Start FL Server 

The server needs to know how many controllers to expect.

```bash
cd src
python3 server.py n_controllers
```

### 2. Start Ryu Controllers (2 Controller Example)

#### **Terminal 2 - Controller 1**

```bash
cd your/path/to/ryu/bin
python3 ryu-manager --observe-links src/controller1.py
```

#### **Terminal 3 - Controller 2 (separate instance)**

```bash
cd your/path/to/ryu/bin
python3 ryu-manager --observe-links --ofp-tcp-listen-port 6634 src/controller2.py
```

> ⚠️ Make sure `controller2.py` listens on a different port than `controller1.py`.

---

#### 3. Terminal 4 - Launch Mininet Topology

In another terminal window, run:

```bash
cd src
sudo -E mn --custom myTopo.py --topo create_topo --switch ovs --controller=remote,ip=127.0.0.1,port=6633 --controller=remote,ip=127.0.0.1,port=6634 --arp --mac --test none --post=nameofattack.txt
```
Replace "nameofattack.txt" with one of the proposed attacks (synandfinflood.txt, synfinudp.txt, etc).

For 3 controllers architecture, run:

```bash
cd src
sudo -E mn --custom topo3Clients.py --topo create_topo --switch ovs --controller=remote,ip=127.0.0.1,port=6633 --controller=remote,ip=127.0.0.1,port=6634 --controller=remote,ip=127.0.0.1,port=6635 --arp --mac --test none --post=nameofattack.txt
```



## 📌 Notes

- IP address and ports used in `server.py` should match the client IPs depending on the setup. Localhost is used.

- **Controllers act as clients** for federated training.
- **An attack on the hosts** is performed to create a dataset on which the controller will begin federated training.
- **Aggregator node (server.py)** performs the model aggregation.





