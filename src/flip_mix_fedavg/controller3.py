#!/usr/bin/python

from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller import ofp_event

from ryu.topology import event
from ryu.topology.api import get_switch, get_link

from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, arp


import networkx as nx
import threading
import time
import flwr as fl
import numpy as np
from sklearn.linear_model import SGDClassifier
import multiprocessing
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
import torch.optim
import random

import subprocess
import socket
import struct



class Controller3(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Controller3, self).__init__(*args, **kwargs)
        self.topology_api_app = self
        self.net = nx.DiGraph()
        self.ldl_started = False
        self.packet_records = []
        self.last_saved_index = 0
        self.counter = 0
        self.idcontroller = 3
        self.last_packet_time = None
        self.mac_encoder = {}
        self.mac_counter = 1
        self.label_flip_done = False


    def encode_mac(self, mac):
        if mac not in self.mac_encoder:
            self.mac_encoder[mac] = self.mac_counter
            self.mac_counter += 1
        return self.mac_encoder[mac]

    def ip_to_int(self, ip):
        return struct.unpack("!I", socket.inet_aton(ip))[0]



    @set_ev_cls(event.EventSwitchEnter)
    def get_topology_data(self, ev):
        switch_list = get_switch(self.topology_api_app, None)
        switches = [switch.dp.id for switch in switch_list]
        self.net.add_nodes_from(switches)

        link_list = get_link(self.topology_api_app, None)
        for link in link_list:
            self.net.add_edge(link.src.dpid, link.dst.dpid, port=link.src.port_no)
            self.net.add_edge(link.dst.dpid, link.src.dpid, port=link.dst.port_no)


    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]

        self.add_flow(datapath, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        
        msg = ev.msg
        datapath = msg.datapath
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        dpid = datapath.id
        src = eth.src
        dst = eth.dst
        
        if dpid != 3:
            return

        self.counter += 1
        
        #print(self.counter)

        if self.counter > 100:    
            ip = pkt.get_protocol(ipv4.ipv4)
            tcp_seg = pkt.get_protocol(tcp.tcp)
            udp_seg = pkt.get_protocol(udp.udp)
            arp_pkt = pkt.get_protocol(arp.arp)
            tcp_fin=0
            tcp_syn=0
            tcp_rst=0
            tcp_psh=0
            tcp_ack=0 
            tcp_urg = 0
            if tcp_seg:
                src_port = tcp_seg.src_port
                dst_port = tcp_seg.dst_port
                flags = tcp_seg.bits

                tcp_fin = int(flags & 0x01 != 0)
                tcp_syn = int(flags & 0x02 != 0)
                tcp_rst = int(flags & 0x04 != 0)
                tcp_psh = int(flags & 0x08 != 0)
                tcp_ack = int(flags & 0x10 != 0)
                tcp_urg = int(flags & 0x20 != 0)
            elif udp_seg:
                src_port = udp_seg.src_port
                dst_port = udp_seg.dst_port
                #tcp_fin = tcp_syn = tcp_rst = tcp_psh = tcp_ack = tcp_urg = 0
            else:
                src_port = None
                dst_port = None
            
            
            ip_atk= ["10.0.0.2", "10.0.0.5"]
            #ip_atkd= ["10.0.0.3", "10.0.0.6"]

            
            if ip is not None:
                src_ip = ip.src
                dst_ip = ip.dst
                ip_proto = ip.proto
            else:
                src_ip = None
                dst_ip = None
                ip_proto = None
                
                
            # 0 = benign
            # 1 = ack
            # 2 = syn
            # 3 = fin
            # 4 = udp
            if udp_seg:
                target = 4
            elif tcp_fin: 
                target = 3
            elif tcp_syn:
                target = 2
            elif tcp_ack:
                target = 1
            #or dei primi 2 in xor col terzo
            #elif src_ip in ip_atk or dst_ip in ip_atk:
            #    target = 3
            else:
                target = 0


                

                
            #src_ip_enc = self.ip_to_int(src_ip) if src_ip else 0
            #dst_ip_enc = self.ip_to_int(dst_ip) if dst_ip else 0

            # Timestamp attuale (in secondi floating point)
            current_time = time.time()

            # Calcolo del delta_time
            if self.last_packet_time is None:
                delta_time = 0.0   # primo pacchetto
            else:
                delta_time = current_time - self.last_packet_time

            # Aggiorna il last_packet_time
            self.last_packet_time = current_time

            # Salva anche timestamp leggibile
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            features = {
                #"src_ip": src_ip,
                #"dst_ip": dst_ip,
                #"src_port": src_port,
                #"dst_port": dst_port,
                #"time": timestamp_str,
                "delta_time": delta_time,
                #"dpid": datapath.id,    #id univoco di uno switch sdn, indica traffico che passa per un determinato switch
                #"src_mac": eth.src,
                #"dst_mac": eth.dst,
                "eth_type": eth.ethertype,
                #"src_ip_enc": src_ip_enc,
                #"dst_ip_enc": dst_ip_enc,
                "ip_proto": ip_proto,
                "pkt_len": len(msg.data),
                "tcp_fin": tcp_fin,
                "tcp_syn": tcp_syn,
                "tcp_rst": tcp_rst,
                "tcp_psh": tcp_psh,
                "tcp_ack": tcp_ack,
                "tcp_urg": tcp_urg,
                "target": target
            }

        
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in features.values()):
                print(f"[Controller3] Pacchetto invalido skippato: {features}")  
            else:
                self.packet_records.append(features)
            
                #csv_path = "/home/tesimagistrale1/Desktop/networkdatasetcontroller3.csv"
                base_path = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(base_path, "networkdatasetcontroller3.csv")

                df = pd.DataFrame([features])   # salva SOLO l'ultimo pacchetto

                df.to_csv(
                    csv_path,
                    mode='a',                                   # append
                    header=not os.path.exists(csv_path),        # scrive l'header solo al primo pacchetto
                    index=False
                )

        
        if src not in self.net:
            self.net.add_node(src)
            self.net.add_edge(dpid, src, port=msg.match['in_port'])
            self.net.add_edge(src, dpid)
            
            print(">>>> Nodes <<<<")
            print(self.net.nodes())
            print(">>>> Edges <<<<")
            print(self.net.edges())

        elif src in self.net and dst in self.net:
            
            if not self.label_flip_done and self.counter >= 10000:
                self.label_flip_done = True
                print(">>> Raggiunti 10000 pacchetti: AVVIO LABEL FLIPPING sui 3 dataset <<<")

                flag_cols = ["tcp_fin", "tcp_syn", "tcp_rst", "tcp_psh", "tcp_ack", "tcp_urg"]
                target_values = [0, 1, 2, 3, 4]
                base_path = os.path.dirname(os.path.abspath(__file__))

                for i in range(1, 4):
                    csv_path = os.path.join(base_path, f"networkdatasetcontroller{i}.csv")
                    if not os.path.exists(csv_path):
                        print(f"[LabelFlip] Dataset {i} non trovato, skip.")
                        continue

                    try:
                        df = pd.read_csv(csv_path)
                        n_rows = len(df)
                        flip_indices = random.sample(range(n_rows), k=int(n_rows * 0.30))

                        for idx in flip_indices:
                            # Randomizza i flag TCP
                            for col in flag_cols:
                                if col in df.columns:
                                    df.at[idx, col] = random.randint(0, 1)
                            # Randomizza la colonna target
                            if "target" in df.columns:
                                df.at[idx, "target"] = random.choice(target_values)

                        df.to_csv(csv_path, index=False)
                        print(f"[LabelFlip] Dataset {i}: label flipping su {len(flip_indices)}/{n_rows} righe completato.")
                    except Exception as e:
                        print(f"[LabelFlip] Errore sul dataset {i}: {e}")
                    
            if not self.ldl_started and self.counter >= 12500:
                self.ldl_started = True 
                print(">>> Raggiunti 12500 pacchetti: STOP alla raccolta dataset 3 <<<")
                
                # Installa una regola catch-all per non ricevere più PacketIn
                ofproto = datapath.ofproto
                parser = datapath.ofproto_parser

                match = parser.OFPMatch()  # Match su tutto
                actions = []               # Nessuna azione verso il controller
                self.add_flow(datapath, 100, match, actions)

                base_path = os.path.dirname(os.path.abspath(__file__))
                script_path = os.path.join(base_path, "fl_client.py")
                #script_path = "/home/tesimagistrale1/Desktop/attack1/project/src/fl_client.py"
                controllerid = 3

                print(f"[Controller] Avvio Client...")
                subprocess.Popen([sys.executable, script_path, str(controllerid)])
                
            

            # Find the shortest path and store on a list.
            path_list = nx.shortest_path(self.net, source=src, target=dst)
            
            # Find next hop of the forwarding path.
            next_hop = path_list[path_list.index(dpid) + 1]
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            
            # Destination on flow table should match to the final destination.
            match = parser.OFPMatch(eth_dst=dst)
            
            # Find out port for next hop.
            out_port = self.net[dpid][next_hop]['port']

            # Forward + copy-to-controller
            actions = [
                parser.OFPActionOutput(out_port),
                parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                       ofproto.OFPCML_NO_BUFFER)
            ]

            # Install the "monitoring" flow rule
            self.add_flow(datapath, 1, match, actions)

            # Send packet out immediately
            parser = datapath.ofproto_parser
            out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=msg.buffer_id,
                in_port=msg.match['in_port'],
                actions=[parser.OFPActionOutput(out_port)]
            )
            datapath.send_msg(out)

            

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Construct flow_mod message and send it
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)




































