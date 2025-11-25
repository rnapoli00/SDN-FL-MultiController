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



class Controller1(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Controller1, self).__init__(*args, **kwargs)
        self.topology_api_app = self
        self.net = nx.DiGraph()
        self.ldl_started = False
        self.packet_records = []
        self.last_saved_index = 0
        self.counter = 0



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

        
        ip = pkt.get_protocol(ipv4.ipv4)
        tcp_seg = pkt.get_protocol(tcp.tcp)
        udp_seg = pkt.get_protocol(udp.udp)
        arp_pkt = pkt.get_protocol(arp.arp)
        if tcp_seg:
            src_port = tcp_seg.src_port
            dst_port = tcp_seg.dst_port
        elif udp_seg:
            src_port = udp_seg.src_port
            dst_port = udp_seg.dst_port
        else:
            src_port = None
            dst_port = None

        features = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dpid": datapath.id,
            "src_mac": eth.src,
            "dst_mac": eth.dst,
            "eth_type": eth.ethertype,
            "src_ip": ip.src if ip else None,
            "dst_ip": ip.dst if ip else None,
            "ip_proto": ip.proto if ip else None,
            "src_port": src_port,
            "dst_port": dst_port,
            "pkt_len": len(msg.data)
        }

        self.packet_records.append(features)
        
        csv_path = "/home/tesimagistrale1/Desktop/networkdatasetcontroller1.csv"

        df = pd.DataFrame([features])   # salva SOLO l'ultimo pacchetto

        df.to_csv(
            csv_path,
            mode='a',                                   # append
            header=not os.path.exists(csv_path),        # scrive l'header solo al primo pacchetto
            index=False
        )

        '''
        if len(self.packet_records) > self.last_saved_index:

            csv_path = "/home/tesimagistrale1/Desktop/networkdataset.csv"

            # prendi solo i nuovi record
            new_records = self.packet_records[self.last_saved_index:]
            df = pd.DataFrame(new_records)

            df.to_csv(csv_path,
                    mode='a',
                    header=not os.path.exists(csv_path),
                    index=False)

            self.last_saved_index = len(self.packet_records)

            print(f"[Controller] Salvati {len(new_records)} nuovi record (totale: {len(self.packet_records)})")
        '''
        
        '''
        if len(self.packet_records) >= self.last_saved_index + 100:

            csv_path = "/home/tesimagistrale1/Desktop/networkdataset.csv"

            # prendi solo i nuovi record
            new_records = self.packet_records[self.last_saved_index:]
            df = pd.DataFrame(new_records)

            df.to_csv(
                csv_path,
                mode='a',
                header=not os.path.exists(csv_path),
                index=False
            )

            # aggiorna l’indice fino a dove hai salvato
            self.last_saved_index = len(self.packet_records)

            print(f"[Controller] Salvati {len(new_records)} nuovi record (totale: {len(self.packet_records)})")
        '''
        '''
        if len(self.packet_records) % 100 == 0:
            df = pd.DataFrame(self.packet_records[-100:])  # solo gli ultimi 100 pacchetti

            csv_path = "/home/tesimagistrale1/Desktop/networkdataset.csv"

            df.to_csv(
                csv_path,
                mode='a',                                   # append
                header=not os.path.exists(csv_path),        # scrive l'header solo se il file non esiste
                index=False
            )

            print(f"[Controller] Aggiunti 100 nuovi record al dataset (totale raccolti: {len(self.packet_records)})")
            
        #if len(self.packet_records) % 100 == 0:
        #    df = pd.DataFrame(self.packet_records)
        #    df.to_csv("/home/tesimagistrale1/Desktop/networkdataset.csv", index=False)
        #    print(f"[Controller] Dataset aggiornato con {len(self.packet_records)} record.")
        '''

        
        if src not in self.net:
            self.net.add_node(src)
            self.net.add_edge(dpid, src, port=msg.match['in_port'])
            self.net.add_edge(src, dpid)
            
            print(">>>> Nodes <<<<")
            print(self.net.nodes())
            print(">>>> Edges <<<<")
            print(self.net.edges())

        elif src in self.net and dst in self.net:
            #print(">>>> Add your logic here <<<<")
            self.counter += 1
            #print(self.counter)
            
            
            '''
            if not self.ldl_started:
                self.ldl_started = True 
                
                script_path = "/home/tesimagistrale1/Desktop/progetto tesi/project/src/fl_client.py"

                print(f"[Controller] Avvio Client...")
                subprocess.Popen([sys.executable, script_path])
                '''
                    
            if not self.ldl_started and self.counter >= 5000:
                self.ldl_started = True 
                print(">>> Raggiunti 5000 pacchetti: STOP alla raccolta dataset <<<")
                
                # Installa una regola catch-all per non ricevere più PacketIn
                ofproto = datapath.ofproto
                parser = datapath.ofproto_parser

                match = parser.OFPMatch()  # Match su tutto
                actions = []               # Nessuna azione verso il controller
                self.add_flow(datapath, 100, match, actions)

                script_path = "/home/tesimagistrale1/Desktop/progetto tesi/project/src/fl_client.py"

                print(f"[Controller] Avvio Client...")
                subprocess.Popen([sys.executable, script_path])
                
            


            '''
            # Find the shortest path and store on a list.
            path_list = nx.shortest_path(self.net, source=src, target=dst, weight=None, method='dijkstra')
            
            # Find next hop of the forwarding path.
            next_hop = path_list[path_list.index(dpid) + 1]

            parser = datapath.ofproto_parser
            
            # Destination on flow table should match to the final destination.
            match = parser.OFPMatch(eth_dst=dst)
            
            # Find out port for next hop.
            out_port = self.net[dpid][next_hop]['port']

            action_forward = [parser.OFPActionOutput(out_port)]
            
            # Add forwarding rule to flow table and set priority to 1.
            self.add_flow(datapath, 1, match, action_forward)
            print("Added rule: eth=", dst, " out_port=", out_port)
            
            # Find switch id for source and destination
            src_id = path_list[1]
            dest_id = path_list[len(path_list) - 2]

            # Forward original packet
            parser = datapath.ofproto_parser

            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                      in_port=msg.match['in_port'], actions=action_forward)
            datapath.send_msg(out)
            '''


            path_list = nx.shortest_path(self.net, source=src, target=dst)
            next_hop = path_list[path_list.index(dpid) + 1]
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            match = parser.OFPMatch(eth_dst=dst)
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




































