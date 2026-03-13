#!/usr/bin/python


import time

from mininet.net import Mininet
from mininet.cli import CLI

from mininet.topo import Topo
from mininet.log import setLogLevel, info



class create_topo(Topo):

    def __init__(self):
        Topo.__init__(self)

        switch_list = []

        switch = self.addSwitch('s1')
        switch_list.append(switch)
        
        for i in range(1, 4):
            host = self.addHost('h' + str(i))
            self.addLink(host, switch)
            
        

        switch = self.addSwitch('s2')
        switch_list.append(switch)
        for i in range(4, 7):
            host = self.addHost('h' + str(i))
            self.addLink(host, switch)

        self.addLink(switch_list[0], switch_list[1])
        
        


def runner():
    #setLogLevel('info')
    # Create and run a custom topo
    curr_topo = create_topo()
    net = Mininet(topo=curr_topo)
    

    net.start()
    CLI(net) # Send user in the CLI for additional manual actions
    net.stop()


if __name__ == '__main__':
    runner()

topos = {
    'create_topo' : create_topo
}



