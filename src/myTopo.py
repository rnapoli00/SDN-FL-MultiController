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
    
    
    '''
    h1 = net.get('h1')
    h2 = net.get('h2')
    print(h1)
    print("Ping test tra h1 e h2:")
    net.ping([h1, h2])

    print("Mostro le info degli host:")
    print(h1.cmd('ifconfig'))
    print(h2.cmd('ifconfig'))
    
    h1 = net.get('h1')
    h2 = net.get('h2')
    h3 = net.get('h3')

    print(h1.cmd('echo la sto runnando di brutto'))
    
    #server TCP dummy
    h3.cmd('python3 -m http.server 8851 &')
    h1.cmd('ping -c5 10.0.0.3 &')     # h1 pinga h3
    h2.cmd('hping3 -S -p 8851 -i u100 10.0.0.3 --flood > /tmp/synflood.log 2>&1 &')

    info("  h3: netstat -ant | grep SYN_RECV")
    info("  h2: pkill hping3  # per fermare attacco")
    '''
    #myScript = "mininetscript.txt"
    #CLI(net, script=myScript) # Batch mode script execution
    CLI(net) # Send user in the CLI for additional manual actions
    #xterm h1
    #h1 python3 'echo la sto runnando di brutto'
    #h3 python3 -m http.server 8851 &
    #h1 ping -c5 h3
    #h2 hping3 -S -p 8851 -i u100 h3 2>&1
    #h1 python3 -c 'print("ciao")'
    #h3 pkill -f http.server
    net.stop()


if __name__ == '__main__':
    runner()

topos = {
    'create_topo' : create_topo
}



