#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re, os, sys
src_path = sys.argv[1]
print("-- Running superscaler plan generator in: ", os.getcwd())
prefix="plan/execution_plan/"
if not os.path.exists(prefix):
	os.makedirs(prefix)

print("-- Processing generated code in ", src_path)
allreduce_names = []
with open(src_path, 'r') as f:
	for line in f:
		if("name: AllReduce_" in line):
			name = re.split('\s+', line)[3]
			#print(name)
			allreduce_names.append(name)


servers_and_ports_raw = input("-- Please type ur deployment env settings, e.g., `10.0.0.21:8000,10.0.0.21:12000` : ")
servers_and_ports = []
for s in servers_and_ports_raw.split(','):
	ip, port = s.split(':')
	servers_and_ports.append(("".join(ip.split()), "".join(port.split())))
#print(servers_and_ports)
comm_type = input("-- Please type ur communication link type, e.g., `PCIE` or `RDMA` : ")
op_type="_SCAllReduce"
for (rank, ip_port) in enumerate(servers_and_ports):
    with open(prefix+str(rank)+".cfg", 'w') as f:
        f.write(ip_port[0]+"\n")
        f.write(str(rank)+"\n")
        f.write(ip_port[1]+"\n")
        f.write(str(len(allreduce_names))+"\n")
        for allreduce_name in allreduce_names:
            f.write(allreduce_name+"\n")
            f.write(op_type+"\n")
            f.write(comm_type+"\n")
            f.write(str(len(servers_and_ports))+"\n")
            for (rank, ip_port) in enumerate(servers_and_ports):
                f.write(ip_port[0]+" "+str(rank)+" "+ip_port[1]+"\n")

print("-- Plan generated")

