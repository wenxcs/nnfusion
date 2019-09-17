# Microsoft (c) 2019, MSRA/NNFUSION Team
# Author: wenxh
# This script is to be used to diff the trace
import os
import re
import sys
import subprocess
import logging
import numpy as np

if len(sys.argv) != 3:
    logging.error("Script doesn't have right arguments.")
    logging.error(
        "python e2e_trace_diff.py nnfusion_debug_trace.txt tf_debug_trace.txt")
    exit(1)


class node:
    def __init__(self, name):
        self.name = name
        self.inputs = list()
        self.outputs = list()
        self.confident_pair = dict()  # "node" : 1.0(confidence)
        self.confident_list = list()
        self.output_data = None

    def add_children(self, children):
        self.outputs.append(children)
        return self

    def add_parent(self, parent):
        self.inputs += parent
        return self

    def add_data(self, data):
        self.output_data = np.array(data, dtype=float)


class tracefile:
    def __init__(self):
        self.allnodes = dict()
        self.entries = list()
        self.outputs = list()
        self.valid_outputs = set()
        self.no_match = set()
        self.match = set()
        self.ind = 0
        self.visited = dict()

    def read_nnfusion_trace(self, file):
        #  node: 0.0  0.0 : input1, input2
        f = open(file).readlines()
        for line in f:
            line = line.strip()
            if line.endswith(":"):
                break
            segs = line.split(":")

            name = segs[0].strip()
            inputs = [v.strip() for v in segs[3].split(",")]
            data1 = [float(v.strip()) for v in segs[1].strip().split(
                "...(")[0].strip().split(" ")]
            data2 = [float(v.strip()) for v in segs[2].strip().split(
                "...(")[0].strip().split(" ")]
            data = data1 + data2

            # create node
            n = node(name)
            n.add_parent(inputs)
            n.add_data(data)
            self.allnodes[name] = n

            # add children
            for p in inputs:
                if p not in self.allnodes.keys():
                    self.allnodes[p] = node(p)
                    self.entries.append(name)
                self.allnodes[p].add_children(name)

            logging.info(("%s <- %s : %s") % (name, inputs, data))

        for nname in self.allnodes.keys():
            if len(self.allnodes[nname].inputs) == 0:
                self.entries.append(nname)
            if len(self.allnodes[nname].outputs) == 0:
                self.outputs.append(nname)

    def read_tf_trace(self, file):
        f = open(file)
        while True:
            # dense_1/bias:0 <- xxx:0, xxx:0
            # [0. 0. 0. 0. 0.
            # 0. 0. 0. 0. 0.] ...(size= 512 end with 0.0 )
            line = f.readline()
            if line:
                if "<-" not in line:
                    break
                ls = line.split("<-")
                name = ls[0].strip()
                inputs = [v.strip() for v in ls[1].strip().split(",")]
                n = node(name)
                n.add_parent(inputs)
                self.allnodes[name] = n

                # add children
                for p in inputs:
                    if p not in self.allnodes.keys():
                        self.allnodes[p] = node(p)
                    self.allnodes[p].add_children(name)

                line = ""
                while "] ...(size=" not in line:
                    line += f.readline()
                # hard fix for bool
                line = line.replace("[False]", "[0]")
                data1 = [float(v) for v in re.split(
                    "\s+", line.split("]")[0][1:].strip())]

                line = ""
                while "] offset= " not in line:
                    line += f.readline()
                # hard fix for bool
                line = line.replace("[False]", "[0]")
                data2 = [float(v) for v in re.split(
                    "\s+", line.split("]")[0][1:].strip())]
                data = data1 + data2

                self.allnodes[name].add_data(data)

                logging.info(("%s <- %s : %s") % (name, inputs, data))
            else:
                break

        for nname in self.allnodes.keys():
            if len(self.allnodes[nname].inputs) == 0:
                self.entries.append(nname)
            if len(self.allnodes[nname].outputs) == 0:
                self.outputs.append(nname)

    def is_child(self, father, son):
        if father == son:
            return True
        start_node = self.allnodes[father]
        for u in start_node.outputs:
            if self.is_child(u, son):
                return True
        return False

    def is_child_matched(self, n, trace, v):
        for child in self.allnodes[n].outputs:
            valid = False
            for u in self.allnodes[child].confident_list:
                if trace.is_child(v, u):
                    valid = True
            if not(valid or self.is_child_matched(child, trace, v)):
                return False
        return True

    def rouge_match(self, trace):
        for v in self.allnodes.keys():
            for u in trace.allnodes.keys():
                if self.allnodes[v].output_data is None or trace.allnodes[u].output_data is None or len(self.allnodes[v].output_data) != len(trace.allnodes[u].output_data):
                    continue
                # logging.info(self.allnodes[v].output_data, trace.allnodes[u].output_data)
                if np.allclose(self.allnodes[v].output_data, trace.allnodes[u].output_data, rtol=1.e-4, atol=1.e-4):
                    self.allnodes[v].confident_list.append(u)
                    # trace.allnodes[u].confident_list.append(v)

    def subgrah_match(self, cur_node, trace, trace_node, dep=0):
        if cur_node + trace_node in self.visited.keys():
            return self.visited[cur_node + trace_node]
        tabs = "".join(["-"]*dep)

        if trace_node in self.allnodes[cur_node].confident_list:
            logging.info("%s%s --allclose--> %s" %
                         (tabs, cur_node, trace_node))
            if cur_node in self.outputs:  # and trace_node in trace.outputs:
                #print("[Confident Path]" + "-> ".join(confident_path))
                # for match in confident_path:
                self.match.add(cur_node)
                self.visited[cur_node+trace_node] = True
                logging.info("%s^------- Confident match path ends here."%(" "*dep))
                return True
        # else:
            # logging.info("%s%s --skip--> None"%(tabs, cur_node))

        ret_flag = False
        for subnode in self.allnodes[cur_node].outputs:
            node_flag = False
            # valid for one case
            for trace_sub_node in self.allnodes[subnode].confident_list:
                if trace.is_child(trace_node, trace_sub_node):
                    if self.subgrah_match(subnode, trace, trace_sub_node, dep + 1):
                        node_flag = True
            # can skip this node
            sub_flag = len(self.allnodes[subnode].outputs) > 0
            for subsubnode in self.allnodes[subnode].outputs:
                logging.info("%s%s --skip-->" %
                         (tabs, cur_node))
                if not self.subgrah_match(subsubnode, trace, trace_node, dep + 1):
                    sub_flag = False
                    break

            if node_flag or sub_flag:
                ret_flag = True

        if ret_flag:
            self.match.add(cur_node)
        else:
            logging.info("%s%s --failed--> none" %
                         (tabs, cur_node))
            self.no_match.add(cur_node)

        self.visited[cur_node+trace_node] = ret_flag
        return ret_flag

    def compare_with(self, trace):
        self.rouge_match(trace)

        logging.info("Entry: " + ", ".join(self.entries))
        logging.info("Result: " + ", ".join(self.outputs))
        # DFS to output all fist missmatch
        # Root level nodes
        for v in self.entries:
            flag = False
            if len(self.allnodes[v].confident_list) == 0:
                if self.allnodes[v].output_data is None:
                    for u in self.allnodes[v].outputs:
                        for trace_node in self.allnodes[u].confident_list:
                            logging.info("%s --skip-->" %
                                (v))
                            if self.subgrah_match(u, trace, trace_node):
                                flag = True
            else:
                for trace_node in self.allnodes[v].confident_list:
                    if self.subgrah_match(v, trace, trace_node):
                        flag = True
            if flag:
                self.match.add(v)
            else:
                self.no_match.add(v)

        #logging.info("Possible match:")
        # logging.info("\n".join(self.match))

        print("\n[Confident match]")
        for item in self.match:
            print(" - %s - possible {%s}" % (item,
                                             ", ".join(self.allnodes[item].confident_list)))
        print("\n[No match]")
        nomatch = self.no_match - self.match
        pomatch = set(self.allnodes.keys()) - self.no_match.union(self.match)
        for item in nomatch:
            print(" - %s - possible {%s}" % (item,
                                             ", ".join(self.allnodes[item].confident_list)))
        print("\n[Uncertain match]")
        for item in pomatch:
            if not self.allnodes[item].output_data is None:
                print(
                    " - %s - possible {%s}" % (item, ", ".join(self.allnodes[item].confident_list)))

        return False


nnf_trace = tracefile()
nnf_trace.read_nnfusion_trace(sys.argv[1])
tf_trace = tracefile()
tf_trace.read_tf_trace(sys.argv[2])

if nnf_trace.compare_with(tf_trace):
    exit(0)
else:
    exit(1)
