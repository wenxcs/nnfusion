# Microsoft (c) 2019, MSRA/NNFUSION Team
# Author: wenxh
# This script is to be used as batch system intergration test in Azure Build Agent
import os
import sys
import subprocess, multiprocessing
import logging
import numpy as np

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

if len(sys.argv) != 3:
    logging.error("Script doesn't have right arguments.")
    exit(1)

if not sys.argv[2].endswith("nnfusion"):
    logging.error("NNFusion cli should named \"nnfusion\"")
    exit(1)

models = sys.argv[1]
nnfusion_cli = sys.argv[2]
capability_list = os.environ.get('DEVICES', '').upper().split(",")
testcase_list = os.environ.get('TESTS', '').upper().split(",")

import testcases
import e2e_evaluator

# Inputs:
# - mfolder: model folder
# - nnfusion: nnfusion place
# generatecode : $nnfusion $mfolder/*.pb -f tensorflow -b nnfusion_engine
# cmake : cd nnfusion_rt/cuda && cmake . && make -j
# test : cd nnfusion_rt/cuda && main_test -> compare result
# clean : rm -rf nnfusion_rt

class E2EManager:
    def __init__(self):
        self.capability = []
        self.capability_detect()
    
    def capability_detect(self):
        if 'DEVICES' in os.environ:
            self.capability = capability_list
            return

        # Detect Cuda
        if os.path.exists("/usr/local/cuda/bin/nvcc"):
            self.capability.append("CUDA")
            logging.info("NVCC is existed.")

        if os.path.exists("/opt/rocm/bin/hcc"):
            self.capability.append("ROCM")
            logging.info("HCC is existed.")
        
        self.capability.append("CPU")
    
        if os.path.exists(nnfusion_cli):
            logging.info("NNFusion CLI is existed.")
        else:
            logging.error("NNFusion CLI not found.")
            exit(1)
    
    def load_test_cases(self):
        if not(os.path.exists(models)):
            logging.error("Model folder is not existed.")
            os.mkdir(models)
        for e in testcases.TestCases:
            if e.valid():
                if "TESTS" in os.environ:
                    if e.casename not in testcase_list:
                        testcases.TestCases.remove(e)
                else:
                    logging.info("%s valid!"%e.casename)
            else:
                testcases.TestCases.remove(e)

    def report(self):
        manager = multiprocessing.Manager()
        report_list = manager.list()
        jobs = []
        for dev in self.capability:
            p = multiprocessing.Process(target=e2e_evaluator.E2EExecutor, args=(testcases.TestCases, dev, report_list))
            jobs.append(p)
            p.start()
        
        if 'SIDECLI' in os.environ:
            p = multiprocessing.Process(target=e2e_evaluator.CLIExecutor, args=("", report_list))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        
        print("=========================================\n\n")
        print("\tE2E Test report")
        print("\n\n=========================================\n")
        report = ("\n".join(report_list))
        print(report)
        if "Failed" in report:
            return -1
        return 0

_m = E2EManager()
_m.load_test_cases()

# Check the status
exit(_m.report())