import uuid, os, logging, sys, multiprocessing, tempfile
nnfusion_cli = sys.argv[2]
nnfusion_cli_arg = os.environ.get('NNF_CLI', '')

class E2EEvaluator:
    def __init__(self, testcase, codegen_folder = "cuda_codegen", default_device = "CUDA", working_foler = "."):
        self.codegen_folder = codegen_folder
        self.default_device = default_device
        self.testcase = testcase
        self.working_foler = working_foler

    def nnfusion_compile(self):
        logging.info("Compiling " + self.testcase.get_filename())
        os.system("cd %s && %s %s -f tensorflow -fdefault_device=%s %s >> nnfusion.log" %
                (self.working_foler, nnfusion_cli, self.testcase.get_filename(), self.default_device, nnfusion_cli_arg))
        if not os.path.exists("%s/nnfusion_rt/%s/nnfusion_rt.h"%(self.working_foler, self.codegen_folder)):
            logging.error("Failed at nnfusion compiling phase.")
            return False
        return True
    
    def build(self):
        os.system("cd %s/nnfusion_rt/%s/ && cmake . >> cmake.log && make -j 2>&1 >> cmake.log"%(self.working_foler, self.codegen_folder))
        if not os.path.exists("%s/nnfusion_rt/%s/main_test"%(self.working_foler, self.codegen_folder)):
            logging.error("Failed at compiling phase.")
            return False
        return True
    
    def allclose(self):
        os.system("cd %s/nnfusion_rt/%s/ && ./main_test > result.txt"%(self.working_foler, self.codegen_folder))
        if not os.path.exists("%s/nnfusion_rt/%s/result.txt"%(self.working_foler, self.codegen_folder)):
            logging.error("Failed at compiling phase.")
            return False
        result_file = open("%s/nnfusion_rt/%s/result.txt"%(self.working_foler, self.codegen_folder))
        results = result_file.readlines()
        if not self.testcase.allclose(results):
            logging.error("%s result missmatch."%self.testcase.casename)
            return False
        return True 

    def report(self):
        os.system("rm -rf %s/nnfusion_rt"%self.working_foler)
        if not self.nnfusion_compile():
            os.system("rm -rf %s/nnfusion_rt"%self.working_foler)
            return False
        if not self.build():
            os.system("rm -rf %s/nnfusion_rt"%self.working_foler)
            return False
        if not self.allclose():
            os.system("rm -rf %s/nnfusion_rt"%self.working_foler)
            return False
        os.system("rm -rf %s/nnfusion_rt"%self.working_foler)
        return True

configs = {"CUDA" : ["cuda_codegen", "CUDA"], "ROCM" : ["rocm_codegen", "ROCm"], "CPU" : ["cpu_codegen","CPU"]}

def E2EExecutor(TestCases, devname, report_list):
    tmpdir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    os.mkdir(tmpdir) # working folder

    for test in TestCases:
        if test.valid():
            eval = E2EEvaluator(test, configs[devname][0], configs[devname][1], tmpdir)
            report = devname + "\t" + test.casename + "\t";
            if eval.report():
                report += "Succeed!"
            else:
                report += "Failed"
            logging.info(report)
            report_list.append(report)

    # clean
    os.system("rm -rf %s"%tmpdir)

def CLIExecutor(info, report_list):
    print(info)
    side_cli = str(os.environ.get('SIDECLI', ''))
    if os.system(side_cli) == 0:
       report_list.append(side_cli + "\tSucceed!") 
    else:
       report_list.append(side_cli + "\tFailed") 