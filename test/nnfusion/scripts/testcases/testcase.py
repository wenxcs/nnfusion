import numpy as np
import logging
import os
from ftplib import FTP
import sys

TestCases = list()
ModelFolder = sys.argv[1]

class TestCase:
    def __init__(self, casename, ground_truth, url = ""):
        self.casename = casename
        self.ground_truth  = ground_truth
        self.rtol = 1.e-4
        self.atol = 1.e-4
        self.url = url
    
    def get_filename(self):
        return os.path.join(ModelFolder, self.casename)

    def allclose(self, result):
        if np.allclose(result, self.ground_truth, rtol=self.rtol, atol=self.atol):
            return True
        logging.error("%s has wrong result." % (self.casename))
        return False
    
    def valid(self):
        if not os.path.exists(os.path.join(ModelFolder, self.casename)):
            logging.error("%s file not existed." % (os.path.join(ModelFolder, self.casename)))
            self.download()
        return os.path.exists(os.path.join(ModelFolder, self.casename))
    
    def download(self):
        if "NNFUSION_SHARE_PWD" in os.environ and len(self.url) != 0:
            if not os.path.exists(os.path.join(ModelFolder, self.casename)):
                with FTP(host="10.190.174.54", user = "nnfusion", passwd=os.environ["NNFUSION_SHARE_PWD"]) as ftp:
                    logging.info("Downloading %s from srgssd-19:%s"%(self.casename, self.url))
                    fhandle = open(os.path.join(ModelFolder, self.casename), 'wb')
                    ftp.retrbinary('RETR ' + self.url, fhandle.write)
                    fhandle.close()