# Microsoft (c) 2019, MSRA/NNFUSION Team
# Author: wenxh
# This script is to be used as batch system intergration test in Azure Build Agent
import os
import sys
import subprocess
import logging
import numpy as np

# todo(wenxh): replace those accurate result.
ground_truth_str = '''Result_1077_0: 
1.546362e-03 5.187878e-04 8.514206e-04 1.672945e-03 5.377632e-04 4.484089e-04 1.265961e-03 5.941496e-04 1.430401e-03 3.447847e-04  .. (size = 1024, ends with 5.333492e-04);
Result_1078_0: 
6.931930e+00  .. (size = 1, ends with 6.931930e+00);
Result_1079_0: 
8.027602e-07 -5.185190e-04 4.414183e-07 8.686796e-07 2.786382e-07 2.323008e-07 6.568488e-07 3.078871e-07 7.423993e-07 1.785829e-07  .. (size = 1024, ends with 2.763489e-07);
Result_1080_0: 
-5.660538e-07 3.656255e-04 -3.112592e-07 -6.125358e-07 -1.964774e-07 -1.638033e-07 -4.631666e-07 -2.171018e-07 -5.234912e-07 -1.259249e-07  .. (size = 1048576, ends with 1.256160e-07);
Result_1081_0: 
-1.335418e-05 2.288632e-06 1.171574e-05 -1.878377e-05 1.534272e-05 -1.544671e-05 -1.653034e-05 -4.949104e-06 1.948970e-05 2.868104e-06  .. (size = 1024, ends with 1.357577e-06);
Result_1082_0: 
9.812855e-06 -1.681722e-06 -8.608909e-06 1.380261e-05 -1.127407e-05 1.135048e-05 1.214675e-05 3.636678e-06 -1.432133e-05 -2.107527e-06  .. (size = 1048576, ends with 1.683170e-06);
Result_1083_0: 
4.023454e-06 8.224578e-08 -1.242741e-06 -5.651873e-08 -3.664542e-06 -9.587163e-06 3.216275e-06 -1.422902e-06 -1.581659e-06 -4.980079e-06  .. (size = 1024, ends with 3.120409e-06);
Result_1084_0: 
-2.956496e-06 2.461846e-08 8.503185e-07 7.438492e-09 -2.949090e-06 3.034614e-06 6.096668e-07 1.788963e-07 -8.968601e-07 -2.138897e-06  .. (size = 1024, ends with 3.868789e-06);
Result_1085_0: 
4.172960e-06 1.650358e-07 -8.254591e-07 1.492260e-07 -3.517100e-06 -8.822678e-06 3.161066e-06 -1.145738e-06 -1.481754e-06 -4.661740e-06  .. (size = 1024, ends with 2.788725e-06);
Result_1087_0: 
-6.979659e-07 -2.760375e-08 1.380656e-07 -2.495942e-08 5.882673e-07 1.475674e-06 -5.287173e-07 1.916352e-07 2.478369e-07 7.797188e-07  .. (size = 4194304, ends with 1.548329e-06);
Result_1088_0: 
1.223381e-07 9.307674e-08 -1.017021e-06 -2.177935e-06 -4.944389e-06 1.406804e-06 4.437769e-07 2.071674e-07 6.120346e-06 -3.031246e-07  .. (size = 4096, ends with 1.153484e-06);
Result_1090_0: 
-7.090462e-08 -5.394533e-08 5.894444e-07 1.262286e-06 2.865665e-06 -8.153544e-07 -2.572038e-07 -1.200699e-07 -3.547225e-06 1.756847e-07  .. (size = 4194304, ends with 2.223027e-06);
Result_1091_0: 
1.987860e-07 2.210786e-06 -2.039957e-06 -2.144095e-06 -4.785308e-06 -5.776134e-06 3.932939e-06 -1.914861e-06 1.332128e-06 -3.703045e-06  .. (size = 1024, ends with 4.123984e-06);
Result_1092_0: 
-1.152122e-07 1.589316e-06 1.524652e-06 1.025542e-06 -3.137675e-06 3.608108e-06 -9.231485e-07 3.433096e-08 1.359460e-06 -1.455075e-06  .. (size = 1024, ends with 7.947857e-06);
Result_1093_0: 
4.353525e-07 2.324124e-06 -1.743085e-06 -1.861073e-06 -4.511257e-06 -5.402847e-06 4.065012e-06 -1.664671e-06 1.447031e-06 -3.437463e-06  .. (size = 1024, ends with 4.121732e-06);
Result_1094_0: 
-2.031026e-07 -1.084261e-06 8.131915e-07 8.682359e-07 2.104611e-06 2.520560e-06 -1.896427e-06 7.766097e-07 -6.750751e-07 1.603660e-06  .. (size = 1048576, ends with 8.314421e-07);
Result_1095_0: 
-3.159592e-06 9.374957e-06 -1.499427e-07 1.876313e-06 -3.756845e-06 -3.047303e-07 3.099408e-06 -4.679643e-06 -5.599778e-06 4.521201e-06  .. (size = 1024, ends with 9.363574e-07);
Result_1096_0: 
3.375954e-06 -1.001694e-05 1.602105e-07 -2.004800e-06 4.014105e-06 3.255975e-07 -3.311650e-06 5.000094e-06 5.983241e-06 -4.830806e-06  .. (size = 1048576, ends with 8.993996e-07);
Result_1097_0: 
7.965902e-09 2.656576e-08 4.711520e-09 1.165803e-08 3.868115e-09 -4.349459e-09 -4.830482e-08 3.117785e-08 2.832978e-08 1.766098e-08  .. (size = 1024, ends with -2.957026e-08);
Result_1098_0: 
-3.834373e-09 -1.278738e-08 -2.267882e-09 -5.611571e-09 -1.861911e-09 2.093605e-09 2.325144e-08 -1.500741e-08 -1.363649e-08 -8.501083e-09  .. (size = 1048576, ends with -5.536882e-08);
Result_1099_0: 
3.436140e-14 2.615685e-13 -1.800782e-13 1.154632e-13 2.260414e-13 1.276756e-13 9.658940e-14 1.485478e-13 -1.667555e-13 -3.232969e-13  .. (size = 1024, ends with -6.439294e-15);
Result_1100_0: 
-4.551228e-09 -3.762379e-08 2.522472e-08 -1.649810e-08 -3.128809e-08 -1.841885e-08 -1.283349e-08 -2.016346e-08 2.261892e-08 4.455760e-08  .. (size = 1048576, ends with -3.636285e-09);
Result_1101_0: 
-1.739063e-06 1.354803e-06 -8.205348e-06 -2.759901e-06 -5.746696e-06 -1.659134e-06 3.268844e-06 1.331822e-06 6.321858e-07 7.794520e-07  .. (size = 1024, ends with 3.850675e-06);
Result_1102_0: 
2.093496e-06 1.606087e-06 4.851771e-06 1.242755e-06 -2.020731e-06 2.481698e-06 -2.112107e-06 7.450653e-07 1.227667e-06 -1.500988e-06  .. (size = 1024, ends with 7.428138e-06);
Result_1103_0: 
-1.588658e-06 1.315943e-06 -7.654319e-06 -2.535821e-06 -5.350067e-06 -1.494685e-06 3.126769e-06 1.312839e-06 6.398708e-07 7.926060e-07  .. (size = 1024, ends with 3.664033e-06);
Result_1105_0: 
-2.121617e-06 5.169102e-07 -8.437761e-06 -2.189575e-06 -4.344193e-06 6.424335e-07 1.908300e-06 2.333979e-06 1.000916e-07 2.493595e-06  .. (size = 4194304, ends with 6.041208e-07);
Result_1106_0: 
-2.896340e-06 -3.609324e-06 -6.919342e-07 1.054056e-06 -6.055911e-08 -4.089437e-06 -1.028964e-06 -7.455086e-07 9.844172e-07 -2.875338e-06  .. (size = 4096, ends with -1.436676e-06);
Result_1108_0: 
2.444823e-06 2.399501e-06 6.151231e-07 -6.549010e-07 2.601588e-08 2.496367e-06 7.836123e-07 7.604668e-07 -7.689225e-07 1.988931e-06  .. (size = 4194304, ends with -2.162470e-06);
Result_1109_0: 
-9.064353e-06 4.240645e-06 -6.506652e-06 -4.462870e-06 -1.791642e-07 -5.093899e-07 5.500298e-06 5.388227e-07 -5.929458e-06 6.045602e-06  .. (size = 1024, ends with 6.620766e-06);
Result_1110_0: 
7.046599e-06 7.146490e-07 1.336655e-06 1.397636e-06 4.339685e-07 2.166340e-06 -1.994791e-06 4.899277e-07 -5.098680e-06 2.054917e-06  .. (size = 1024, ends with 1.001468e-05);
Result_1111_0: 
-8.788145e-06 3.961067e-06 -6.291569e-06 -4.395699e-06 -2.933801e-07 -6.266558e-07 5.169713e-06 3.609846e-07 -5.816144e-06 5.635639e-06  .. (size = 1024, ends with 6.214695e-06);
Result_1112_0: 
-2.250657e-06 1.014561e-06 -1.610017e-06 -1.125727e-06 -7.501317e-08 -1.611182e-07 1.324012e-06 9.175358e-08 -1.489370e-06 1.442461e-06  .. (size = 1048576, ends with -1.048254e-06);
Result_1113_0: 
-9.457514e-07 6.054996e-06 -7.305138e-06 6.446697e-06 4.164933e-06 -3.541224e-06 1.065920e-06 2.271277e-06 -9.987616e-07 4.569631e-06  .. (size = 1024, ends with 5.251849e-07);
Result_1114_0: 
7.182226e-07 -4.688392e-06 5.640633e-06 -4.990683e-06 -3.210080e-06 2.741461e-06 -8.118699e-07 -1.751018e-06 7.687860e-07 -3.530006e-06  .. (size = 1048576, ends with 3.311555e-07);
Result_1115_0: 
-2.870041e-08 3.525554e-08 1.279645e-08 3.762216e-08 1.736175e-08 7.242384e-09 -1.328662e-08 4.807877e-08 -4.751318e-08 -9.350146e-08  .. (size = 1024, ends with -1.430306e-08);
Result_1116_0: 
1.625462e-08 -1.584583e-08 -2.792677e-09 -3.784940e-09 -4.063875e-09 1.264627e-09 1.140445e-08 -1.442209e-08 1.654749e-08 2.608247e-08  .. (size = 1048576, ends with -2.080437e-08);
Result_1117_0: 
-1.851297e-14 1.216388e-14 1.516148e-15 -1.182388e-13 1.014744e-13 5.728751e-14 -8.260059e-14 8.514023e-15 -2.373102e-14 2.542411e-14  .. (size = 1024, ends with 1.040279e-13);
Result_1118_0: 
-4.995809e-09 7.468190e-09 6.145316e-10 -6.842511e-08 3.608974e-08 2.364716e-08 -3.645786e-08 1.232534e-08 -2.931779e-09 6.206978e-09  .. (size = 1048576, ends with -5.796644e-09);
Result_1119_0: 
-9.832001e-06 4.001720e-06 -2.275980e-06 -3.928241e-06 -4.840797e-07 -1.644565e-06 6.068208e-06 -3.260439e-06 -6.991760e-06 8.354605e-06  .. (size = 1024, ends with 4.967658e-06);
Result_1120_0: 
4.783215e-06 1.023947e-06 -1.022645e-07 1.606263e-06 3.550116e-07 2.364475e-06 -1.245824e-06 -1.462336e-06 -4.667810e-06 3.378594e-06  .. (size = 1024, ends with 9.033456e-06);
Result_1121_0: 
-1.904823e-04 1.265356e-04 1.494366e-05 -1.269235e-04 1.053809e-05 -1.244524e-04 1.399907e-04 -8.512859e-05 -1.442225e-04 3.183183e-05  .. (size = 524288, ends with 0.000000e+00);
Result_1122_0: 
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 2048, ends with 1.616584e-04);
Result_1124_0: 
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 31254528, ends with 0.000000e+00);
'''.strip().split("\n")


def extract_data(strs):
    data = list()
    for i in range(1, len(strs), 2):
        data.append([float(v.strip())
                     for v in strs[i].strip().split("..")[0].strip().split(" ")])
    return data

def all_allclose(a, b):
    for u in a:
        flag = False
        for v in b:
            if np.allclose(u, v, rtol=1.e-4, atol=1.e-4):
                flag = True
        if not flag:
            return False
    return True

# Inputs:
# - mfolder: model folder
# - nnfusion: nnfusion place
# generatecode : $nnfusion $mfolder/*.pb -f tensorflow -b nnfusion_engine
# cmake : cd nnfusion_rt/cuda && cmake . && make -j
# test : cd nnfusion_rt/cuda && main_test -> compare result
# clean : rm -rf nnfusion_rt


# error args, not executing command
if not os.path.exists("/usr/local/cuda/bin/nvcc"):
    logging.info("NVCC is not existed, thus skip the test.")
    exit(0)

if len(sys.argv) != 3:
    logging.error("Script doesn't have right arguments.")
    exit(1)
if not sys.argv[2].endswith("nnfusion"):
    logging.error("NNFusion cli should named \"nnfusion\"")
    exit(1)

pbfile = sys.argv[1]
nnfusion_cli = sys.argv[2]

# check
if not(os.path.exists(pbfile) and os.path.exists(nnfusion_cli)):
    logging.error("NNFusion cli or model folder is not existed.")
    exit(1)

os.system("rm -rf nnfusion_rt")
logging.info("Compiling " + pbfile)
os.system("%s %s -f tensorflow -m graph -b nnfusion >> nnfusion.log" %
            (nnfusion_cli, pbfile))
if not os.path.exists("nnfusion_rt/cuda_codegen/nnfusion_rt.cu"):
    logging.error("Failed at nnfusion compiling phase.")
    exit(2)
os.system(
    "cd nnfusion_rt/cuda_codegen/ && cmake . >> cmake.log && make -j 2>&1 >> cmake.log")
if not os.path.exists("nnfusion_rt/cuda_codegen/main_test"):
    logging.error("Failed at nvcc compiling phase.")
    exit(3)
os.system("cd nnfusion_rt/cuda_codegen/ && ./main_test > result.txt")
if not os.path.exists("nnfusion_rt/cuda_codegen/result.txt"):
    logging.error("Failed at nvcc compiling phase.")
    exit(4)
result_file = open("nnfusion_rt/cuda_codegen/result.txt")
results = result_file.readlines()
if len(results) >= 86:  # or results[1].strip() != ground_truth[pbfile]:
    a_data = extract_data(ground_truth_str)
    b_data = extract_data(results[:86])
    if not all_allclose(a_data, b_data):
        logging.error("%s has wrong result" % pbfile)
        exit(5)
    else:
        print("%s has right result!" % pbfile)
else:
    exit(6)
os.system("rm -rf nnfusion_rt")
print("All Done!.")
exit(0)
