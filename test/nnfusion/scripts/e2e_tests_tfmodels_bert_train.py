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
1.544510e-03 5.187932e-04 8.498678e-04 1.672361e-03 5.371213e-04 4.475404e-04 1.268131e-03 5.944323e-04 1.431170e-03 3.446568e-04  .. (size = 1024, ends with 5.337071e-04);
Result_1078_0: 
6.931930e+00  .. (size = 1, ends with 6.931930e+00);
Result_1079_0: 
8.018042e-07 -5.185244e-04 4.406168e-07 8.683848e-07 2.783083e-07 2.318530e-07 6.579847e-07 3.080372e-07 7.428076e-07 1.785185e-07  .. (size = 1024, ends with 2.765375e-07);
Result_1080_0: 
-5.659449e-07 3.659949e-04 -3.110046e-07 -6.129401e-07 -1.964409e-07 -1.636510e-07 -4.644315e-07 -2.174247e-07 -5.243028e-07 -1.260054e-07  .. (size = 1048576, ends with 1.265145e-07);
Result_1081_0: 
-1.332740e-05 2.287762e-06 1.171699e-05 -1.878105e-05 1.528210e-05 -1.544563e-05 -1.653889e-05 -4.949784e-06 1.949194e-05 2.867539e-06  .. (size = 1024, ends with 1.354560e-06);
Result_1082_0: 
9.835939e-06 -1.688423e-06 -8.647419e-06 1.386086e-05 -1.127855e-05 1.139924e-05 1.220609e-05 3.653059e-06 -1.438552e-05 -2.116312e-06  .. (size = 1048576, ends with 1.672330e-06);
Result_1083_0: 
4.010397e-06 7.705847e-08 -1.234441e-06 -5.667698e-08 -3.667661e-06 -9.579394e-06 3.199177e-06 -1.424704e-06 -1.590944e-06 -4.969962e-06  .. (size = 1024, ends with 3.132574e-06);
Result_1084_0: 
-2.959768e-06 2.305249e-08 8.396623e-07 7.497442e-09 -2.945438e-06 3.038705e-06 6.100580e-07 1.787325e-07 -9.075532e-07 -2.149023e-06  .. (size = 1024, ends with 3.867455e-06);
Result_1085_0: 
4.160983e-06 1.599335e-07 -8.190249e-07 1.489618e-07 -3.519679e-06 -8.815227e-06 3.144247e-06 -1.147778e-06 -1.491618e-06 -4.653040e-06  .. (size = 1024, ends with 2.801462e-06);
Result_1088_0: 
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 4194304, ends with 0.000000e+00);
Result_1089_0: 
1.213113e-07 9.373085e-08 -1.016309e-06 -2.173354e-06 -4.942209e-06 1.403874e-06 4.430128e-07 2.088841e-07 6.118522e-06 -3.005498e-07  .. (size = 4096, ends with 1.157557e-06);
Result_1092_0: 
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 4194304, ends with 0.000000e+00);
Result_1093_0: 
1.898215e-07 2.213361e-06 -2.027665e-06 -2.146940e-06 -4.788152e-06 -5.774094e-06 3.902149e-06 -1.920934e-06 1.317687e-06 -3.694547e-06  .. (size = 1024, ends with 4.127382e-06);
Result_1094_0: 
-1.104422e-07 1.587524e-06 1.507623e-06 1.027575e-06 -3.132145e-06 3.607176e-06 -9.124978e-07 3.215603e-08 1.347644e-06 -1.463923e-06  .. (size = 1024, ends with 7.932575e-06);
Result_1095_0: 
4.268575e-07 2.326963e-06 -1.731265e-06 -1.863802e-06 -4.514025e-06 -5.400975e-06 4.035154e-06 -1.670638e-06 1.432965e-06 -3.429388e-06  .. (size = 1024, ends with 4.125680e-06);
Result_1096_0: 
-1.989582e-07 -1.084597e-06 8.069425e-07 8.687182e-07 2.103987e-06 2.517394e-06 -1.880785e-06 7.786841e-07 -6.679049e-07 1.598438e-06  .. (size = 1048576, ends with 8.027499e-07);
Result_1097_0: 
-3.154378e-06 9.371539e-06 -1.530389e-07 1.880959e-06 -3.755856e-06 -2.968825e-07 3.099908e-06 -4.681477e-06 -5.594826e-06 4.519275e-06  .. (size = 1024, ends with 9.370412e-07);
Result_1098_0: 
3.372723e-06 -1.002024e-05 1.636322e-07 -2.011158e-06 4.015838e-06 3.174327e-07 -3.314484e-06 5.005528e-06 5.982102e-06 -4.832098e-06  .. (size = 1048576, ends with 8.945815e-07);
Result_1099_0: 
8.040404e-09 2.656325e-08 4.851545e-09 1.172066e-08 3.690670e-09 -4.288581e-09 -4.824872e-08 3.111869e-08 2.826362e-08 1.763753e-08  .. (size = 1024, ends with -3.409675e-08);
Result_1100_0: 
-3.890124e-09 -1.285188e-08 -2.347284e-09 -5.670712e-09 -1.785627e-09 2.074910e-09 2.334379e-08 -1.505590e-08 -1.367456e-08 -8.533425e-09  .. (size = 1048576, ends with -6.370678e-08);
Result_1101_0: 
-3.053113e-15 -2.220446e-14 1.310063e-14 -6.439294e-15 -1.332268e-14 -7.771561e-15 -5.218048e-15 -9.547918e-15 5.773160e-15 3.153033e-14  .. (size = 1024, ends with -9.325873e-15);
Result_1102_0: 
-4.420960e-09 -3.793448e-08 2.532219e-08 -1.651176e-08 -3.146413e-08 -1.856118e-08 -1.295095e-08 -2.021253e-08 2.267732e-08 4.465061e-08  .. (size = 1048576, ends with -3.355180e-09);
Result_1103_0: 
-1.748085e-06 1.360814e-06 -8.193892e-06 -2.765757e-06 -5.751328e-06 -1.659536e-06 3.239632e-06 1.326028e-06 6.183017e-07 7.851202e-07  .. (size = 1024, ends with 3.855787e-06);
Result_1104_0: 
2.113597e-06 1.587375e-06 4.822906e-06 1.257274e-06 -2.003219e-06 2.456821e-06 -2.114418e-06 7.362243e-07 1.228597e-06 -1.517283e-06  .. (size = 1024, ends with 7.441260e-06);
Result_1105_0: 
-1.596644e-06 1.322614e-06 -7.644470e-06 -2.542030e-06 -5.354545e-06 -1.495097e-06 3.098485e-06 1.307198e-06 6.274302e-07 7.965976e-07  .. (size = 1024, ends with 3.669268e-06);
Result_1108_0: 
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 4194304, ends with 0.000000e+00);
Result_1109_0: 
-2.893574e-06 -3.612715e-06 -6.950969e-07 1.056151e-06 -5.786964e-08 -4.082796e-06 -1.023971e-06 -7.521986e-07 9.901407e-07 -2.878798e-06  .. (size = 4096, ends with -1.432845e-06);
Result_1112_0: 
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 4194304, ends with 0.000000e+00);
Result_1113_0: 
-9.071521e-06 4.249607e-06 -6.497080e-06 -4.474256e-06 -1.845125e-07 -5.094379e-07 5.464687e-06 5.393709e-07 -5.945678e-06 6.048203e-06  .. (size = 1024, ends with 6.630944e-06);
Result_1114_0: 
7.091621e-06 6.774977e-07 1.336122e-06 1.404333e-06 4.423827e-07 2.153067e-06 -1.995372e-06 4.766699e-07 -5.118230e-06 2.074395e-06  .. (size = 1024, ends with 1.001714e-05);
Result_1115_0: 
-8.794153e-06 3.969339e-06 -6.281978e-06 -4.406192e-06 -2.984084e-07 -6.264560e-07 5.135205e-06 3.617363e-07 -5.831268e-06 5.637904e-06  .. (size = 1024, ends with 6.224062e-06);
Result_1116_0: 
-2.252126e-06 1.016680e-06 -1.607630e-06 -1.128478e-06 -7.627008e-08 -1.609879e-07 1.315190e-06 9.201190e-08 -1.493324e-06 1.443113e-06  .. (size = 1048576, ends with -1.033119e-06);
Result_1117_0: 
-9.587338e-07 6.051191e-06 -7.304034e-06 6.443077e-06 4.173269e-06 -3.528910e-06 1.064826e-06 2.268939e-06 -9.877965e-07 4.566366e-06  .. (size = 1024, ends with 5.272806e-07);
Result_1118_0: 
7.282027e-07 -4.685446e-06 5.639809e-06 -4.987862e-06 -3.216520e-06 2.731884e-06 -8.110813e-07 -1.749285e-06 7.603487e-07 -3.527439e-06  .. (size = 1048576, ends with 3.419552e-07);
Result_1119_0: 
-2.871021e-08 3.526333e-08 1.283677e-08 3.760022e-08 1.724133e-08 7.268226e-09 -1.332717e-08 4.806781e-08 -4.750936e-08 -9.351647e-08  .. (size = 1024, ends with -1.989211e-08);
Result_1120_0: 
1.630416e-08 -1.593508e-08 -2.874272e-09 -3.847246e-09 -3.964546e-09 1.375307e-09 1.133465e-08 -1.442670e-08 1.657000e-08 2.607115e-08  .. (size = 1048576, ends with -3.270689e-08);
Result_1121_0: 
2.489675e-14 -1.094957e-14 -2.862294e-15 1.044720e-13 -9.181544e-14 -5.073719e-14 7.499557e-14 -1.847134e-14 8.021361e-15 -2.225997e-14  .. (size = 1024, ends with -1.976197e-14);
Result_1122_0: 
-4.909587e-09 7.523746e-09 5.170608e-10 -6.830059e-08 3.592241e-08 2.352415e-08 -3.637915e-08 1.224611e-08 -2.912289e-09 6.132526e-09  .. (size = 1048576, ends with -8.453666e-09);
Result_1123_0: 
-9.844985e-06 4.015466e-06 -2.269122e-06 -3.934371e-06 -4.858296e-07 -1.640125e-06 6.027154e-06 -3.259017e-06 -6.998987e-06 8.361216e-06  .. (size = 1024, ends with 4.980716e-06);
Result_1124_0: 
4.814303e-06 9.958537e-07 -8.088949e-08 1.575477e-06 3.437940e-07 2.361214e-06 -1.214349e-06 -1.469330e-06 -4.657990e-06 3.383823e-06  .. (size = 1024, ends with 9.051908e-06);
Result_1125_0: 
-1.907903e-04 1.271152e-04 1.543891e-05 -1.271294e-04 1.068962e-05 -1.242892e-04 1.388441e-04 -8.514979e-05 -1.445845e-04 3.206256e-05  .. (size = 524288, ends with 0.000000e+00);
Result_1126_0: 
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 2048, ends with 1.620747e-04);
Result_1129_0: 
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 31254528, ends with 0.000000e+00);
'''.strip().split("\n")

def extract_data(strs):
    data = list()
    for i in range(1, len(strs), 2):
        data.append([float(v.strip())
                     for v in strs[i].strip().split("..")[0].strip().split(" ")])
    return data

def all_allclose(a, b):
    cnt = 0
    for u in a:
        flag = False
        for v in b:
            if np.allclose(u, v, rtol=1.e-4, atol=1.e-4):
                flag = True
        if not flag:
            print("Mismatch#%d: %s"%(cnt, u))
            return False
        cnt += 1
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
    a_data = extract_data(ground_truth_str[:86])
    b_data = extract_data(results[:86])
    if not all_allclose(b_data, a_data):
        logging.error("%s has wrong result" % pbfile)
        exit(5)
    else:
        print("%s has right result!" % pbfile)
else:
    exit(6)
os.system("rm -rf nnfusion_rt")
print("All Done!.")
exit(0)
