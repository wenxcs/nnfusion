import logging, os
from testcases.testcase import *

class TestSingleOutput(TestCase):
    def __init__(self, casename, strdata, url = "", format = "tensorflow", extra_args = "", backend = None):
        self.casename = casename
        self.ground_truth = [float(v.strip()) for v in strdata.split("..")[0].strip().split(" ")]
        self.url = url
        self.rtol = 1.e-2
        self.atol = 1.e-2
        self.format = format
        self.extra_args = extra_args
        self.backend = backend or []
    
    # Get data from output of main_test
    def allclose(self, raw_strdata):
        floatdata = [float(v.strip()) for v in raw_strdata[1].split("..")[0].strip().split(" ")]
        return TestCase.allclose(self,floatdata)

class TestMultiOutput(TestCase):
    def __init__(self, casename, strdata, url = "", format = "tensorflow", extra_args = "", backend = None):
        self.casename = casename
        self.ground_truth = self.extract_data(strdata.split("\n"))
        self.url = url
        self.rtol = 1.e-2
        self.atol = 1.e-2
        self.format = format
        self.extra_args = extra_args
        self.backend = backend or []

    def extract_data(self, strs):
        data = list()
        for i in range(1, len(strs), 2):
            data.append([float(v.strip())
                for v in strs[i].strip().split("..")[0].strip().split(" ")])
        return data

    def all_allclose(self, a, b):
        cnt = 0
        for u in a:
            flag = False
            for v in b:
                if len(u) == len(v) and np.allclose(u, v, rtol=self.rtol, atol=self.atol):
                    flag = True
            if not flag:
                print("Mismatch#%d: %s"%(cnt, u))
                return False
            cnt += 1
        return True

    def allclose(self, raw_strdata):
        if not self.all_allclose(self.extract_data(raw_strdata), self.ground_truth):
            logging.error("%s has wrong result." % (self.casename))
            return False
        return True

# Add Test cases
TestCases.append(TestSingleOutput("frozen_random-weights_bert_large.pb", "0.001335 0.001490 0.000675 0.002558 0.000761 0.001435 0.000518 0.001516 0.000738 0.001183  .. (size = 1001, ends with 0.000281);", "nnfusion/frozen_models/frozen_random-weights_bert_large.pb"))
TestCases.append(TestSingleOutput("frozen_alexnet_infer_batch_1.pb", "0.000914 -0.030341 -0.006662 -0.010238 0.014080 0.024311 0.006832 -0.035370 0.017920 0.038856  .. (size = 1001, ends with 0.022597);", "nnfusion/frozen_models/frozen_alexnet_infer_batch_1.pb"))
TestCases.append(TestSingleOutput("frozen_resnet50_infer_batch_1.pb", "-0.001597 0.030608 -0.002212 0.037812 0.030037 0.039713 -0.006352 0.051142 0.016946 -0.009263  .. (size = 1001, ends with 0.043752);", "nnfusion/frozen_models/frozen_resnet50_infer_batch_1.pb"))
TestCases.append(TestSingleOutput("frozen_vgg11_infer_batch_1.pb", "-0.003832 -0.008819 0.004029 -0.003441 0.012382 -0.003776 0.001756 -0.014141 -0.005059 -0.001504  .. (size = 1001, ends with 0.008471);", "nnfusion/frozen_models/frozen_vgg11_infer_batch_1.pb"))
TestCases.append(TestSingleOutput("frozen_inception3_infer_batch_1.pb", "-0.000079 -0.000875 -0.000871 -0.000491 0.000316 -0.000246 0.000187 0.000502 -0.000710 -0.000311  .. (size = 1001, ends with -0.000542);", "nnfusion/frozen_models/frozen_inception3_infer_batch_1.pb"))

TestCases.append(TestMultiOutput("frozen_bert_train.const_folded.pb", '''Result_1077_0:
1.544510e-03 5.187932e-04 8.498678e-04 1.672361e-03 5.371213e-04 4.475404e-04 1.268131e-03 5.944323e-04 1.431170e-03 3.446568e-04  .. (size = 1024, ends with 5.337071e-04);
Result_1078_0:
6.931930e+00  .. (size = 1, ends with 6.931930e+00);
Result_1125_0:
-1.907903e-04 1.271152e-04 1.543891e-05 -1.271294e-04 1.068962e-05 -1.242892e-04 1.388441e-04 -8.514979e-05 -1.445845e-04 3.206256e-05  .. (size = 524288, ends with 0.000000e+00);
''', "nnfusion/frozen_models/frozen_bert_train.const_folded.pipline.pb"))

if "NNFUSION_BLOCKFUSION" in os.environ:
    TestCases.append(TestSingleOutput("frozen_lstm_infer_batch_1.const_folded.pb", '''-2.988511e-01 2.256856e-01 -1.658812e-01 1.369080e-01 -3.430056e-01 -2.845020e-01 -2.236749e-01 3.384702e-01 -8.138472e-02 -1.279101e-01  .. (size = 256, ends with -3.586947e-02);''', "nnfusion/frozen_models/blockfusion_models/frozen_lstm_infer_batch_1.const_folded.pb"))
    TestCases.append(TestSingleOutput("frozen_resnext29_cifar_infer_batch_1.const_folded.pb", '''-1.366695e+01 -2.129843e+00 -1.658100e+01 2.353264e-01 -8.940872e+00 -1.665310e+01 -3.185829e+01 2.800117e+00 -3.312237e+00 -9.553578e+00  .. (size = 10, ends with -9.553578e+00);''', "nnfusion/frozen_models/blockfusion_models/frozen_resnext29_cifar_infer_batch_1.const_folded.pb"))
    TestCases.append(TestMultiOutput("frozen_seq2seq_infer_batch_1.const_folded.pb", '''Result_32450_0:
1.081029e-02 6.760932e-03 5.091093e-03 -2.683104e-03 9.393914e-03 7.971724e-04 8.058791e-03 6.377231e-04 -1.490641e-03 -5.653641e-03  .. (size = 128, ends with 1.050940e-02);
Result_32451_0:
-7.735598e-05 1.048572e-04 1.761352e-05 1.030918e-04 -7.215205e-05 4.760705e-06 9.691977e-05 5.477592e-06 -5.094828e-05 5.035079e-05  .. (size = 128, ends with 1.024876e-06);''', "nnfusion/frozen_models/blockfusion_models/frozen_seq2seq_infer_batch_1.const_folded.pb"))
    TestCases.append(TestSingleOutput("frozen_nasnet_mobile_imagenet_infer_batch_1.const_folded.pb", '''0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 1000, ends with 0.000000e+00);''', "nnfusion/frozen_models/blockfusion_models/frozen_nasnet_mobile_imagenet_infer_batch_1.const_folded.pb"))
    # TestCases.append(TestSingleOutput("frozen_deepspeech2_infer_batch_1.const_folded.pb", '''''', "nnfusion/frozen_models/blockfusion_models/frozen_deepspeech2_infer_batch_1.const_folded.pb"))
    
# Graph which has variable and optimizer;
TestCases.append(TestMultiOutput("frozen_bert_train_bs_s1_seq_128_layer2_sgd.const_folded.pb", 
'''Result_1226_0:
4.852030e+00  .. (size = 1, ends with 4.852030e+00);
Result_1227_0:
1.000000e+00 1.000001e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 128, ends with 1.000000e+00);
Result_1228_0:
1.000000e+00 1.000001e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 131072, ends with 1.000000e+00);
Result_1229_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1230_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1048576, ends with 1.000000e+00);
Result_1231_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1232_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1233_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1234_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 4194304, ends with 1.000000e+00);
Result_1235_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 4096, ends with 1.000000e+00);
Result_1236_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 4194304, ends with 1.000000e+00);
Result_1237_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1238_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1239_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1240_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1048576, ends with 1.000000e+00);
Result_1241_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1242_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1048576, ends with 1.000000e+00);
Result_1243_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1244_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1245_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1048576, ends with 1.000000e+00);
Result_1246_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1048576, ends with 1.000000e+00);
Result_1247_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1248_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1249_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1250_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 4194304, ends with 1.000000e+00);
Result_1251_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 4096, ends with 1.000000e+00);
Result_1252_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 4194304, ends with 1.000000e+00);
Result_1253_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1254_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1255_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1256_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1048576, ends with 1.000000e+00);
Result_1257_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1258_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1048576, ends with 1.000000e+00);
Result_1259_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1260_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1261_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1048576, ends with 1.000000e+00);
Result_1262_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1048576, ends with 1.000000e+00);
Result_1263_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1264_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 1024, ends with 1.000000e+00);
Result_1265_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 524288, ends with 1.000000e+00);
Result_1266_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 31254528, ends with 1.000000e+00);
Result_1267_0:
1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00  .. (size = 2048, ends with 1.000000e+00);
''', "nnfusion/frozen_models/frozen_bert_train_bs_s1_seq_128_layer2_sgd.const_folded.pb"))

# onnx inference
TestCases.append(TestMultiOutput("bert_for_pretraining_without_loss_optimized_layer_norm_no_trainable_dropout.onnx", '''Result_397_0:
1.223377e+00 -1.294775e+00 1.223377e+00 -1.294775e+00 1.223377e+00 -1.294775e+00  .. (size = 6, ends with -1.294775e+00);
Result_398_0:
-1.709945e+00 -2.038319e+00 -2.782065e+00 -1.859875e+00 -2.576831e+00 -2.621146e+00 -2.706791e+00 -2.926065e+00 -2.851139e+00 -3.096867e+00  .. (size = 46891008, ends with 0.000000e+00);''', "nnfusion/frozen_models/onnx_inference/bert_for_pretraining_without_loss_optimized_layer_norm_no_trainable_dropout.onnx", "onnx", "-p \"batch:3;sequence:512;dynamic_prediction_count:20\""))
# onnx training
TestCases.append(TestMultiOutput("bert_for_pretraining_without_loss_optimized_layer_norm_bw_grad_output.onnx", '''Result_1604_0:
1.223377e+00 -1.294775e+00 1.223377e+00 -1.294775e+00 1.223377e+00 -1.294775e+00  .. (size = 6, ends with -1.294775e+00);
Result_1605_0:
2.595676e+00  .. (size = 1, ends with 2.595676e+00);
Result_1606_0:
-1.709945e+00 -2.038319e+00 -2.782065e+00 -1.859875e+00 -2.576831e+00 -2.621146e+00 -2.706791e+00 -2.926065e+00 -2.851139e+00 -3.096867e+00  .. (size = 46891008, ends with 0.000000e+00);
Result_1607_0:
1.263408e+01  .. (size = 1, ends with 1.263408e+01);
Result_1608_0:
1.522975e+01  .. (size = 1, ends with 1.522975e+01);
Result_1609_0:
-4.891759e-06 -6.159856e-06 -6.066868e-06 1.880386e-07 -5.193404e-06 1.136649e-05 3.329500e-06 3.001903e-06 -4.308903e-06 -4.625066e-07  .. (size = 31260672, ends with -6.961398e-07);
Result_1610_0:
2.000000e+00  .. (size = 1, ends with 2.000000e+00);
Result_1604_0:
3.229324e+03 -3.229144e+03 3.229324e+03 -3.229144e+03 3.229324e+03 -3.229144e+03  .. (size = 6, ends with -3.229144e+03);
Result_1605_0:
inf  .. (size = 1, ends with inf);
Result_1606_0:
1.114992e+03 1.444014e+01 1.038971e+03 1.699926e+03 1.001128e+03 1.027282e+03 4.636901e+02 7.454852e+02 8.460545e+02 9.482241e+02  .. (size = 46891008, ends with 7.806078e+02);
Result_1607_0:
inf  .. (size = 1, ends with inf);
Result_1608_0:
inf  .. (size = 1, ends with inf);
Result_1609_0:
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 31260672, ends with 0.000000e+00);
Result_1610_0:
3.000000e+00  .. (size = 1, ends with 3.000000e+00);
Result_1604_0:
7.856802e+02 -7.855007e+02 7.856802e+02 -7.855007e+02 7.856802e+02 -7.855007e+02  .. (size = 6, ends with -7.855007e+02);
Result_1605_0:
inf  .. (size = 1, ends with inf);
Result_1606_0:
-3.639895e+03 5.112187e+01 -3.381732e+03 -5.607864e+03 -3.261618e+03 -3.342046e+03 -1.468408e+03 -2.408962e+03 -2.738109e+03 -3.082231e+03  .. (size = 46891008, ends with -2.581743e+03);
Result_1607_0:
0.000000e+00  .. (size = 1, ends with 0.000000e+00);
Result_1608_0:
inf  .. (size = 1, ends with inf);
Result_1609_0:
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 31260672, ends with 0.000000e+00);
Result_1610_0:
4.000000e+00  .. (size = 1, ends with 4.000000e+00);
Result_1604_0:
2.997780e+03 -2.997916e+03 2.997780e+03 -2.997916e+03 2.997780e+03 -2.997916e+03  .. (size = 6, ends with -2.997916e+03);
Result_1605_0:
inf  .. (size = 1, ends with inf);
Result_1606_0:
-1.046690e+04 7.833135e+01 -9.721594e+03 -1.616823e+04 -9.373619e+03 -9.609612e+03 -4.223886e+03 -6.919831e+03 -7.868719e+03 -8.858474e+03  .. (size = 46891008, ends with -7.374409e+03);
Result_1607_0:
inf  .. (size = 1, ends with inf);
Result_1608_0:
inf  .. (size = 1, ends with inf);
Result_1609_0:
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 31260672, ends with 0.000000e+00);
Result_1610_0:
5.000000e+00  .. (size = 1, ends with 5.000000e+00);
Result_1604_0:
3.928722e+03 -3.928856e+03 3.928722e+03 -3.928856e+03 3.928722e+03 -3.928856e+03  .. (size = 6, ends with -3.928856e+03);
Result_1605_0:
inf  .. (size = 1, ends with inf);
Result_1606_0:
-1.385244e+04 1.101200e+02 -1.286818e+04 -2.134766e+04 -1.241255e+04 -1.272132e+04 -5.606303e+03 -9.175699e+03 -1.042558e+04 -1.173242e+04  .. (size = 46891008, ends with -9.775431e+03);
Result_1607_0:
1.360564e+00  .. (size = 1, ends with 1.360564e+00);
Result_1608_0:
inf  .. (size = 1, ends with inf);
Result_1609_0:
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00  .. (size = 31260672, ends with 0.000000e+00);
Result_1610_0:
6.000000e+00  .. (size = 1, ends with 6.000000e+00);''', "nnfusion/frozen_models/onnx_training/bert_for_pretraining_without_loss_optimized_layer_norm_bw_grad_output.onnx", "onnx", "-p \"batch:3;sequence:512;dynamic_prediction_count:20\"", ["CUDA"]))
