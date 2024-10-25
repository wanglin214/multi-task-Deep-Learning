# multi-task-Deep-Learning

#the codes are supported for the manuscript :Simultaneous estimation of basement depth and density contrast by gravity anomaly via multi-task Deep Learning

#Hardware requirements: NVIDIA GPU;Program language: Python 3.10 (Pytorch 1.3);Software required: PyCharm, Anaconda 

1 the data file includes the train, validation and test set; the model file HybirdNet64.py is included in the netmodel file.

2 the trainMod.py is used to optimize the weight parameters for the multi_task DL architecture which are included in the params file.

3 the testMod.py is used to test the generalization ability of the trained parameters.

4 inv_real.py is used to invert the field gravity anomaly into basement map.

5 the related functions are included in the utils file.

#the multi-task DL architecture is constructed on a hybrid CNN-MLP Component：

![Fig6网络模型框架](https://github.com/user-attachments/assets/2e99074c-74b4-4d03-8a1d-cd303ed820fd)
