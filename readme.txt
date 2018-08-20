This code is for our paper "Hyperspectral image classification with Squeeze Multi-Bias network". 
If you use this code, please kindly cite our paper:
Leyuan Fang, Guangyun Liu, Shutao Li, Pedram Ghamisi, and Jón Atli Benediktsson, "Hyperspectral image classification with Squeeze Multi-Bias network", accepted by the IEEE TGRS.

If you have any questions, please contact us. 
Email:  liuguangyun@hnu.edu.cn  fangleyuan@gmail.com

This code is tested on the Ubuntu 16.04 system and caffe framework. Before running this code, 
you should correctly install ubuntu system and caffe framework. For caffe installation, you can 
refer to this guildeline "http://caffe.berkeleyvision.org/installation.html". After installing the caffe framework,
you should download the caffe-bias from this guildeline "https://github.com/hli2020/caffe/tree/bias", and recompile it.

After correctly installing and compiling the environment, you can run this code by the following procedures. 
For the convenience, we take the Indian Pines image as a example. For the Houston and 
Washington DC images, please make some changes referring to codes. 
(1) Opening the matlab and changing the current path to the unzipped path, 
    running the "generating_data_h5.m" to generate the training and test samples, which are saved in 
    the ./data_h5/indian. Noteworthily, you should save the labels and indexs of the training and test samples.
(2) Opening the terminal and changing the current path to the unzipped path, then running this script:
    "sh train_SMBN.sh. 
    This script executes the training of SMBN and obtains a model.
(3) After training, running the following script to test network in the terminal same as (2):
    "sh test_SMBN.sh >& test_indian_pines.log"
    This script executes the test of SMBN and generate the corresponding test log file.
(4) running the "extract_prob.m" in matlab. 
    This script is used to extract probability (indian_pines_prob.txt) from the "test_indian_pines.log".
(5) running the "calculating_result.m" to calculate the matrics (OA, AA, Kappa, CA) and draw classification map.
