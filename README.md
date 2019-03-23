# tf-mobilenet-SSD

## 0. Acknowledgement

This repo is higly inspired by [SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow) thanks to [@Paul Balanca](https://github.com/balancap)'s excellent work!



## 1. Prepare tf_records for training

Please download the **VOC2007** datasets from

- [PASCAL VOC 2007 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
- [PASCAL VOC 2007 test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)



Place the **VOC2007** datasets as

> \- VOC2007
>
> ​    -- train
>
> ​    -- test

where the downloaded voc_07_trainval and voc_07_test are unzip under the**./VOC2007/train** and **./VOC2007/test **respectively.

then run `sh convert_train_dataset.sh` to convert the raw images datasets into tf_records file under the directory **./tf_records** .

## 2. Run training

The pretrained **Mobilenet-V1-224** checkpoint is under the **./ckpt**

Please run `sh train.sh` for training, the checkpoints are saved

under **./logs**

## 3. Run evaluation

Please run `sh eval.sh` for evaluation, 

make sure you have modified test dataset path in the **convert_train_dataset.sh** and 

use `sh convert_train_dataset.sh to prepare test tf_records in advance.`



