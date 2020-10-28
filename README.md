# ABSNet
- ABSNet: Aesthetics-Based Saliency Network using Multi-Task Convolutional Network(IEEE Signal Processing Letters2020)
- @ARTICLE{ABSNet,  
           >>author={Jing Liu, Jincheng Lv, Min Yuan, Jing Zhang, and Yuting Su},  
           journal={IEEE Signal Processing Letters},   
           title={Aesthetics-Based Saliency Network using Multi-Task Convolutional Network},   
           year={2020},  
           volume={},  
           number={},  
           pages={},  
           doi={}
           }

### This is a Tensorflow implementation of IEEE Signal Processing Letters2020.

## Requisites

- opencv-python 4.1.1
- numpy 1.16.1
- tenserflow 1.14.0
- tensorboard 1.14.0
- keras 2.2.2
- scipy 1.2.1

## Usage

### 1. Clone the repository

### 2. Download the datasets

Download the following datasets and unzip them into `data` folder.

* [SALICON](http://salicon.net/challenge-2017/) dataset. 
* [MIT300](http://saliency.mit.edu/downloads.html) dataset.
* [MIT1003](http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html) dataset. 

### 3. Download the pre-trained models for backbone

Download pre-trained model [BaiduYun](https://pan.baidu.com/s/1n1ghq2miscXB_cLQKJddPw) (pwd: **4jk1**).  
This pre-trained model is for multi-task dalited resnet(MT-DRN), you should set 'weightfile' correctly.

### 4. Test our model

- For testing sal branch, you should set config.yaml correctly and load pretrained pkl correctly, and then run main_sal_drn.py.

### 5. Download code for metrics, evaluation, and more.
https://github.com/cvzoya/saliency

### 6. Test Results of MT-DRN on SALICON Validation/TEST Set and MIT1003 DataSet.
- SALICON Validation Set: Link：https://pan.baidu.com/s/1dPkwMpotQQhSQql4r42TFQ  pwd: oxgw 
- SALICON Test Set: Link: https://pan.baidu.com/s/1FygM6C1Su_6bcCivBW3QQw  pwd: 6yqq 
- MIT1003 DataSet: Link: https://pan.baidu.com/s/1z-q9BLrRBql2AZUUS_5mnA  pwd: cwgx 

### 7. Some Comparsions in our paper.
![Image text](https://github.com/TJUMMG/ABSNet/blob/main/PNG/%E5%9B%BE%E7%89%871.png)
![Image text](https://github.com/TJUMMG/ABSNet/blob/main/PNG/%E5%9B%BE%E7%89%872.png)

