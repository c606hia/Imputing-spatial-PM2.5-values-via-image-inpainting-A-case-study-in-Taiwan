# Imputing-spatial-PM2.5-values-via-image-inpainting-A-case-study-in-Taiwan
透過圖像修補技術補空間中的 PM2.5 值：以台灣為例

## How to run RFR
### Setup  
Running on Ubuntu 18.04.6 LTS with Python 3.9.12
```
pip install torch==1.12.0
pip install torchmetrics==0.11.4
pip install torchtext==0.14.0
pip install torchvision==0.14.0
pip install numpy
pip install pandas==2.0.1
pip install matplotlib==3.7.1
pip install maskedtensor==0.10.0
pip install rich==13.3.5
pip install scikit-learn==1.2.2
```
* python run.py --model_path YOUR_MODEL_PATH_TO_LOAD --num_iters EPOCH_NUM --txt RESULT_FILE_NAME --batch_size YOUR_VATCH_SIZE --result_save_path IMAGE_SAVE_PATH --model_save_path WHERE_TO_SAVE_YOUR_MODEL
* ADD -test IF YOU WANT RUN TEST DIRECTLY

## OTHER SETTING

switch the dataset in run.py to change the random setting
* ar -> random in all time
* sr -> random in start
* switch the RFRnet in model.py to change the batchnorm setting
* bnoff -> cancel batchnorm

## Data
[data link](https://drive.google.com/file/d/1Dgm9rdVpZBIXb58lBcONK8Icj2m-2wAN/view?usp=sharing)  

點資訊(M,D,H,Lon,Lat,PM2.5)  
* a_202101.csv
* E_202101.csv
 
圖資訊  
* a_by_loc -> airbox 站點圖
* by_loc_main -> EPA 站點圖
* e+a_mask_gt_main -> EPA+airbox kriging結果
* mask -> 台灣地區遮罩
