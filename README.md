# Imputing-spatial-PM2.5-values-via-image-inpainting-A-case-study-in-Taiwan
透過圖像修補技術補空間中的 PM2.5 值：以台灣為例

# How to run RFR:

python run.py --model_path YOUR_MODEL_PATH_TO_LOAD --num_iters EPOCH_NUM --txt RESULT_FILE_NAME --batch_size YOUR_VATCH_SIZE --result_save_path IMAGE_SAVE_PATH --model_save_path WHERE_TO_SAVE_YOUR_MODEL
ADD -test IF YOU WANT RUN TEST DIRECTLY

# OTHER SETTING:

swith the dataset in run.py to change the random setting
ar -> random in all time
sr -> random in start
switch the RFRnet in model.py to change the batchnorm setting
bnoff -> cancel batchnorm
