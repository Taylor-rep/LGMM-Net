import numpy as np
import glob
import os
from PIL import Image

# 参数
height = 256
width = 256
channels = 3

# 数据路径
Dataset_add = './ISIC2018/'
Tr_add = 'ISIC2018_Task1-2_Training_Input'

Tr_list = glob.glob(os.path.join(Dataset_add, Tr_add, '*.jpg'))
num_samples = len(Tr_list)  # 实际样本数量（应该是2594）

# 初始化数组
Data_train_2018 = np.zeros((num_samples, height, width, channels), dtype=np.float64)
Label_train_2018 = np.zeros((num_samples, height, width), dtype=np.float64)

print('Reading ISIC 2018')
for idx in range(num_samples):
    print(f'Processing {idx + 1}/{num_samples}')
    
    # 图像读取与缩放
    img_path = Tr_list[idx]
    img = Image.open(img_path).convert('RGB')
    img = img.resize((width, height), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float64)
    Data_train_2018[idx, :, :, :] = img_array

    # 获取文件名
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    # 掩码路径
    mask_path = os.path.join(Dataset_add, 'ISIC2018_Task1_Training_GroundTruth', f'{img_name}_segmentation.png')
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((width, height), Image.Resampling.BILINEAR)
    mask_array = np.array(mask, dtype=np.float64)
    Label_train_2018[idx, :, :] = mask_array

print('Finished reading ISIC 2018.')

# 划分数据集
Train_img = Data_train_2018[0:1815]
Validation_img = Data_train_2018[1815:1815+259]
Test_img = Data_train_2018[1815+259:]

Train_mask = Label_train_2018[0:1815]
Validation_mask = Label_train_2018[1815:1815+259]
Test_mask = Label_train_2018[1815+259:]

# 保存为 .npy 文件
np.save('data_train', Train_img)
np.save('data_val', Validation_img)
np.save('data_test', Test_img)

np.save('mask_train', Train_mask)
np.save('mask_val', Validation_mask)
np.save('mask_test', Test_mask)

print("Train data shape:", Train_img.shape)
print("Validation data shape:", Validation_img.shape)
print("Test data shape:", Test_img.shape)

print("Train mask shape:", Train_mask.shape)
print("Validation mask shape:", Validation_mask.shape)
print("Test mask shape:", Test_mask.shape)
