# -*- coding: utf-8 -*-
"""
Code created on Sat Jun  8 18:15:43 2019
@author: Reza Azad
"""

"""
Reminder added on December 6, 2023. 
Reminder Created on Wed Dec 6 2023
@author: Renkai Wu
1.Note that the scipy package should need to be degraded. Otherwise, you need to modify the following code. ##scipy==1.2.1
2.Add a name that displays the file to be processed. If it does not appear, the output npy file is incorrect.
"""
import numpy as np
import scipy.io as sio
from PIL import Image
import glob
import os

# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare ISIC 2017 data set #################################################
Dataset_add = '../ISIC2017/'
Tr_add = 'ISIC2017_Task1-2_Training_Input'

Tr_list = glob.glob(Dataset_add + Tr_add + '/*.jpg')
# It contains 2000 training samples
Data_train_2017    = np.zeros([2000, height, width, channels])
Label_train_2017   = np.zeros([2000, height, width])

print('Reading ISIC 2017')
print(Tr_list)
for idx in range(len(Tr_list)):
    print(idx+1)
    # 读取图片并缩放
    img = Image.open(Tr_list[idx]).convert('RGB')
    img = img.resize((width, height), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float64)
    Data_train_2017[idx, :,:,:] = img_array

    # 获取图片名（不带扩展名）
    img_name = os.path.splitext(os.path.basename(Tr_list[idx]))[0]
    # 构造mask路径
    mask_path = Dataset_add + 'ISIC2017_Task1_Training_GroundTruth/' + img_name + '_segmentation.png'
    # 读取mask并缩放
    img2 = Image.open(mask_path).convert('L')
    img2 = img2.resize((width, height), Image.Resampling.BILINEAR)
    img2_array = np.array(img2, dtype=np.float64)
    Label_train_2017[idx, :,:] = img2_array    
         
print('Reading ISIC 2017 finished')

################################################################ Make the train and test sets ########################################    
# We consider 1250 samples for training, 150 samples for validation and 600 samples for testing

Train_img      = Data_train_2017[0:1250,:,:,:]
Validation_img = Data_train_2017[1250:1250+150,:,:,:]
Test_img       = Data_train_2017[1250+150:2000,:,:,:]

Train_mask      = Label_train_2017[0:1250,:,:]
Validation_mask = Label_train_2017[1250:1250+150,:,:]
Test_mask       = Label_train_2017[1250+150:2000,:,:]


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)



