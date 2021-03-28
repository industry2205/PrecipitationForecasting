import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import tensorflow as tf

# Generation 생성 시, 고정적으로 이미지 Augmentation 추가를 위한 
def augmentation(x_train,y_train):
    y_train = y_train.reshape(40,40,1)

    # Degree 90
    rotate_X_90 = np.zeros_like(x_train)
    rotate_Y_90 = np.zeros_like(y_train)

    rotate_x=np.zeros([x_train.shape[0],x_train.shape[1],10])
    rotate_y=np.zeros([y_train.shape[0],y_train.shape[1],1])

    for i in range(10):
        rotate_x[:,:,i]=np.rot90(x_train[:,:,i])

    rotate_y=np.rot90(y_train)

    rotate_X_90[:,:,:] = rotate_x
    rotate_Y_90 = rotate_y.reshape(40,40,1)

    # Degree 180
    rotate_X_180 = np.zeros_like(x_train)
    rotate_Y_180 = np.zeros_like(y_train)

    rotate_x=np.zeros([x_train.shape[0],x_train.shape[1],10])
    rotate_y=np.zeros([y_train.shape[0],y_train.shape[1],1])

    for i in range(10):
        rotate_x[:,:,i]=np.rot90(x_train[:,:,i])
        rotate_x[:,:,i]=np.rot90(rotate_x[:,:,i])

    rotate_y=np.rot90(y_train)
    rotate_y=np.rot90(rotate_y)


    rotate_X_180[:,:,:] = rotate_x
    rotate_Y_180 = rotate_y.reshape(40,40,1)

    # Degree 270
    rotate_X_270 = np.zeros_like(x_train)
    rotate_Y_270 = np.zeros_like(y_train)

    rotate_x=np.zeros([x_train.shape[0],x_train.shape[1],10])
    rotate_y=np.zeros([y_train.shape[0],y_train.shape[1],1])

    for i in range(10):
        rotate_x[:,:,i]=np.rot90(x_train[:,:,i])
        rotate_x[:,:,i]=np.rot90(rotate_x[:,:,i])
        rotate_x[:,:,i]=np.rot90(rotate_x[:,:,i])
    rotate_y=np.rot90(y_train)
    rotate_y=np.rot90(rotate_y)
    rotate_y=np.rot90(rotate_y)

    rotate_X_270[:,:,:] = rotate_x
    rotate_Y_270 = rotate_y.reshape(40,40,1)

    x_train = tf.stack([x_train, rotate_X_90,rotate_X_180,rotate_X_270], axis = 0)
    y_train = tf.stack([y_train, rotate_Y_90,rotate_Y_180,rotate_Y_270], axis = 0)
    x_train = x_train.numpy()
    y_train = y_train.numpy()
    
    x_T = np.zeros_like(x_train)
    y_T = np.zeros_like(y_train)
    
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[3]):
            x_T[i,:,:,j] = x_train[i,:,:,j].T
        y_T[i,:,:,0] = y_train[i,:,:,0].T
        
    x_train = np.concatenate((x_train, x_T), axis = 0)
    y_train = np.concatenate((y_train, y_T), axis = 0)
    
    return x_train, y_train
  
 
# Dataset을 Generation 형태로 변환
AUTO = tf.data.experimental.AUTOTUNE # Prefetch를 사용하여 속도 효율 

def trainGenerator():
    train_path = '/kaggle/input/dacon-project1/train'
    train_files = sorted(glob.glob(train_path + '/*'))
 
    for i,file in enumerate(train_files):
        
        dataset = np.load(file)        
        target= dataset[:,:,-1].reshape(40,40,1)         
        cutoff_labels = np.where(target < 0, 0, target)          
        feature = dataset[:,:,:10]
        
        if (cutoff_labels > 0).sum() < 50:                        
            continue
        
        feature,cutoff_labels = augmentation(feature,cutoff_labels)        
        # Tensor 형태로 가져오면 연산이 느리고 numpy가 빠르다
        for i in range(len(feature)):
            img = feature[i,:,:,:]
            label = cutoff_labels[i,:,:,:]
        
            yield (img,label)

train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([40,40,10]),tf.TensorShape([40,40,1])))
train_dataset = train_dataset.shuffle(30000)
train_dataset = train_dataset.batch(128).prefetch(AUTO)


# 이미지 Higher Resolution
def trainGenerator():
    
    train_path = '/kaggle/input/dacon-project1/train'
    train_files = sorted(glob.glob(train_path + '/*'))
        
    for i,file in enumerate(train_files):
        
        dataset = np.load(file)
        target= dataset[:,:,-1].reshape(40,40,1)      
        cutoff_labels = np.where(target < 0, 0, target)   

        im = []
        for i in range(0,9):
            img = Image.fromarray(dataset[:,:,i])
            img = img.resize((160,160))
            arr_img = np.array(img)
            im.append(list(arr_img))
        feature = np.array(im)
        feature = np.transpose(feature,(1,2,0))
        
        if (cutoff_labels > 0).sum() < 50:            
            
            continue

        yield (feature,cutoff_labels)
