import tensorflow as tf
from keras import models,Sequential,layers
from keras.models import Model, Input, load_model
from keras.layers import Conv2D, SeparableConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, DepthwiseConv2D
from keras.layers import Activation, BatchNormalization, Dropout, Flatten, Reshape, Dense, Softmax, multiply, Add, Input, ReLU
from keras.layers import GlobalMaxPooling2D, Permute, Concatenate, Lambda
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.optimizers import SGD,Adam
from keras.regularizers import l2
from sklearn.metrics import log_loss
from keras import backend as K
from keras.activations import sigmoid


# BatchNormalization, Relu를 고정하여 사용하기 위해 Conv, Sepconv를 함수로 지정
def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu', weight_decay=1e-5):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)

    return x

def sepconv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu', weight_decay=1e-5,
                 depth_multiplier=1):
    x = SeparableConv2D(filters, kernel_size, padding=padding, strides=strides, depth_multiplier=depth_multiplier,
                        depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)

    if activation:
        x = Activation(activation)(x)

    return x

# Xception의 구조를 이용하여, Filter, Pooling, Transepose를 변경 및 추가
def Xception(model_input):
    ## Entry flow
    x = conv2d_bn(model_input, 16, (3, 3)) 
#     x = conv2d_bn(x, 16, (3, 3), strides=2) 
    x = conv2d_bn(x, 32, (3, 3))

    for fliters in [64, 128, 256]: 
        if fliters in [64,128]:
            residual = conv2d_bn(x, fliters, (1, 1), strides=2, activation=None)
        else:
            residual = conv2d_bn(x, fliters, (1, 1), activation=None)
        x = Activation(activation='relu')(x)
        x = sepconv2d_bn(x, fliters, (3, 3))
        x = sepconv2d_bn(x, fliters, (3, 3), activation=None)
        if fliters == 64:
            x1 = sepconv2d_bn(x, fliters, (3, 3), activation=None)
            x = sepconv2d_bn(x, fliters, (3, 3), activation=None)
            x = MaxPooling2D((3, 3), padding='same', strides=2)(x)
            
        elif fliters == 128:
            x2 = sepconv2d_bn(x, fliters, (3, 3), activation=None)
            x = sepconv2d_bn(x, fliters, (3, 3), activation=None)
            x = MaxPooling2D((3, 3), padding='same', strides=2)(x)
            
        else:
            x3 = sepconv2d_bn(x, fliters, (3, 3), activation=None)
            x = sepconv2d_bn(x, fliters, (3, 3), activation=None)            
        x = Add()([x, residual])

    ## Middle flow
    for i in range(8):  # (19, 19, 728)
        residual = x

        x = sepconv2d_bn(x, 256, (3, 3))
        x = sepconv2d_bn(x, 256, (3, 3))
        x = sepconv2d_bn(x, 256, (3, 3), activation=None)

        x = Add()([x, residual])

    ## Exit flow
    residual = conv2d_bn(x, 384, (1, 1), activation=None)  # (19, 19, 728) -> (10, 10, 1024)

    x = Activation(activation='relu')(x)
    x = sepconv2d_bn(x, 256, (3, 3))
    x = sepconv2d_bn(x, 384, (3, 3), activation=None)  # (19, 19, 728) -> (19, 19, 1024)
    x4 = sepconv2d_bn(x, 384, (3, 3), activation=None)  # (19, 19, 1024) -> (10, 10, 1024)

    x = Add()([x, residual])

    x = sepconv2d_bn(x, 512, (3, 3))
    x = sepconv2d_bn(x, 640, (3, 3))
    
    deconv3 = Conv2DTranspose(384, (3, 3), strides=(2, 2), padding="same")(x)
    x = concatenate([deconv3, x2])
    x = Dropout(0.25)(x)
    x = Conv2D(384, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    
    deconv2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = concatenate([deconv2, x1])
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    
    output_layer = Conv2D(1, (1,1), padding="same", activation='relu')(x)
    
    model = Model(model_input, output_layer)
    return model

input_shape = (40, 40, 10)

model_input = Input(shape=input_shape)

model = Xception(model_input)
