from __future__ import division
from keras import layers,regularizers,constraints,initializers
from keras.models import Model
from keras.layers import Lambda,Conv2D,LSTM, Concatenate,GlobalAveragePooling2D,Cropping2D
from keras.engine.topology import Layer

from keras.layers.core import Dropout, Activation,Dense,Reshape,Flatten
from keras.layers import Input, merge,Add,multiply,concatenate,InputSpec
from keras.layers.convolutional import Convolution2D, MaxPooling2D,UpSampling2D ,Deconvolution2D,ZeroPadding2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

import tensorflow as tf
import numpy as np

from SpatialPyramidPooling import SpatialPyramidPooling



############ loss functions
def emd_loss(logits, labels, reduce_mean=True, r=2):
    """Returns the earth mover distance between to arrays of ratings, 
    based on cumulative distribution function
    
    Args:
      y, y_hat: a mini-batch of ratings, each composed of a count of scores 
                shape = (None, n), array of count of scores for score from 1..n
      reduce_mean: apply tf.reduce_mean()
      r: r=2 for rmse loss (default) or r=1 for absolute val
    Returns:
      float 
    """
    def _cum_CDF(x): 
        # e.g. cdf([1,1,1,1]) ==  [ 0.25,  0.5 ,  0.75,  1.  ]  
        x = tf.to_float(x)
        cs = tf.cumsum(x, axis=1, reverse=False)
        total = cs[:,-1:]           # last column == cumulative sum
        cdf = tf.divide(cs, total)
        return cdf
    r = float(r)
    m,n = tf.convert_to_tensor(logits).get_shape().as_list()
    cdf_loss = tf.subtract(_cum_CDF(logits), _cum_CDF(labels))
    emd_loss = tf.pow( tf.reduce_mean( tf.pow(cdf_loss, r), axis=1 ), 1/r)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_ = tf.add_n([tf.reduce_mean(emd_loss)] + regularization_losses)
    #tf.summary.scalar('loss', loss_)
    if reduce_mean:
      return loss_
    else:    
      return  tf.reshape(emd_loss, [m,1])

def kld(y_true, y_pred):
#        """
#        Kullback-Leiber divergence
#        :param y_true: groundtruth.
#        :param y_pred: prediction.
#        :return: loss value (one symbolic value per batch element).
#        """
        P = y_true
#        print P.shape 
        P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
#        print P.shape 
        Q = y_pred
        Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

        kld = K.sum(P * K.log(K.epsilon() + P/(eps + Q)), axis=[1, 2, 3])
#        print kld.shape     
        return kld


def TVdist(y_true, y_pred):
        P = y_true
#        print P.shape 
        P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
#        print P.shape 
        Q = y_pred
        Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

        tv = K.sum( K.abs(P - Q) , axis=[1, 2, 3])
#        print kld.shape     
        return tv*0.5

def absTVdist(y_true, y_pred):
        P = y_true
#        print P.shape 
        P = P / (K.epsilon() + K.sum(K.abs(P), axis=[1, 2, 3], keepdims=True))
#        print P.shape 
        Q =  y_pred
        Q = Q / (K.epsilon() + K.sum(K.abs(Q), axis=[1, 2, 3], keepdims=True))

        tv = K.sum( K.abs(P - Q) , axis=[1, 2, 3])
#        print kld.shape     
        return tv*0.5

def SoftTVdist(y_true, y_pred):
        P = K.exp(y_true)
#        print P.shape 
        P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
#        print P.shape 
        Q = K.exp(y_pred)
        Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

        kld = K.sum( K.abs(P - Q) , axis=[1, 2, 3])
#        print kld.shape     
        return tv*0.5
##################################

class BatchNorm(BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when inferencing
        """
        return super(self.__class__, self).call(inputs, training=training)


def diconv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              dilation_rate=(1, 1),
              activation='relu',
              use_bias=True,
              train_bn=None,              
              bnn= True,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        use_bias=use_bias,    #usually true
        name=conv_name)(x)
    if bnn:
#        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = BatchNorm(axis=bn_axis, scale=False, name=bn_name)(x, training=train_bn)
    x = Activation(activation, name=name)(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block,use_bias=True,dilation_rate=(1, 1), train_bn=None):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), use_bias=use_bias, name=conv_name_base + '2a')(input_tensor)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,dilation_rate=dilation_rate,
               padding='same',use_bias=use_bias,  name=conv_name_base + '2b')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),use_bias=use_bias,  name=conv_name_base + '2c')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2c')(x, training=train_bn)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,use_bias=True, strides=(2, 2),dilation_rate=(1, 1), train_bn=None):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,use_bias=use_bias, 
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, dilation_rate=dilation_rate, padding='same',use_bias=use_bias, 
               name=conv_name_base + '2b')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=use_bias, name=conv_name_base + '2c')(x)
    #x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2c')(x, training=train_bn)
    shortcut = Conv2D(filters3, (1, 1), strides=strides,use_bias=use_bias, 
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNorm(axis=bn_axis, name=bn_name_base + '1')(shortcut, training=train_bn)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

#keras resnet without bias.
def M_ResNet50nob(input_tensor=None, train_bn=None):
    input_shape = (None, None,3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # conv_1
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, (7, 7), strides=(2, 2), use_bias=False, name='conv1')(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(x)
    # conv_2
    x = conv_block(x, 3, [64, 64, 256], stage=2, use_bias=False,block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, use_bias=False,block='b', train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2,use_bias=False, block='c', train_bn=train_bn)
    # conv_3
    x = conv_block(x, 3, [128, 128, 512], stage=3, use_bias=False,block='a', strides=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3,use_bias=False, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, use_bias=False,block='c', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3,use_bias=False, block='d', train_bn=train_bn)
    # conv_4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, use_bias=False,block='a',strides=(1, 1),dilation_rate=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,use_bias=False, block='b',dilation_rate=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, use_bias=False,block='c',dilation_rate=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, use_bias=False,block='d',dilation_rate=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,use_bias=False, block='e',dilation_rate=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,use_bias=False, block='f',dilation_rate=(2, 2), train_bn=train_bn)
    # conv_5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, use_bias=False,block='a',strides=(1, 1),dilation_rate=(4, 4), train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5,use_bias=False, block='b',dilation_rate=(4, 4), train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5,use_bias=False, block='c',dilation_rate=(4, 4), train_bn=train_bn)

    model = Model(img_input, x)

    # Load weights
    model.load_weights('models/resnet50.h5')
    # for layer in model.layers:
    #     layer.training = False 
    return model         


def DINetEncoder(img_rows=480, img_cols=640,train_bn=None, weight_file=None):
    inputimage = Input(shape=(img_rows, img_cols,3))
    base_model = M_ResNet50nob(input_tensor=inputimage, train_bn=train_bn)  # 
    
#    for layer in base_model.layers:
#        layer.trainable = False
    x5 = base_model.output

    x5 = Convolution2D(256, kernel_size=(1, 1), activation='relu',padding='same',dilation_rate=(1, 1),use_bias=False)(x5) 

    b1 = Convolution2D(256, kernel_size=(3, 3), activation='relu',padding='same',dilation_rate=(4, 4),use_bias=False)(x5) 

    b2 = Convolution2D(256, kernel_size=(3, 3), activation='relu',padding='same',dilation_rate=(8, 8),use_bias=False)(x5) 

    b3 = Convolution2D(256, kernel_size=(3, 3), activation='relu',padding='same',dilation_rate=(16, 16),use_bias=False)(x5) 

    x=layers.add([b1, b2,b3])  # 2618
    # x=layers.add([b1,b2]) 
    model = Model(inputs=[inputimage], outputs=[x,b1, b2,b3])
    
    # load weights from pretrained DINet
    # weight_file = 'models/fromAuthors/EncOfDINet.TVdist.EXP0-Salicon-lr=0.0001-bs=10.05-0.2638-0.2038.h5'
    # model.load_weights(weight_file)
    # print("\n\nload weights from "+weight_file+"\n\n")

    # Load weights from pretrained-from-AVA
    if weight_file:
        model.load_weights(weight_file)
        print("\n\nload weights from "+weight_file+"\n\n")

#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model 

#--------dec1--------#
### modify this to conv(7,7,32)-conv(7,7,8)-conv(7,7,1)
def DINetAesDecoder(img_rows=480, img_cols=640):  
    ### sal branch 
    inputimage = Input(shape=(int(img_rows/8), int(img_cols/8),256))
    x = Convolution2D(32, kernel_size=(7, 7), activation='relu',padding='same',use_bias=False)(inputimage) 
    x = Convolution2D(8, kernel_size=(7, 7), activation='relu',padding='same',use_bias=False)(x) 
    final_output = Convolution2D(1, kernel_size=(7, 7), activation='sigmoid',padding='same',use_bias=False)(x)
    final_output_up = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*8, int(x.shape[2])*8)))(final_output)  
    
    #### aes branch
    x_aes = SpatialPyramidPooling([1,2,4])(inputimage)
    x_aes = Flatten()(x)
    # x_aes = L2Normalization()(x_aes)
    x_aes = Lambda(lambda x: K.l2_normalize(x,axis=-1))(x_aes)
    x_aes = Dense(128, activation='relu')(x_aes)
    final_aes = Dense(10, activation='softmax')(x_aes)

    # model = Model(inputs=[inputimage], outputs=[final_output_up, final_aes])
    model = Model(inputs=[inputimage], outputs=[final_aes])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model     

# train_bn=False better performance.
def DINetAes(img_rows=480, img_cols=640, train_bn=False, weight_file=None):
    inputimage = Input(shape=(img_rows, img_cols,3))
    x,_,_,_  = DINetEncoder(img_rows, img_cols, train_bn, weight_file)(inputimage)
    # final_output_up, final_aes = DINetDecoder(img_rows, img_cols)(x)
    # model = Model(inputs=[inputimage], outputs=[final_output_up, final_aes])
    final_aes = DINetAesDecoder(img_rows, img_cols)(x)
    model = Model(inputs=[inputimage], outputs=[final_aes])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model

#--------dec2--------#
### modify this to conv(7,7,32)-conv(7,7,8)-conv(7,7,1)
def DINetSalDecoder(img_rows=480, img_cols=640):  
    ### sal branch 
    inputimage = Input(shape=(int(img_rows/8), int(img_cols/8),256))
    x = Convolution2D(32, kernel_size=(7, 7), activation='relu',padding='same',use_bias=False)(inputimage) 
    x = Convolution2D(8, kernel_size=(7, 7), activation='relu',padding='same',use_bias=False)(x) 
    final_output = Convolution2D(1, kernel_size=(7, 7), activation='sigmoid',padding='same',use_bias=False)(x)
    final_output_up = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*8, int(x.shape[2])*8)))(final_output)  

    model = Model(inputs=[inputimage], outputs=[final_output_up])
    # model = Model(inputs=[inputimage], outputs=[final_aes])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model 

# train_bn=False better performance.
def DINetSal(img_rows=480, img_cols=640, train_bn=False, weight_file=None):
    inputimage = Input(shape=(img_rows, img_cols,3))
    x,_,_,_  = DINetEncoder(img_rows, img_cols, train_bn, weight_file)(inputimage)
    final_output_up = DINetSalDecoder(img_rows, img_cols)(x)
    model = Model(inputs=[inputimage], outputs=[final_output_up])
   
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model

#--------dec1+2--------#
### modify this to conv(7,7,32)-conv(7,7,8)-conv(7,7,1)
def DINetDecoder(img_rows=480, img_cols=640): 
    inputimage1 = Input(shape=(int(img_rows/8), int(img_cols/8),256))
    inputimage2 = Input(shape=(int(img_rows/8), int(img_cols/8),256)) 
  
    #### aes branch
    x_aes = SpatialPyramidPooling([1,2,4])(inputimage1)
    x_aes = Flatten()(x_aes)
    # x_aes = L2Normalization()(x_aes)
    x_aes = Lambda(lambda x: K.l2_normalize(x,axis=-1))(x_aes)
    x_aes = Dense(128, activation='relu')(x_aes)
    final_aes = Dense(10, activation='softmax')(x_aes)

    #### sal branch 
    x = Convolution2D(32, kernel_size=(7, 7), activation='relu',padding='same',use_bias=False)(inputimage2) 
    x = Convolution2D(8, kernel_size=(7, 7), activation='relu',padding='same',use_bias=False)(x) 
    final_output = Convolution2D(1, kernel_size=(7, 7), activation='sigmoid',padding='same',use_bias=False)(x)
    final_output_up = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*8, int(x.shape[2])*8)))(final_output)  
  
    model = Model(inputs=[inputimage1,inputimage2], outputs=[final_aes,final_output_up])
    # model = Model(inputs=[inputimage], outputs=[final_aes])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)
    return model 

# train_bn=False better performance.
def DINetMTL(img_rows=480, img_cols=640, train_bn=False, weight_file=None):
    inputimage1 = Input(shape=(img_rows, img_cols,3))
    inputimage2 = Input(shape=(img_rows, img_cols,3))
    #aes
    x1,_,_,_  = DINetEncoder(img_rows, img_cols, train_bn, weight_file)(inputimage1)
    #sal
    x2,_,_,_  = DINetEncoder(img_rows, img_cols, train_bn, weight_file)(inputimage2)
    final_aes, final_output_up = DINetDecoder(img_rows, img_cols)([x1,x2])
    model = Model(inputs=[inputimage1,inputimage2], outputs=[final_aes,final_output_up])
   
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model

def DINetDecodernoas(img_rows=480, img_cols=640):   
    inputimage = Input(shape=(int(img_rows/8), int(img_cols/8),256))
    x = Convolution2D(256, kernel_size=(3, 3),padding='same',use_bias=False)(inputimage) 
    x = Convolution2D(256, kernel_size=(3, 3),padding='same',use_bias=False)(x) 

    final_output = Convolution2D(1, kernel_size=(3, 3),padding='same',use_bias=False)(x)   
    # avoid getting 'negative' saliency map.  It can be commented by using the absTVdist 
    final_output =  Lambda(lambda x:  (x- tf.reduce_min(x)))(final_output)  
    final_output =  Lambda(lambda x:  x/tf.reduce_max(x))(final_output)  

    final_output_up = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*8, int(x.shape[2])*8)))(final_output)  

#    
    model = Model(inputs=[inputimage], outputs=[final_output_up])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model 


def DINetvisual2o(img_rows=480, img_cols=640, train_bn=False):
    inputimage = Input(shape=(img_rows, img_cols,3))
    
#    for layer in base_model.layers:
#        layer.trainable = False
    x,b1,b2,b3 = DINetEncoder(img_rows, img_cols)(inputimage)
    # for layer in DINetEncoder.layers:
    #     layer.trainable = False
    final_output_up = DINetDecoder(img_rows, img_cols)(x)
    final_output_up2 = DINetDecodernoas(img_rows, img_cols)(x)  #for training  the whole model with two different decoders.
    # final_output_up2 = DINetDecodernoas(img_rows, img_cols)(b1)  #for visualizing b1
    model = Model(inputs=[inputimage], outputs=[final_output_up,final_output_up2])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model    