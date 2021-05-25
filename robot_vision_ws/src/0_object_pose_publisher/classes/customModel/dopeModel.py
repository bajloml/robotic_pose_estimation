import sys
import os

import tensorflow as tf

class dopeModel(tf.keras.Model):
    def __init__(self,            
                pretrained=False,
                freezeLayers=10,
                numBeliefMap=9,
                numAffinity=16,
                stop_at_stage=6,  # number of stages to process (if less than total number of stages)):
                inp_shape=(400,400,3)):
        # Initialize the necessary components of tf.keras.Model
        super(dopeModel, self).__init__()
        # Now we initalize the needed layers - order does not matter.
        # -----------------------------------------------------------
        # load vgg19 network
        #vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_tensor=None,
        #                                    input_shape=None, pooling=None, classes=1000,
        #                                    classifier_activation='softmax'
        #                                    )
        self.numAffinity = numAffinity
        self.numBeliefMap = numBeliefMap
        self.stop_at_stage = stop_at_stage
        self.inp_shape = inp_shape
        self.freezeLayers = freezeLayers
        self.pretrained = pretrained
        
        if self.pretrained is False:
            print("Training network without imagenet weights.")
            vgg19 = tf.keras.applications.VGG19(include_top=False, weights=None, input_shape=self.inp_shape)  # input_shape=(400, 400, 3) ??
        else:
            print("Training network pretrained on imagenet.")
            vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=self.inp_shape)  # input_shape=(400, 400, 3) ??
      
        self.inp = vgg19.layers[0]
        self.out_1 = vgg19.layers[1]
        self.out_2 = vgg19.layers[2]
        self.out_3 = vgg19.layers[3]
        self.out_4 = vgg19.layers[4]
        self.out_5 = vgg19.layers[5]
        self.out_6 = vgg19.layers[6]
        self.out_7 = vgg19.layers[7]
        self.out_8 = vgg19.layers[8]
        self.out_9 = vgg19.layers[9]
        self.out_10 = vgg19.layers[10]

        del vgg19

        # freeze layers
        for layer in range(self.freezeLayers):
            self.layers[layer].trainable = False
        
        #self.out_11 = tf.keras.layers.BatchNormalization(name = 'add_BN_after_vgg')
        self.out_12 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name = 'add_Conv2D_512_1')
        self.out_13 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name = 'add_Conv2D_512_2')
        self.out_14 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name = 'add_Conv2D_512_3')
        self.out_15 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name = 'add_Conv2D_512_4')
        self.out_16 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name = 'add_Conv2D_512_5')
        self.out_17 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name = 'add_Conv2D_512_6')
        #self.out_18 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name = 'add_MaxPooling2D_512_3')
        #self.out_19 = tf.keras.layers.BatchNormalization(name = 'add_BN_512_3')
        self.out_20 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name = 'add_Conv2D_256_1')
        #self.out_21 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name = 'add_Conv2D_256_2')
        self.out_22 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name = 'add_Conv2D_128_1')
        #self.out_23 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name = 'add_Conv2D_128_2')
        self.out_24 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name = 'add_MaxPooling2D_128_1')
        self.out_25 = tf.keras.layers.BatchNormalization(name = 'add_BN_128_1')

        #self.dropout = tf.keras.layers.Dropout(0.5)

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        
        self.m1_2 = dopeModel.create_stage(128, numBeliefMap, True, name='m_1_2_')
        self.m2_2 = dopeModel.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False, name='m_2_2_')
        self.m3_2 = dopeModel.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False, name='m_3_2_')
        self.m4_2 = dopeModel.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False, name='m_4_2_')
        self.m5_2 = dopeModel.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False, name='m_5_2_')
        self.m6_2 = dopeModel.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False, name='m_6_2_')
        
        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages

        self.m1_1 = dopeModel.create_stage(128, numAffinity, True, name='m_1_1_')
        self.m2_1 = dopeModel.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False, name='m_2_1_')
        self.m3_1 = dopeModel.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False, name='m_3_1_')
        self.m4_1 = dopeModel.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False, name='m_4_1_')
        self.m5_1 = dopeModel.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False, name='m_5_1_')
        self.m6_1 = dopeModel.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False, name='m_6_1_')
    # Forward pass of model
    def call(self, inputs):
        '''Runs inference on the neural network'''
        x = self.inp(tf.keras.applications.vgg19.preprocess_input(inputs, data_format="channels_last"))
        x = self.out_1(x)
        x = self.out_2(x)
        x = self.out_3(x)
        x = self.out_4(x)
        x = self.out_5(x)
        x = self.out_6(x)
        x = self.out_7(x)
        x = self.out_8(x)
        x = self.out_9(x)
        x = self.out_10(x) 

        #x = self.out_11(x)
        x = self.out_12(x)
        x = self.out_13(x)
        x = self.out_14(x)
        x = self.out_15(x)
        x = self.out_16(x)
        x = self.out_17(x)
        #x = self.out_18(x)
        #x = self.out_19(x)
        x = self.out_20(x)
        #x = self.out_21(x)
        x = self.out_22(x)
        #x = self.out_23(x)
        x = self.out_24(x)
        x = self.out_25(x)

        #x = self.dropout(x)

        out1 = x

        # 1 stage
        #beliefs
        out1_2 = self.m1_2(out1)
        #affinities
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage ==1:
          return out1_2, out1_1

        # 2 stage
        out2 = tf.concat([out1_2, out1_1, out1], axis=3) #torch.cat([out1_2, out1_1, out1], 1)
        #beliefs
        out2_2 = self.m2_2(out2)
        #affinities
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage ==2:
          return out2_2, out2_1

        # 3 stage
        out3 = tf.concat([out2_2, out2_1, out1], axis=3) #torch.cat([out1_2, out1_1, out1], 1)
        #beliefs
        out3_2 = self.m3_2(out3)
        #affinities
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage ==3:
          return out3_2, out3_1

        # 4 stage
        out4 = tf.concat([out3_2, out3_1, out1], axis=3) #torch.cat([out1_2, out1_1, out1], 1)
        #beliefs
        out4_2 = self.m4_2(out4)
        #affinities
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage ==4:
          return out4_2, out4_1

        # 5 stage
        out5 = tf.concat([out4_2, out4_1, out1], axis=3) #torch.cat([out1_2, out1_1, out1], 1)
        #beliefs
        out5_2 = self.m5_2(out5)
        #affinities
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage ==5:
          return out5_2, out5_1

        # 6 stage
        out6 = tf.concat([out5_2, out5_1, out1], axis=3) #torch.cat([out1_2, out1_1, out1], 1)
        #beliefs
        out6_2 = self.m6_2(out6)
        #affinities
        out6_1 = self.m6_1(out6)

        if self.stop_at_stage ==6:
          return out6_2, out6_1

    def modelForPlot(self):
        x = tf.keras.layers.Input(shape=self.inp_shape)
        model = tf.keras.Model(inputs=[x], outputs=[self.call(x)])
        return model

    @staticmethod
    def create_stage(in_channels, out_channels, first=False, name=''):
        '''Create the neural network layers for a single stage.'''
        #identity initializer
        initializer = tf.keras.initializers.GlorotNormal()

        model = tf.keras.models.Sequential()
        mid_channels = 128
        if first:
            padding = 'same'
            kernel = 3
            count = 6
            final_channels = 512 #512
        else:
            padding = 'same'
            kernel = 7 #7
            count = 10 #10
            final_channels = mid_channels #mid_channels

        # First convolution
        model.add(tf.keras.layers.Conv2D(
                                        input_shape=(50,50,in_channels),
                                        data_format="channels_last",
                                        filters=in_channels,
                                        kernel_size=(kernel, kernel),
                                        strides=(1, 1),
                                        padding=padding,
                                        activation='relu',
                                        name='Conv2D'+ name+'1',
                                        kernel_initializer = initializer)
                                        )

        # Middle convolutions
        for i in range(1, count-1):
            model.add(tf.keras.layers.Conv2D(
                                            filters=mid_channels,
                                            kernel_size=(kernel, kernel),
                                            strides=(1, 1),
                                            padding=padding,
                                            activation='relu',
                                            name='Conv2D'+ name+str(i+2),
                                            kernel_initializer = initializer)
                                            )

        # Penultimate convolution
        model.add(tf.keras.layers.Conv2D(filters=final_channels, kernel_size=(kernel, kernel), padding = padding,  strides=(1, 1), activation='relu', name='Conv2D'+ name+'penultimate',
                                        kernel_initializer = initializer))
        # Last convolution
        model.add(tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(kernel, kernel), padding = padding, strides=(1, 1), activation='relu', name='Conv2D'+ name+'final',
                                        kernel_initializer = initializer))
        model.trainable=True
        #model.summary()

        return model