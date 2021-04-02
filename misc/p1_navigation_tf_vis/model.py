import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization

class QNetwork(tf.keras.Model):
    def __init__(self, action_size, c1_filters=32, c2_filters=64):
        super(QNetwork, self).__init__()
        self.c1 = Conv2D(filters=c1_filters, kernel_size=8, strides=4, activation='relu')
        self.bn = BatchNormalization()
        self.c2 = Conv2D(filters=c2_filters, kernel_size=4, strides=2, activation='relu')
        self.c3 = Conv2D(filters=c2_filters, kernel_size=3, strides=1, activation='relu')
        self.f = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(action_size)

    def call(self, state):
        # def scale_lumininance(img):
        #     return np.dot(img[..., :3], [0.299, 0.587, 0.114])
        #
        # def normalize(img):
        #     return img / 255
        #
        # x = scale_lumininance(state)
        # x = normalize(x)
        # x = tf.expand_dims(x, axis=-1)

        x = self.c1(state)
        #x = self.bn(x)
        x = self.c2(x)
        #x = self.bn(x)
        x = self.c3(x)  # 7,7,64
        # x = self.bn(x)
        x = self.f(x) #3136
        x = self.fc1(x) #512
        return self.fc2(x) #4

