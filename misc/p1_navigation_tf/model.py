import tensorflow as tf
from tensorflow.keras.layers import Dense

class QNetwork(tf.keras.Model):
    def __init__(self, action_size, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.fc1 = Dense(fc1_units, activation='relu')
        self.fc2 = Dense(fc2_units, activation='relu')
        self.fc3 = Dense(action_size)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)
