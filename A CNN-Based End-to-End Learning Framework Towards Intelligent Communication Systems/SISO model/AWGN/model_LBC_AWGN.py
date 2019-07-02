# This script trains the conv1D-based Linear Block Codes/Modulation
# by ZKY 2019/02/15

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Lambda, BatchNormalization, Input, Conv1D, TimeDistributed, Flatten, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as KR
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from keras.optimizers import Adam

'''
 --- COMMUNICATION PARAMETERS ---
'''

# Bits per Symbol
k = 4

# Number of symbols
L = 10

# Channel Use
n = 1

# Effective Throughput
#  bits per symbol / channel use
R = k / n

# Eb/N0 used for training
train_Eb_dB = 9

# Noise Standard Deviation
noise_sigma = np.sqrt(1 / (2 * R * 10 ** (train_Eb_dB / 10)))


# Number of messages used for training, each size = k*L
batch_size = 64
nb_train_word = batch_size*200

'''
 --- GENERATING INPUT DATA ---
'''

# Generate training binary Data
train_data = np.random.randint(low=0, high=2, size=(nb_train_word, k * L))
# Used as labeled data
label_data = copy.copy(train_data)
train_data = np.reshape(train_data, newshape=(nb_train_word, L, k))

# Convert Binary Data to integer
tmp_array = np.zeros(shape=k)
for i in range(k):
    tmp_array[i] = 2 ** i
int_data = tmp_array[::-1]

# Convert Integer Data to one-hot vector
int_data = np.reshape(int_data, newshape=(k, 1))
one_hot_data = np.dot(train_data, int_data)
vec_one_hot = to_categorical(y=one_hot_data, num_classes=2 ** k)

# used as Label data
label_one_hot = copy.copy(vec_one_hot)

'''
 --- NEURAL NETWORKS PARAMETERS ---
'''

early_stopping_patience = 100

epochs = 150

optimizer = Adam(lr=0.001)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=early_stopping_patience)


# Learning Rate Control
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, min_lr=0.0001)

# Save the best results based on Training Set
modelcheckpoint = ModelCheckpoint(filepath='./' + 'model_LBC_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5',
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='auto', period=1)


# Define Power Norm for Tx
def normalization(x):
    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2 * mean)  # 2 = I and Q channels


# Define Channel Layers including AWGN and Flat Rayleigh fading
#  x: input data
#  sigma: noise std
def channel_layer(x, sigma):

    w = KR.random_normal(KR.shape(x), mean=0.0, stddev=sigma)

    return x + w


model_input = Input(batch_shape=(batch_size, L, 2 ** k), name='input_bits')

e = Conv1D(filters=256, strides=1, kernel_size=1, name='e_1')(model_input)
e = BatchNormalization(name='e_2')(e)
e = Activation('elu', name='e_3')(e)

e = Conv1D(filters=256, strides=1, kernel_size=1, name='e_7')(e)
e = BatchNormalization(name='e_8')(e)
e = Activation('elu', name='e_9')(e)

e = Conv1D(filters=2 * n, strides=1, kernel_size=1, name='e_10')(e)  # 2 = I and Q channels
e = BatchNormalization(name='e_11')(e)
e = Activation('linear', name='e_12')(e)

e = Lambda(normalization, name='power_norm')(e)

# AWGN channel
y_h = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(e)

# Define Decoder Layers (Receiver)
d = Conv1D(filters=256, strides=1, kernel_size=1, name='d_1')(y_h)
d = BatchNormalization(name='d_2')(d)
d = Activation('elu', name='d_3')(d)

d = Conv1D(filters=256, strides=1, kernel_size=1, name='d_7')(d)
d = BatchNormalization(name='d_8')(d)
d = Activation('elu', name='d_9')(d)

# Output One hot vector and use Softmax to soft decoding
model_output = Conv1D(filters=2 ** k, strides=1, kernel_size=1, name='d_10', activation='softmax')(d)

# Build System Model
sys_model = Model(model_input, model_output)
encoder = Model(model_input, e)

# Print Model Architecture
sys_model.summary()

# Compile Model
sys_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# print('encoder output:', '\n', encoder.predict(vec_one_hot, batch_size=batch_size))

print('starting train the NN...')
start = time.clock()

# TRAINING
mod_history = sys_model.fit(vec_one_hot, label_one_hot,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.3, callbacks=[modelcheckpoint,reduce_lr])

end = time.clock()

print('The NN has trained ' + str(end - start) + ' s')


# Plot the Training Loss and Validation Loss
hist_dict = mod_history.history

val_loss = hist_dict['val_loss']
loss = hist_dict['loss']
acc = hist_dict['acc']
# val_acc = hist_dict['val_acc']
print('loss:',loss)
print('val_loss:',val_loss)

epoch = np.arange(1, epochs + 1)

plt.semilogy(epoch,val_loss,label='val_loss')
plt.semilogy(epoch, loss, label='loss')

plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('Binary cross-entropy loss')

plt.show()



