# This script trains the conv1D-based DQPSK/Modulation
# by ZKY 2019/01/28

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
import tensorflow as tf

import matplotlib.pyplot as plt
from keras.optimizers import Adam

'''
 --- COMMUNICATION PARAMETERS ---
'''

# Bits per Symbol
k = 2

# Number of symbols
L = 20

# Channel Use
n = 1

# Effective Throughput
#  (bits per symbol)*( number of symbols) / channel use
R = k / n

# Eb/N0 used for training
train_Eb_dB = 14

# Noise Standard Deviation
noise_sigma = np.sqrt(1 / (2 * R * 10 ** (train_Eb_dB / 10)))


# Number of messages used for training, each size = k*L
batch_size = 32
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

epochs = 200

optimizer = Adam(lr=0.001)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=early_stopping_patience)


# Learning Rate Control
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, min_lr=0.0001)

# Save the best results based on Training Set
modelcheckpoint = ModelCheckpoint(filepath='./' + 'model_DLBC_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'Rayleigh' + '.h5',
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='auto', period=1)

'''
 --- CUSTOMIZED NETWORK LAYERS ---
'''

# Define Power Normalization Layer for Tx
def normalization(x):

    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2 * mean)  # 2 = I and Q channels


# Construct [c,-d] for Complex_mul Layer
tmp_array_1 = KR.ones(shape=(1, 1))
n_sign_array_1 = KR.concatenate([tmp_array_1, -tmp_array_1], axis=-1)

# Construct [d,c] for Complex_mul Layer
tmp_array_2 = KR.ones(shape=(1, 2))

# Define Complex Number Multiplication: s(t) = s(t-1)*x(t)
def Complex_mul(previous_output, current_input):

    # s_t-1 = a+bi
    # x_t = c+di

    # (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
    # construct x_t1 = [c,-d]

    x_t1 = current_input*n_sign_array_1

    # construct x_t2 = [d,c]
    x_t2 = current_input*tmp_array_2

    x_t2 = KR.reverse(x_t2, axes=-1)

    # ac - bd
    tmp1 = previous_output*x_t1
    tmp2 = KR.sum(tmp1,axis=-1)

    # ad + bc
    tmp3 = previous_output*x_t2
    tmp4 = KR.sum(tmp3,axis=-1)

    a_real = KR.expand_dims(tmp2, axis=1)
    a_img = KR.expand_dims(tmp4, axis=1)

    # (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
    a_complex_array = KR.concatenate([a_real, a_img], axis=-1)

    a_complex = KR.squeeze(a_complex_array,axis=0)

    return a_complex

# Define Complex Number Division: x(t)=s(t)/s(t-1)
def Complex_div(x):

    # Get the first of the tensor to the second to last (tmp1:a+bi)
    tmp1 = x[:-1, :]

    # Get the second of the tensor to the last          (tmp2:c+di)
    tmp2 = x[1:, :]

    # Define complex Number division of tmp2/tmp1 = (a+bi)/(c+di)
    # (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i

    # construct tmp3[-d,c]
    tmp3 = tmp1 * n_sign_array_1

    tmp3 = KR.reverse(tmp3, axes=-1)

    # ac + bd
    ac_bd = tmp1 * tmp2

    ac_bd = KR.sum(ac_bd, axis=-1)

    # bc-ad
    bc_ad = tmp2 * tmp3

    bc_ad = KR.sum(bc_ad, axis=-1)

    # c^2+d^2
    denorm = KR.sum(tmp1 ** 2, axis=-1)

    # ac + bd/c^2+d^2, bc-ad/c**2+d**2
    a_real = ac_bd / denorm
    a_img = bc_ad / denorm

    a_real = KR.expand_dims(a_real, axis=-1)

    a_img = KR.expand_dims(a_img, axis=-1)

    a_complex_array1 = KR.concatenate([a_real, a_img], axis=-1)

    a_complex_array2 = x[0, :]
    a_complex_array2 = KR.reshape(a_complex_array2, shape=(1, 2))

    a_complex_array = KR.concatenate([a_complex_array2, a_complex_array1], axis=0)

    return a_complex_array

# Define Differential Encoding Layer
output = []
out = []
def diff_encoder(x):
    global output,out

    # For each Batch
    for i in range(batch_size):
        for j in range(n):

            tmp = tf.slice(x, (i, 0, 2*j), (1, L, 2))
            tmp = tf.reshape(tmp, (L, 2))
            # Differential coding
            tx_single_channel_out = tf.scan(Complex_mul, tmp)

            tx_single_channel_out = tf.reshape(tx_single_channel_out, shape=(1, L, 2))

            if j == 0:
                out = tx_single_channel_out
            else:
                out = KR.concatenate([out,tx_single_channel_out])
        # Prepare for output
        if i == 0:
            output = out
        else:
            output = tf.concat([output, out], axis=0)

    return output

# Define Differential Decoding Layer
def diff_decoder(x):
    global output,out

    # For each Batch
    for i in range(batch_size):

        for k in range(0,2*n,2):

            # Differential coding
            rx_single_channel_out = Complex_div(x[i,:,k:k+2])

            rx_single_channel_out = KR.reshape(rx_single_channel_out,shape=(1,L,2))

            if k == 0:
                out = rx_single_channel_out
            else:
                out = KR.concatenate([out,rx_single_channel_out],axis=-1)
        # Prepare for output
        if i == 0:
            output = out
        else:
            output = tf.concat([output, out], axis=0)

    return output

# Define Channel Layers including AWGN
#  x: input data
#  sigma: noise std
def channel_layer(x, sigma):

    # AWGN noise
    w = KR.random_normal(KR.shape(x), mean=0.0, stddev=sigma)

    # Rayleigh noise h
    h_2D = KR.random_normal(shape=(batch_size, 2), mean=0, stddev=np.sqrt(1 / 2))
    h = KR.repeat(h_2D, L)

    # ---- For Complex Number division of h*x
    # (a+bi)*(c+di) = (ac-bd)+(bc+ad)i
    # construct h1[c,-d]
    tmp_array = KR.ones(shape=(batch_size, L, 1))
    n_sign_array = KR.concatenate([tmp_array, -tmp_array], axis=2)
    h1 = h * n_sign_array

    # construct h2
    h2 = KR.reverse(h,axes=2)

    # ac - bd
    tmp = h1 * x
    h1x = KR.sum(tmp, axis=-1)

    # bc + ad
    tmp = h2 * x
    h2x = KR.sum(tmp, axis=-1)

    a_real = KR.expand_dims(h1x, axis=2)
    a_img = KR.expand_dims(h2x, axis=2)

    a_complex_array = KR.concatenate([a_real, a_img], axis=-1)

    return a_complex_array + w



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

e = Lambda(diff_encoder,name='e_13')(e)

e = Lambda(normalization, name='power_norm')(e)

# AWGN channel
y_h = Lambda(channel_layer, arguments={'sigma': noise_sigma}, name='channel_layer')(e)

# Define Decoder Layers (Receiver)
d = Lambda(diff_decoder,name='d_0')(y_h)

d = Conv1D(filters=256, strides=1, kernel_size=1, name='d_1')(d)
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
                            validation_split=None, callbacks=[modelcheckpoint])

end = time.clock()

print('The NN has trained ' + str(end - start) + ' s')


# Plot the Training Loss and Validation Loss
hist_dict = mod_history.history

# val_loss = hist_dict['val_loss']
loss = hist_dict['loss']
acc = hist_dict['acc']
# val_acc = hist_dict['val_acc']
print(loss)
epoch = np.arange(1, epochs + 1)

# plt.semilogy(epoch,val_loss,label='val_loss')
plt.semilogy(epoch, loss, label='loss')

plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('Binary cross-entropy loss')

plt.show()



