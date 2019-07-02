# This script Calculate BLER of the conv1D-based Linear Block Codes/Modulation
# by ZKY 2019/02/15

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Lambda, BatchNormalization, Input, Conv1D, TimeDistributed, LeakyReLU, Flatten, Activation
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as KR
import copy


'''
 --- COMMUNICATION PARAMETERS ---
'''

# number of information bits
k = 4

# codeword Length
L = 10

# Channel use
n = 1

# Effective Throughput
#  (bits per symbol)*( number of symbols) / channel use
R = k/n

# Eb/N0 used for training(load_weights)
train_Eb_dB = 7

# Number of messages used for test, each size = k*L
batch_size = 64
num_of_sym = batch_size*1000

# Initial Vectors
Vec_Eb_N0 = []
Bit_error_rate = []

'''
 --- GENERATING INPUT DATA ---
'''

# Initialize information Data 0/1
in_sym = np.random.randint(low=0, high=2, size=(num_of_sym, k * L))
label_data = copy.copy(in_sym)
in_sym = np.reshape(in_sym,newshape=(num_of_sym,L,k))

# Convert Binary Data to integer
tmp_array = np.zeros(shape=k)
for i in range(k):
    tmp_array[i]=2**i
int_data = tmp_array[::-1]

# Convert Integer Data to one-hot vector
int_data = np.reshape(int_data,newshape=(k,1))
one_hot_data = np.dot(in_sym,int_data)
# print(one_hot_data)
vec_one_hot = to_categorical(y=one_hot_data, num_classes=2**k)

# used as Label data
label_one_hot = copy.copy(vec_one_hot)


# Define Channel Layers including AWGN and Flat Rayleigh fading
def channel_layer(x,sigma):
    w = KR.random_normal(KR.shape(x), mean=0.0, stddev=sigma)

    return x + w


def normalization(x):
    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2*mean)  # 2 = number of NN into the channel


print('start simulation ...' + str(k) + '_' + str(L)+'_'+str(n))



'''
 --- DEFINE THE Neural Network(NN) ---
'''

# Eb_N0 in dB
for Eb_N0_dB in range(0,21):

    # Noise Sigma at this Eb
    noise_sigma = np.sqrt(1 / (2 * R * 10 ** (Eb_N0_dB / 10)))

    # Define Encoder Layers (Transmitter)
    model_input = Input(batch_shape=(None, L, 2 ** k), name='input_bits')

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

    # Build the model
    model = Model(inputs=model_input, outputs=model_output)

    encoder = Model(model_input,e)
    # Load Weights from the trained NN
    model.load_weights('./' + 'model_LBC_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5',
                       by_name=False)

    '''
    RUN THE NN
    '''

    # RUN Through the Model and get output
    decoder_output = model.predict(vec_one_hot, batch_size=batch_size)
    encoder_output = encoder.predict(vec_one_hot, batch_size=batch_size)

    '''
     --- CALULATE BLER ---

    '''

    # Decode One-Hot vector
    position = np.argmax(decoder_output, axis=2)
    tmp = np.reshape(position,newshape=one_hot_data.shape)

    error_rate = np.mean(np.not_equal(one_hot_data,tmp))


    print('Eb/N0 = ', Eb_N0_dB)
    print('BLock Error Rate = ', error_rate)

    print('\n')

    # Store The Results
    Vec_Eb_N0.append(Eb_N0_dB)
    Bit_error_rate.append(error_rate)



'''
PLOTTING
'''
# Print BER
# print(Bit_error_rate)

print(Vec_Eb_N0, '\n', Bit_error_rate)

with open('BLER_model_LBC_'+str(k)+'_'+str(n)+'_'+str(L)+'_AWGN'+'.txt', 'w') as f:
    print(Vec_Eb_N0, '\n', Bit_error_rate, file=f)
f.closed

# Plot BER Figure
plt.semilogy(Vec_Eb_N0, Bit_error_rate, color='red')
label = [str(k) + '_' + str(L)]
plt.legend(label, loc=0)
plt.xlabel('Eb/N0')
plt.ylabel('BER')
plt.title(str(k) + '_' + str(n)+'_'+str(L))
plt.grid('true')
plt.show()


