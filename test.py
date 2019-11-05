import numpy as np
import os
from os.path import isfile
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, Conv2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from keras import regularizers
import librosa
import librosa.display
import matplotlib.pyplot as plt
import random
import numpy as np
np.random.seed(1001)
import os
import shutil
import IPython
import seaborn as sns
from scipy.io import wavfile
from tqdm import tqdm_notebook 
import IPython.display as ipd
import librosa
import numpy as np
import scipy
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras.utils import Sequence, to_categorical
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation)
from keras.utils import Sequence, to_categorical
from keras import backend as K


test_folder = raw_input('Enter Test Folder location: ')
print(test_folder)


class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=5,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)

emotions = {'happy':0, 'sad':1, 'disgust':2, 'neutral':3, 
               'fear':4}

reverse_emotions = {v: k for k, v in emotions.items()}


file_name = []
def prepare_test_data(config, data_dir,size):
    X = np.empty(shape=(size, config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    index = 0
    for file in os.listdir(data_dir):
            file_path = data_dir+'/' + file
            data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")
            file_name.append(file)
            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
            data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
            data = np.expand_dims(data, axis=-1)
            X[index,] = data
           
    return X    


size = len(os.listdir(test_folder))

config = Config(sampling_rate=16000, audio_duration=1, n_folds=10, 
                learning_rate=0.0001, use_mfcc=True, n_mfcc=50)

Xtest  = prepare_test_data(config,test_folder,size)

mean = np.mean(Xtest, axis=0)
std = np.std(Xtest, axis=0)
Xtest = (Xtest - mean)/std

from keras.models import load_model
model  = load_model('../model.h5')

predictions = model.predict(Xtest)

index = 0
for file in file_name:
    ypred = np.argmax(predictions[index])
    index = index+1
    label = reverse_emotions[ypred]
    with open('predictions.txt', 'a') as pred_file:
      pred_file.write(file+" , " + label+'\n')
      
       
      
      


