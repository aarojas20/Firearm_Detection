'''The following is adapted from Lukas Biewald from WandB--TY to Lukas for sharing his code online. 
I have made modifications to specify the location of the datasets, remove downsampling of the MFCC spectrograms, remove specification of the sampling rate, and indicate the duration rate'''

import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
#from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

DATA_PATH_TEST = "../data/test_bal/"
# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH_TEST):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    print('The number of labels is: ', len(labels))
    return labels, label_indices, tf.keras.utils.to_categorical(label_indices)


# convert file to wav2mfcc
# Mel-frequency cepstral coefficients
def wav2mfcc(file_path, n_mfcc=20, max_len=170):
    wave, sr = librosa.load(file_path, mono=True,duration=4) # removed: sr=None
    wave = np.asfortranarray(wave)                   # remove downsampling
    mfcc = librosa.feature.mfcc(wave, n_mfcc=n_mfcc) # removed: sr=16000,

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

# path='/Users/aarojas/Documents/Data_Science_Resources/Insight_20B/AAR_Insight_Project/Firearm_Detection/data/raw/ESC-50-master/audio/'
# birds='1-100038-A-14.wav'
# mfcc=wav2mfcc(path+birds)
# print(mfcc.shape)
# librosa.display.specshow(mfcc)
# plt.ylabel('MFCC')
# plt.colorbar()
# plt.savefig('first_notasfortran.png')
# plt.close()
#
# x,sr=librosa.load(path+birds,duration=5)
# mfcc=librosa.feature.mfcc(x)
# print(mfcc.shape)
# librosa.display.specshow(mfcc)
# plt.ylabel('MFCC')
# plt.colorbar()
# plt.savefig('second_notasfortran.png')

def save_data_to_array(path=DATA_PATH_TEST, max_len=170, n_mfcc=20):  #adjust max_len=170
    '''saves to a numpy array file (.npy)
    it combines the mfcc vectors, the labels'''
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len, n_mfcc=n_mfcc)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


def get_test(split_ratio=.9, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH_TEST)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)



def prepare_dataset(path=DATA_PATH_TEST):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, duration=4) # remove: sr=None, add: duration =4
            # Downsampling
#             wave = wave[::3]                           # no downsampling
            mfcc = librosa.feature.mfcc(wave, n_mfcc=20) # removed sr=16000, add n_mfcc=20
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=DATA_PATH_TEST):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]
