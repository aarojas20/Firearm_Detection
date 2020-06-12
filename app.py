import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import librosa
#import pickle

# load the model from disk
model_path="models/"
cnn_model=load_model(model_path+'cnn_model.h5')

#cnn_model_pkl = pickle.load(open(model_path+'cnn_model.pickle', 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)
def wav2mfcc(file_path, n_mfcc=20, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = np.asfortranarray(wave[::3])
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=n_mfcc)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

path="data/external/"
audio_clip1='5-195710-A-10.wav' # ?
audio_clip2='2-121978-A-29.wav' #?
audio_clip3='T_17P.wav'
audio_dict={
'Audio clip 1':audio_clip1,
'Audio clip 2': audio_clip2,
'Audio clip 3': audio_clip3}

st.title('Firearm Alarm')
st.header('Listening for Firearms in Your Home')

st.text('The following are a set of sample audio clips that can be input into the model.')


st.audio(path+audio_clip1)


st.text('This is audio clip 1.')

st.audio(path+audio_clip2)

st.text('This is audio clip 2.')

st.audio(path+audio_clip3)

option = st.selectbox('Select the clip you would like the model to analyze.',('Audio clip 1', 'Audio clip 2', 'Audio clip 3'))
st.write('You selected:', option)

if st.button('Analyze '+option):

    mfcc=wav2mfcc(path+audio_dict[option])
    X_test = np.reshape(mfcc,(1, 20, 11, 1))
    # print(mfcc.shape)
    # print(X_test.shape)
    Y_predict=cnn_model.predict(X_test)
    # print('Y_predict=',Y_predict.round()[0])
    # print(Y_predict[0].shape)
    # print(Y_predict[0][0])

    if Y_predict.round()[0][0]==0 :

        st.write("This doesn't sound like a firearm.")

        # print('Not a firearm')
    if Y_predict.round()[0][0]==1:

        st.write("This is a firearm! Contacting local authorities...")

        # print('A firearm')
else:
    st.write('Click the button to analyze the audio clip.')
