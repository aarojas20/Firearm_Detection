import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import librosa

import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

#-------------------------------------------------
st.title('Firearm Alarm')
st.header('Listening for Firearms in Your Home')
st.text('For how long would you like Firearm Alarm to listen?')
t=st.slider('Select the time (hours).',min_value=1,max_value=24,step=1)

x = np.arange(0, 48000)/16000
fig, ax=plt.subplots()
ax.set_ylim(-1, 1)
line, = ax.plot(x, np.zeros(48000),color='m',linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Sound Wave')
the_plot = st.pyplot(plt)

# load the model from disk
model_path="models/"
cnn_model=load_model(model_path+'cnn_model.h5')

def wav2mfcc(wave, n_mfcc=20, max_len=11):
    '''wave is a np array'''
    #wave, sr = librosa.load(file_path, mono=True, sr=None)
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

def record(sr=16000, channels=1, duration=3, filename='pred_record.wav'):
    """
    Records live voice
    """
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels).reshape(-1)
    sd.wait()

    line.set_ydata(recording)
    the_plot.pyplot(plt)

    return recording


if st.button('Start listening with Firearm Alarm'):
    with st.spinner("Listening..."):
        for i in range(0,int(t*3600)):

            recording=record()

            ## run it through the model
            mfcc=wav2mfcc(recording)

            X_test = np.reshape(mfcc,(1, 20, 11, 1))
            Y_predict=cnn_model.predict(X_test)

            if Y_predict.round()[0][0]==0 :
                plt.text(0,.8,'All sounds safe.',fontsize=14,color='slateblue')
                #st.write("All sounds safe.")

            if Y_predict.round()[0][0]==1:
                plt.text(.5,.8,'This is a firearm!')
                #st.write("This is a firearm! Contacting local authorities...")

            plt.show()
else:
    st.write('Click the button to start listening.')
