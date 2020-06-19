import streamlit as st
#import argparse
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import time

#import sounddevice as sd
#import soundfile as sf
import matplotlib.pyplot as plt
##
# def int_or_str(text):
#     """Helper function for argument parsing."""
#     try:
#         return int(text)
#     except ValueError:
#         return text
#
#
# parser = argparse.ArgumentParser(add_help=False)
# parser.add_argument(
#     '-l', '--list-devices', action='store_true',
#     help='show list of audio devices and exit')
# args, remaining = parser.parse_known_args()
# if args.list_devices:
#     print(sd.query_devices())
#     parser.exit(0)
# parser = argparse.ArgumentParser(
#     description=__doc__,
#     formatter_class=argparse.RawDescriptionHelpFormatter,
#     parents=[parser])
# parser.add_argument(
#     'filename', nargs='?', metavar='FILENAME',
#     help='audio file to store recording to')
# parser.add_argument(
#     '-d', '--device', type=int_or_str,
#     help='input device (numeric ID or substring)')
# parser.add_argument(
#     '-r', '--samplerate', type=int, help='sampling rate')
# parser.add_argument(
#     '-c', '--channels', type=int, default=1, help='number of input channels')
# parser.add_argument(
#     '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
# args = parser.parse_args(remaining)
#
#
# if args.samplerate is None:
#     device_info = sd.query_devices(args.device, 'input')
#     # soundfile expects an int, sounddevice provides a float:
#     args.samplerate = int(device_info['default_samplerate'])

def wav2mfcc(wave, sr=22050,n_mfcc=20, max_len=170):
    '''wave is a np array'''
    wave = np.asfortranarray(wave)
    mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=n_mfcc)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


# load the model from disk
model_path="models/"
cnn_model=load_model(model_path+'cnn_model_accuracy_92.0_alpha_0.0002.h5')

#-------------------------------------------------

st.title('Firearm Alarm')
st.header('Listening for Firearms in Your Home')

##-----------------------------------------------------------------------------
path="data/external/"
audio_clip1='5-195710-A-10.wav' # ?
audio_clip2='2-121978-A-29.wav' #?
audio_clip3='T_17P.wav'
audio_dict={
'Audio clip 1':audio_clip1,
'Audio clip 2': audio_clip2,
'Audio clip 3': audio_clip3}

st.text('The following are a set of sample audio clips that can be input into the model.')


st.audio(path+audio_clip1)


st.text('This is audio clip 1.')

st.audio(path+audio_clip2)

st.text('This is audio clip 2.')

st.audio(path+audio_clip3)

option = st.selectbox('Select the clip you would like the model to analyze.',('Audio clip 1', 'Audio clip 2', 'Audio clip 3'))
st.write('You selected:', option)

if st.button('Analyze '+option):
    wave, sr = librosa.load(path+audio_dict[option], mono=True, sr=22050)
    mfcc=wav2mfcc(wave,sr=sr)
    X_test = np.reshape(mfcc,(1, 20, 170, 1))

    Y_predict=cnn_model.predict(X_test)
    print(Y_predict)

    if Y_predict.round()[0][0]==0 :

        st.write("This doesn't sound like a firearm.")

    if Y_predict.round()[0][0]==1:

        st.write("This is a firearm! Contacting local authorities...")

else:
    st.write('Click the button to analyze the audio clip.')

###############################################----------------------------------

st.header('Firearm Alarm in Action')
st.text('For how long would you like Firearm Alarm to listen?')
t=st.slider('Select the time (hours).',min_value=1,max_value=24,step=1)

x = np.arange(0, 4,1/22050)
fig, ax=plt.subplots()
ax.set_ylim(-1, 1)
line, = ax.plot(x, np.zeros(len(x)),color='m',linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Sound Wave')
the_plot = st.pyplot(plt)


def updateplot(wave):
    """
    update the plot with the wave file
    """

    line.set_ydata(wave)
    the_plot.pyplot(plt)


sample='data/external/Cooking_clip.mp3'
if st.button('See an example with Firearm Alarm'):
    with st.spinner("Listening..."):
        array,sr=librosa.load(sample)
        tiempo=librosa.get_duration(array) #time in seconds
        for t in range(0,int(tiempo),4):
            wave, sr = librosa.load(sample, mono=True,offset=t,duration=4)

            ## run it through the model
            mfcc=wav2mfcc(wave)

            X_test = np.reshape(mfcc,(1, 20, 170, 1))
            Y_predict=cnn_model.predict(X_test)

            if Y_predict.round()[0][0]==0 :
                plt.text(0,.8,'All sounds safe.',fontsize=14,color='slateblue')
                #st.write("All sounds safe.")

            if Y_predict.round()[0][0]==1:
                plt.text(.5,.8,'This is a firearm!')
                #st.write("This is a firearm! Contacting local authorities...")

            updateplot(wave)
            time.sleep(3)

            plt.show()
else:
    st.write('Click the button to start listening.')

############################################-----------------------------------






















# st.header('Try Firearm Alarm with your own sound clips')
#
# uploaded_file = st.file_uploader("Upload a .WAV file", type="wav")
# if uploaded_file is not None:
#     uploaded_file=uploaded_file.read().decode("utf-8")
#     print(type(uploaded_file))
#     array,sr=librosa.load(uploaded_file)
#     tiempo=librosa.get_duration(array) #time in seconds
#
# x = np.arange(0, 4,1/22050)
# fig, ax=plt.subplots()
# ax.set_ylim(-1, 1)
# line, = ax.plot(x, np.zeros(len(x)),color='royalblue',linewidth=2)
# plt.xlabel('Time (s)')
# plt.ylabel('Sound Wave')
# the_plot = st.pyplot(plt)
#
# if uploaded_file is not None:
#     if st.button('Analyze your sound clip'):
#         with st.spinner("Listening..."):
#             array,sr=librosa.load(uploaded_file)
#             tiempo=librosa.get_duration(array) #time in seconds
#             for t in range(0,int(tiempo),4):
#                 wave, sr = librosa.load(uploaded_file, mono=True,offset=t,duration=4)
#
#                 ## run it through the model
#                 mfcc=wav2mfcc(wave)
#
#                 X_test = np.reshape(mfcc,(1, 20, 170, 1))
#                 Y_predict=cnn_model.predict(X_test)
#
#                 if Y_predict.round()[0][0]==0 :
#                     plt.text(0,.8,'All sounds safe.',fontsize=14,color='slateblue')
#                     #st.write("All sounds safe.")
#
#                 if Y_predict.round()[0][0]==1:
#                     plt.text(.5,.8,'This is a firearm!')
#                     #st.write("This is a firearm! Contacting local authorities...")
#
#                 updateplot(wave)
#                 time.sleep(3)
#
#                 plt.show()
#     else:
#         st.write('Click the button to start listening.')
