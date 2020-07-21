import streamlit as st
import argparse
import numpy as np
from tensorflow.keras.models import load_model
import librosa

import sounddevice as sd
import matplotlib.pyplot as plt
##
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'filename', nargs='?', metavar='FILENAME',
    help='audio file to store recording to')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
args = parser.parse_args(remaining)
#
#
if args.samplerate is None:
    device_info = sd.query_devices(args.device, 'input')
    # soundfile expects an int, sounddevice provides a float:
    args.samplerate = int(device_info['default_samplerate'])

def wav2mfcc(wave, sr=args.samplerate,n_mfcc=20, max_len=170):
    '''wave is a np array'''
    #wave, sr = librosa.load(file_path, mono=True, sr=None)
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

def record(sr=args.samplerate, channels=args.channels, duration=4, filename='pred_record.wav'):
    """
    Records live environment surroundings
    """
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels).reshape(-1)
    sd.wait()

    line.set_ydata(recording)
    the_plot.pyplot(plt)

    return recording

# load the model
model_path="models/"
cnn_model=load_model(model_path+'bal_cnn_model_accuracy_97.7_alpha_0.0001.h5')

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
#-----------------------------------------------
# select a sidebar to navigate between different options of the app
options=['Test with some sample clips', 'Record your sound']
page=st.sidebar.radio('Select an option',options)

st.sidebar.header('Firearm-Alarm Options')
st.sidebar.markdown('The first option will allow you to test firearm-alarm with some pre-recorded sound clips.')
st.sidebar.markdown('The second option will enable you to have firearm-alarm listen to your environment and report its findings.')
#######
if page==options[0]: #The first option is selected
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

        if Y_predict.round()[0][0]==1 :

            st.write("This doesn't sound like a firearm.")

        if Y_predict.round()[0][0]==0:

            st.write("This is a firearm! Contacting local authorities...")

    else:
        st.write('Click the button to analyze the audio clip.')

###############################################----------------------------------
elif page==options[1]: #if the second page is selected
    st.header('Firearm Alarm in Action')
    st.text('For how long would you like Firearm Alarm to listen?')
    t=st.slider('Select the time (hours).',min_value=1,max_value=24,step=1)

    x = np.arange(0, 4,1/int(args.samplerate))
    fig, ax=plt.subplots()
    ax.set_ylim(-1, 1)
    line, = ax.plot(x, np.zeros(len(x)),color='m',linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Sound Wave')
    the_plot = st.pyplot(plt)

    if st.button('Start listening with Firearm Alarm'):
        with st.spinner("Listening..."):
            for i in range(0,int(t*3600)):

                recording=record()

                ## run it through the model
                mfcc=wav2mfcc(recording)

                X_test = np.reshape(mfcc,(1, 20, 170, 1))
                Y_predict=cnn_model.predict(X_test)

                if Y_predict.round()[0][0]==1 :
                    plt.text(0,.8,'All sounds safe.',fontsize=14,color='slateblue')
                    #st.write("All sounds safe.")

                if Y_predict.round()[0][0]==0:
                    plt.text(.5,.8,'This is a firearm!')
                    #st.write("This is a firearm! Contacting local authorities...")

                plt.show()
    else:
        st.write('Click the button to start listening.')
