import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import time

import matplotlib.pyplot as plt


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

def updateplot(wave,txt_output):
    """
    update the plot with the wave file
    """

    line.set_ydata(wave)
    the_plot.pyplot(plt)
    text.set_text(txt_output)

# load the model from disk
model_path="models/"
cnn_model=load_model(model_path+'bal_cnn_model_accuracy_98.2_alpha_0.0001.h5')

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
options=['Test with some sample clips', 'Test with a youtube video']
page=st.sidebar.radio('Select an option',options)

st.sidebar.header('Firearm-Alarm Options')
st.sidebar.markdown('The first option will allow you to test firearm-alarm with some pre-recorded sound clips.')
st.sidebar.markdown('The second option will enable you to have firearm-alarm listen to a youtube clip: https://www.youtube.com/watch?v=1N_m3tsPyP0.')
#-----------------------------------------------
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

    x = np.arange(0, 4,1/22050)
    fig, ax=plt.subplots()
    ax.set_ylim(-1, 1)
    line, = ax.plot(x, np.zeros(len(x)),color='m',linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Sound Wave')
    the_plot = st.pyplot(plt)
    text=plt.text(0,.8,'',fontsize=14)


    sample='data/external/Real_life_gunshot_sound_effects.wav'
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

                if Y_predict.round()[0][0]==1 :
                    txt_output='No firearm sound(s) detected'
                    # text.set_text('No firearm sounds detected')

                if Y_predict.round()[0][0]==0:
                    txt_output='Firearm sound(s) detected!'
                    # text.set_text('Firearm sounds detected!')

                updateplot(wave,txt_output)
                time.sleep(3)

                plt.show()
    else:
        st.write('Click the button to start listening.')

#-----------------------------------
