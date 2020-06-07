import streamlit as st
import numpy as np
import librosa 
esc50path="/Users/aarojas/Documents/Data_Science_Resources/Insight_20B/AAR_Insight_Project/Firearm_Detection/data/raw/ESC-50-master/audio/"
fapath="/Users/aarojas/Documents/Data_Science_Resources/Insight_20B/AAR_Insight_Project/Firearm_Detection/data/raw/Firearms/"
audio_clip1='5-160614-H-48.wav' #fireworks
audio_clip2='5-244526-A-26.wav' #laughing
audio_clip3='385785__rijam__rifle-shot-m1-garand.mp3'

st.title('Firearm Alarm')
st.header('Listening for Firearms in Your Home')

st.text('The following are a set of sample audio clips that can be input into the model.')


st.audio(esc50path+audio_clip1)


st.text('This is audio clip 1.')

st.audio(esc50path+audio_clip2)

st.text('This is audio clip 2.')

st.audio(fapath+audio_clip3)

option = st.selectbox('Select the clip you would like the model to analyze.',('Audio clip 1', 'Audio clip 2', 'Audio clip 3'))
st.write('You selected:', option)

if st.button('Analyze '+option):
    st.write('The result is...tbd')
else:
    st.write('Click the button to analyze the audio clip.')
