import streamlit as st
import numpy as np
import librosa
import pickle

# load the model from disk
model_path="models/"
log_res_model="logistic_regression_model.sav"
loaded_model = pickle.load(open(model_path+log_res_model, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

path="data/external/"
audio_clip1='5-160614-H-48.wav' #fireworks
audio_clip2='5-244526-A-26.wav' #laughing
audio_clip3='385785__rijam__rifle-shot-m1-garand.mp3'

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
    st.write('The result is...tbd')
else:
    st.write('Click the button to analyze the audio clip.')
