import streamlit as st
import numpy as np
import pandas as pd
directory="/Users/aarojas/Documents/Data_Science_Resources/Insight_20B/AAR_Insight_Project/Firearm_Detection/data/raw/ESC-50-master/audio/"
file='1-115545-A-48.wav' #fireworks
st.title('Stay Safe')
st.header('Listening for Firearms in Your Home')

st.text('The following is an example audio clip that is input into the model.')




st.audio(directory+file)


st.text('This audio clip ____.')
