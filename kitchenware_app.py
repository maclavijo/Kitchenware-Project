import streamlit as st
import pandas as pd
from PIL import Image
from httprequest import httpreq

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: black; font-size: 45px'>Kitchenware Classifier</h1>")#, unsafe_allow_html=True)
st.write("#")
st.write("#")

img = '0190.jpg'
factor = 0.4
imag = Image.open(f'./test/{img}')
x,y = imag.size 
x,y = int(factor * x), int(factor * y)
imag = imag.resize((x, y))
a = httpreq(img)
st.image(imag, caption="test")
st.write(a)
