import os
import streamlit as st
import pandas as pd
from PIL import Image
from httprequest import httpreq

st.set_page_config(layout="wide")

col1, col2, col3  = st.columns(3)

with col2:

    st.title("Kitchenware Classifier")
    #st.markdown("<h1 style='text-align: center; color: black; font-size: 45px'>Kitchenware Classifier</h1>")#, unsafe_allow_html=True)
    st.write("#")
    st.write("#")

c1, c2, c3, c4, c5  = st.columns(5)

testList = pd.read_csv('test.csv')

# Using object notation
add_selectbox1 = st.sidebar.selectbox(
    "Choose Category?",
    ("Cup", "Glass", "Plate", "Spoon", "Fork", "Knife"),
)

imageList = {
    "Cup"       : ['0000', '0008', '0015', '2744', '3242', '3247', '8170'],
    "Glass"     : ['0022', '1239', '3168', '2103', '5788', '7522', '9374'],
    "Plate"     : ['0019', '0967', '2724', '3135', '4673', '7263', '9168'],
    "Spoon"     : ['0190', '0848', '1739', '3049', '4366', '6106', '9085'],
    "Fork"      : ['0136', '1206', '2113', '3833', '5565', '7261', '9271'],
    "Knife"     : ['0018', '0510', '1742', '2721', '3277', '4770', '8204'],
     }

add_selectbox2 = st.sidebar.selectbox(
    "Choose Image?",
    (imageList[add_selectbox1]),
)

img = add_selectbox2 + '.jpg'
imag = Image.open(f'./images/{img}')
factor = 0.4
x,y = imag.size 
x,y = int(factor * x), int(factor * y)
imag = imag.resize((x, y))

print(img)
a = httpreq(img)

with c2:
    st.image(imag, caption=f"This should be a {add_selectbox1}")

with c4:    
    st.write('\n')
    st.write('\n')
    st.subheader('Predicted values:\n\n')
    df = pd.DataFrame.from_dict({'Label':a.keys(), 'Probability (%)':a.values()})
    df.set_index('Label', inplace=True)
    df.sort_values(by='Probability (%)', ascending=False, inplace=True)
    st.dataframe(df.style.highlight_max(axis=0), width=250)
    


#st.dataframe(df)