import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import plotly.express as px

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


st.title("Bu model o'pkaning reyntgen tasvirlarini klassifikatsiya qiluvchi model")
st.image("https://upload.wikimedia.org/wikipedia/commons/1/15/Radiology_2706_1407_empyema_progression_nevit.gif", width=330)

file1 = st.file_uploader('Rasm yuklash', type=['png', 'jpg', 'jpeg', 'gif', 'svg'])
if file1:
    st.image(file1)

    img1 = PILImage.create(file1)

    model1 = load_learner('lung_disease_model.pkl')


    pred, pred_id, probs =  model1.predict(img1)
    foiz = probs[pred_id]*100
    if foiz>90:
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

        fig = px.bar(x=probs*100, y=model1.dls.vocab)
        st.plotly_chart(fig)
    else:
        st.markdown("Bu rasm o'pka tasviri emas")
  
st.title("Bu model o'pka rakini aniqlovchi model")
st.image("https://thumbs.gfycat.com/ImpoliteUnfoldedChameleon-size_restricted.gif", width=330)

file2 = st.file_uploader('Image upload', type=['png', 'jpg', 'jpeg', 'gif', 'svg'])
if file2:
    st.image(file2)

    img2 = PILImage.create(file2)

    model2 = load_learner('cancer_classification_model.pkl')


    pred, pred_id, probs =  model2.predict(img2)
    foiz = probs[pred_id]*100
    if foiz>90:
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

        fig = px.bar(x=probs*100, y=model2.dls.vocab)
        st.plotly_chart(fig)
    else:
        st.markdown("Bu rasm o'pka tasviri emas")
                    
            

