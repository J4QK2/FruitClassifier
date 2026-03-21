import streamlit as st
import joblib
import numpy as np
from PIL import Image

model = joblib.load("model.pkl")

uploaded_file = st.file_uploader("Upload image", type=["jpg"])


st.title("Fruit Classifier app")

st.divider()

st.write("This app can tell what fruit is shown in the picture")

st.divider()

fruit_class_button = st.button("What fruit is this?")

if fruit_class_button:
    st.balloons()

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.resize((224,224))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1,224,224,3)
        st.image(img, caption="Uploaded Image")
    
    prediction = model.predict(img_array)
    print(prediction)

    pred_class = np.argmax(prediction, axis=1)
    st.write("Predicted class index:", pred_class[0])
    st.write(prediction)
else:
    st.write("error")