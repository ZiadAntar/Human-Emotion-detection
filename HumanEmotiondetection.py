import streamlit as st
import numpy as np
import cv2
from deepface import DeepFace 
from PIL import Image
st.title("Human Emotion detection")
st.write("Upload Image")

def analyze_img(image):
    analysis=DeepFace.analyze(image,actions=["emotion"])
    return analysis[0]["emotion"]

upload_file=st.file_uploader("choose file",type=["png","jpg","jpeg"])

if upload_file is not None:
    img=Image.open(upload_file)
    img_np=np.array(img)
    st.image(img_np,channels="RGB")
    emotion_scores=analyze_img(img_np)
    detectemotion=max(emotion_scores,key=emotion_scores.get)
    st.write(f"Detected emotion",{detectemotion})
    # print(emotion_scores)