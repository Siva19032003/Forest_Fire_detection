import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns



IM_SIZE = 224
MODEL_PATH = "Fire_detection.keras"   # <- Use the final .keras file you saved
CLASS_NAMES = ["Nowildfire", "Wildfire"]  # Must match training order

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_trained_model()

# ---------------- PREDICT FUNCTION ----------------
def predict_image(model, image):
    img = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img, (IM_SIZE, IM_SIZE))
    arr = img_resized.astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr, verbose=0)
    idx = np.argmax(preds[0])
    predicted_class = CLASS_NAMES[idx]
    confidence = float(preds[0][idx])
    return predicted_class, confidence, preds[0]

# ---------------- VISUALIZATION ----------------
def plot_probabilities(probs):
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, probs, color=['green', 'red'])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    for i, v in enumerate(probs):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center')
    st.pyplot(fig)

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Forest Fire Detection")

# Sidebar Navigation
menu = st.sidebar.radio(
    "Menu",
    ["Home", "Upload Image", "About"]
)

# ---------------- HOME PAGE ----------------
if menu == "Home":
    st.title("Forest Fire Detection using Satellite Images")
    

    st.subheader("""
    Welcome to the **Forest Fire Detection App**!  
    This tool uses a **CNN deep learning model** trained on satellite images  
    to classify whether an image contains **Wildfire** or **No Wildfire**.""")
    
    st.subheader(""" **Explore the Dataset**
    The dataset comes from the [Wildfire Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset).  
    It consists of **satellite images** categorized into two classes:
    - **No wild fire** 
    - **Wild fire** 
    """)

    import os
    import random
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image

    DATA_ROOT = "dataset"  # Change this if dataset is elsewhere
    train_path = os.path.join(DATA_ROOT, "train")
    valid_path = os.path.join(DATA_ROOT, "valid")
    test_path  = os.path.join(DATA_ROOT, "test")

    if os.path.exists(DATA_ROOT):
        # Count images per class
        def count_images(path):
            if not os.path.exists(path): return {}
            return {cls: len(os.listdir(os.path.join(path, cls))) for cls in os.listdir(path)}

        train_counts = count_images(train_path)
        test_counts  = count_images(test_path)

        st.subheader("Dataset Overview")
        st.write("**Training Set:**", train_counts)
        st.write("**Test Set:**", test_counts)
        
        st.subheader("Dataset Visualizations")

    DATA_ROOT = "dataset"  # same as in Home
    train_path = os.path.join(DATA_ROOT, "train")
    valid_path = os.path.join(DATA_ROOT, "valid")
    test_path  = os.path.join(DATA_ROOT, "test")

    if os.path.exists(DATA_ROOT):
        # --- Class Distribution ---
        st.write("**Class Distribution**")
        def count_images(path):
            if not os.path.exists(path): return {}
            return {cls: len(os.listdir(os.path.join(path, cls))) for cls in os.listdir(path)}

        train_counts = count_images(train_path)
        
        test_counts  = count_images(test_path)

        dist_data = []
        for split_name, counts in zip(["Train", "Test"], [train_counts,test_counts]):
            for cls, count in counts.items():
                dist_data.append({"Split": split_name, "Class": cls, "Count": count})
        df = pd.DataFrame(dist_data)

        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=df, x="Split", y="Count", hue="Class", ax=ax)
        plt.title("Image Distribution")
        st.pyplot(fig)

        # --- Random Samples ---
        st.write("**Random Sample Images**")
        cols = st.columns(3)
        for i, cls in enumerate(CLASS_NAMES):
            class_folder = os.path.join(train_path, cls)
            if os.path.exists(class_folder) and len(os.listdir(class_folder)) > 0:
                for _ in range(2):  # show 2 samples per class
                    sample_img = random.choice(os.listdir(class_folder))
                    img_path = os.path.join(class_folder, sample_img)
                    img = Image.open(img_path)
                    with cols[i % 4]:
                        st.image(img, caption=f"{cls}", use_container_width=True)

        # --- Image Size Distribution ---
        st.subheader("**Image Size Distribution**")
        sizes = []
        for split in [train_path, valid_path, test_path]:
            for cls in CLASS_NAMES:
                folder = os.path.join(split, cls)
                if os.path.exists(folder):
                    for img_file in random.sample(os.listdir(folder), min(20, len(os.listdir(folder)))):  # sample 20
                        img_path = os.path.join(folder, img_file)
                        try:
                            img = Image.open(img_path)
                            sizes.append({"Class": cls, "Width": img.width, "Height": img.height})
                        except:
                            pass
        if sizes:
            df_sizes = pd.DataFrame(sizes)
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(data=df_sizes, x="Width", y="Height", hue="Class", ax=ax)
            plt.title("Image Size Distribution")
            st.pyplot(fig)

        # --- Color Histogram ---
        st.subheader("4. Color Histogram")
        for cls in CLASS_NAMES:
            class_folder = os.path.join(train_path, cls)
            if os.path.exists(class_folder) and len(os.listdir(class_folder)) > 0:
                sample_img = random.choice(os.listdir(class_folder))
                img_path = os.path.join(class_folder, sample_img)
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img)

                fig, ax = plt.subplots(figsize=(6,3))
                colors = ('r','g','b')
                for i, col in enumerate(colors):
                    hist = cv2.calcHist([arr], [i], None, [256], [0,256])
                    ax.plot(hist, color=col)
                plt.title(f"Color Histogram - {cls}")
                st.pyplot(fig)

    else:
        st.warning("Dataset not found. Please place it in `dataset/`.")



    # --- Project Details ---
    st.subheader("Project Details")
    st.write("""
    - **Goal:** Early detection of wildfires using Satellite Images.  
    - **Model:** CNN Model  
    - **Input:** Satellite images (224×224).  
    - **Output:** Binary classification  (**Wildfire / Nowildfire*)*.  
    - **Tech Stack:** Python, TensorFlow/Keras, OpenCV, Streamlit.  

    This project demonstrates how **deep learning** can be applied to real-world environmental challenges.
    """)




# ---------------- UPLOAD IMAGE PAGE ----------------
elif menu == "Upload Image":
    st.title("Upload Image for Prediction")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            pred_class, conf, probs = predict_image(model, image)

            st.success(f"Prediction: **{pred_class}**")
            st.write(f"**Confidence:** {conf:.2%}")

            st.subheader("Class Probabilities")
            st.write({c: f"{float(p):.2%}" for c, p in zip(CLASS_NAMES, probs)})

            st.subheader("Probability Visualization")
            plot_probabilities(probs)




# ---------------- ABOUT PAGE ----------------
elif menu == "About":
    st.title("About this Project")
    st.write("""
    - **Project:** Forest Fire Detection Using Satellite Image  
    - **Model:** CNN  
    - **Dataset:** [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)  
    - **Developer:** SivaKumar M  
    - **Contact:** [sivakumarmurugan193@gmail.com](mailto:sivakumarmurugan193@gmail.com)  
    -**github** [github.com/Siva19032003/Forest_Fire_detection](https://github.com/Siva19032003/Forest_Fire_detection)

    This application helps in **early detection of forest fires** using satellite imagery.  
    """)