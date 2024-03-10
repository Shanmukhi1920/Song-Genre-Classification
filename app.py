import streamlit as st
import numpy as np
from joblib import load

# Load scaler, PCA and model
model =load('song_classifier.pkl')
pca = load('pca.pkl')  
scaler = load('scaler.pkl')

# Streamlit UI
st.title('Song Genre Classification')
st.write("""
### Prediction Model Information
This app predicts the genre of a song as either **Hip Hop** or **Rock** based on various musical features.
Please adjust the sliders below to set the values for these features, and then click **Classify** to see the predicted genre.
""")

# Creating form for user input
with st.form("my_form"):
    st.write("Please enter the song features:")

    # Sliders for each feature, setting min and max values based on our dataset summary
    acousticness = st.slider('Acousticness', min_value=0.0, max_value=1.0, value=0.01)
    danceability = st.slider('Danceability', min_value=0.05, max_value=0.96, value=0.01)
    energy = st.slider('Energy', min_value=0.0, max_value=1.0, value=0.01)
    instrumentalness = st.slider('Instrumentalness', min_value=0.0, max_value=1.0, value=0.01)
    liveness = st.slider('Liveness', min_value=0.025, max_value=0.97, value=0.01)
    speechiness = st.slider('Speechiness', min_value=0.023, max_value=0.97, value=0.01)
    tempo = st.slider('Tempo', min_value=29.0, max_value=250.0, value=0.01)
    valence = st.slider('Valence', min_value=0.01, max_value=0.98, value=0.01)

    # Every form must have a submit button
    submitted = st.form_submit_button("Predict Song Genre")
    if submitted:
        # Preprocess inputs
        features = np.array([acousticness,danceability,energy,instrumentalness,liveness,speechiness,tempo,valence]).reshape(1, -1)
        
        # Apply scaling and PCA if necessary
        scaled_features = scaler.transform(features)
        pca_features = pca.transform(scaled_features)
        
        # Predict and display the result
        prediction = model.predict(pca_features)
        st.write(f'For the given song features, the predicted Genre is: **{prediction[0]}**')
