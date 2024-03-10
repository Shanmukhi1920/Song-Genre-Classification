
# Song Genre Classification

## Project Overview
This project focuses on classifying songs based on features such as danceability, energy, acousticness, tempo, etc. into two music genres (Hip-Hop and Rock) using various machine learning techniques. The primary objective is to accurately predict the genre of a song based on its features. This repository contains the code, dataset, and documentation and web application for the project.

## Dataset
The analysis is based on two datasets:
- `echonest-metrics.json`: contains track metadata with genre labels
- `fma-rock-vs-hiphop.csv`: contains track metrics with the features

## Methods Used
- Dimensionality reduction using Principal Component Analysis (PCA).
- Machine Learning models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree (DT)
  - Random Forest (RF)


## Installation and Usage
To set up the project, follow these steps
1. Clone the repository
```bash
git clone https://github.com/Shanmukhi1920/Song_Genre_Classification
```
2. Navigate the project directory
```bash
cd Song_Genre_Classification
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run Jupyter Notebook to view the project
```bash
jupyter notebook
```
In Jupyter, open the `Song_Genre_Classification.ipynb` notebook to view the full analysis.

## Web Application
The `app.py` file in the repository launches a web application built with Streamlit that allows users to input song features and receive a genre classification in real-time.

## Running the Web App
Ensure you have Streamlit installed. If not, install it using pip:
```bash
pip install streamlit
```
Launch the app by running the following command in the terminal:
```bash
streamlit run app.py
```

## Saved Model and Preprocessing Files
The best-performing model, along with the PCA transformation and scaler used for preprocessing, are saved as .pkl files. These include:

`song_classifier.pkl`: The trained classification model.

`scaler.pkl`: The scaler used to normalize features.

`pca.pkl`: The PCA transformation applied to reduce dimensionality.

## Results
**Performance Consistency:** Logistic Regression and SVM demonstrated consistent performance across cross-validation and test sets.

**Overfitting in Tree-based Models:** Decision Tree and Random Forest showed a significant difference in train and test accuracies, indicating potential overfitting.

**Hyperparameter Tuning:** Post-tuning, the Decision Tree and Random Forest models exhibited improved test set performance with accuracies around 82% and 85%, respectively.

**Insight on Model Tuning:** The Random Forest Classifier's test accuracy decreased slightly from 85.49% to 84.84% after tuning, highlighting that hyperparameter tuning does not always enhance performance. This serves as an important lesson in model optimization.
