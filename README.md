

## Project Overview
This project focuses on classifying songs as either Hip-Hop or Rock based on audio features like danceability, energy, and tempo using machine learning techniques. The goal is to accurately predict a song's genre, demonstrating how AI can be applied to music analysis. This work has potential applications in the music industry, including automating playlist creation, enhancing recommendation systems, and assisting in music production. It also serves as a valuable tool for music researchers studying genre characteristics. 

## Dataset
The analysis is based on two datasets:
- `data/echonest-metrics.json`: contains track metadata with genre labels
- `data/fma-rock-vs-hiphop.csv`: contains track metrics with the features

## Methodology

### 1. Data Collection and Preprocessing

- Used a dataset compiled by The Echo Nest, containing audio features of songs classified as either 'Hip-Hop' or 'Rock'.
- Features included acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, and valence.
- Merged track metadata with audio features using track IDs.
- Handled class imbalance by undersampling the majority class ('Rock') to match the number of 'Hip-Hop' samples.

### 2. Exploratory Data Analysis (EDA)

- Visualized the distribution of each audio feature using histograms.
- Created box plots to compare feature distributions between genres.
- Computed correlation matrices using both Pearson and Spearman methods to assess feature relationships.

### 3. Feature Engineering and Selection

- Applied StandardScaler to normalize all features (mean=0, std=1).
- Performed Principal Component Analysis (PCA) for dimensionality reduction.
- Analyzed scree plot and cumulative explained variance to determine the optimal number of components.
- Selected 6 principal components, explaining approximately 85% of the variance.

### 4. Model Development and Evaluation

- Implemented four classification models: Logistic Regression, Support Vector Machine (SVM), Decision Tree, and Random Forest.
- Used 5-fold cross-validation to assess initial model performance.
- Performed GridSearchCV for hyperparameter tuning of Decision Tree and Random Forest models.
- Evaluated the best models on the test set using accuracy, precision, recall, and F1-score.

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
In Jupyter, open the `Song_Genre_Classification.ipynb` notebook in the `notebooks/` directory to view the full analysis.

## Web Application
The `src/app.py` file in the repository launches a web application built with Streamlit that allows users to input song features and receive a genre classification in real-time.

## Running the Web App
Ensure you have Streamlit installed. If not, install it using pip:
```bash
pip install streamlit
```
Launch the app by running the following command in the terminal:
```bash
streamlit run src/app.py
```

## Saved Model and Preprocessing Files
The best-performing model, along with the PCA transformation and scaler used for preprocessing, are saved as .pkl files. These include:

`models/song_classifier.pkl`: The trained classification model.

`models/scaler.pkl`: The scaler used to normalize features.

`models/pca.pkl`: The PCA transformation applied to reduce dimensionality.

## Results
**Performance Consistency:** Logistic Regression and SVM demonstrated consistent performance across cross-validation and test sets.

**Overfitting in Tree-based Models:** Decision Tree and Random Forest showed a significant difference in train and test accuracies, indicating potential overfitting.

**Hyperparameter Tuning:** Post-tuning, the Decision Tree and Random Forest models exhibited improved test set performance with accuracies around 82% and 85%, respectively.

**Insight on Model Tuning:** The Random Forest Classifier's test accuracy decreased slightly from 85.49% to 84.84% after tuning, highlighting that hyperparameter tuning does not always enhance performance. This serves as an important lesson in model optimization.
