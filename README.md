# 📰 Fake News Detection App

A Machine Learning-powered web application that detects whether a news article is **Real** or **Fake** using Natural Language Processing (NLP) techniques and a trained classification model.

## Features

* Detects Fake and Real news articles.
* Supports single news prediction.
* Batch prediction using CSV/Excel datasets.
* Displays confidence scores and probability distribution.
* Extracts possible title and author information.
* Generates downloadable prediction reports.
* Interactive Streamlit dashboard.

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas
* Plotly
* Joblib
* NLP (TF-IDF Vectorization)

## Project Structure

```text
FakeNewsDetection/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
└── models/
    ├── model.pkl
    └── vectorizer.pkl
```

## Installation

### Clone the Repository

```bash
git clone git@github.com:HarshVSK3112004/FakeNewsDetection.git
cd FakeNewsDetection
```

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Model Details

* Vectorizer: TF-IDF
* Machine Learning Algorithm: Logistic Regression / Passive Aggressive Classifier
* Dataset: Fake and Real News Dataset
* Accuracy: ~99% (depending on training dataset)

## Usage

1. Enter a news article in the text box.
2. Click **Predict**.
3. View:

   * Prediction (Real/Fake)
   * Confidence Score
   * Probability Distribution
   * Extracted Title & Author

You can also upload a CSV or Excel file for batch predictions.

## Screenshots

* Single Prediction
* Dataset Testing
* Probability Distribution
* Downloadable Reports

## Author

**Harsh Vardhan Singh Kushwah**

* B.Tech Information Technology
* Amity University Gwalior

## License

This project is intended for educational and research purposes.
