# Market Sales Prediction

## Overview

This project aims to predict product sales across different retail outlets using Machine Learning. By analyzing product characteristics and store information, the model estimates the expected sales for each item.

The project helps retailers understand the factors affecting sales and supports data-driven business decisions.

## Dataset

The dataset contains information about products and outlets, including:

* Item Weight
* Item Fat Content
* Item Visibility
* Item Type
* Item MRP (Maximum Retail Price)
* Outlet Size
* Outlet Location Type
* Outlet Type
* Outlet Establishment Year

### Target Variable

* **Item_Outlet_Sales** – Sales of a product in a particular outlet.

## Project Workflow

### 1. Data Preprocessing

* Removed unnecessary identifier columns
* Handled missing values
* Encoded categorical variables
* Prepared data for machine learning models

### 2. Exploratory Data Analysis (EDA)

* Analyzed sales distribution
* Examined product categories
* Investigated outlet characteristics
* Identified relationships between features and sales

### 3. Model Development

The project uses **XGBoost Regressor** to predict product sales.

Model Parameters:

* n_estimators = 500
* max_depth = 5
* learning_rate = 0.03

### 4. Model Saving

The trained model is stored using Pickle:

```python
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Pickle
* Streamlit

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/market-sales-prediction.git
cd market-sales-prediction
```

Install required packages:

```bash
pip install -r requirements.txt
```

## Run the Project

Train the model:

```bash
python train.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```text
market-sales-prediction/
│
├── data/
│   └── market.csv
│
├── model.pkl
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

## Features

* Predict outlet sales using product and store attributes
* Handle missing values automatically
* Interactive Streamlit user interface
* Fast and accurate predictions with XGBoost

## Business Value

This project can help businesses:

* Forecast product sales
* Optimize inventory management
* Improve pricing strategies
* Identify high-performing outlets
* Support business planning and decision-making

## Future Improvements

* Hyperparameter optimization
* Advanced feature engineering
* Model comparison with Random Forest and LightGBM
* Cloud deployment
* Real-time sales forecasting

## Author
Rasul Ibrahimov
Developed as a Machine Learning regression project for retail sales prediction using XGBoost and Streamlit.
 
