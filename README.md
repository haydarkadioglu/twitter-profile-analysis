# Note: I'm stil working on it

# Sentence Analysis Project

This project performs sentence analysis to determine how "good" or "bad" a sentence is, by predicting the percentage of positivity or negativity. The project uses machine learning techniques and regression models to perform this analysis.

## Models Used
The project utilizes various regression models for sentence analysis. Some of the models include:

- **Linear Regressor**: `MultiOutputRegressor(LinearRegression())`
- **AdaBoost Regressor**: `MultiOutputRegressor(AdaBoostRegressor())`
- **K Neighbors Regressor**: `MultiOutputRegressor(KNeighborsRegressor())`
- **CatBoost Regressor**: `MultiOutputRegressor(CatBoostRegressor(learning_rate=0.1, depth=6, iterations=500, verbose=0))`
- **XGBoost Regressor**: `MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror'))`

Some models, such as **Random Forest Regressor** and **Decision Tree Regressor**, have been included but are currently commented out.

## Project Overview

This project uses natural language processing (NLP) techniques to process text data and make predictions using various regression algorithms. The developed model determines the sentiment of each sentence, predicting how "good" or "bad" it is as a percentage.

### Libraries Used
- `sklearn`: For regression and modeling.
- `xgboost`: For using the XGBoost model.
- `catboost`: For using the CatBoost model.
- `pandas`: For data processing.
- `numpy`: For mathematical computations.

## Installation

To run the project, clone the repository and install the required libraries:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<username>/sentence-analysis.git
   cd sentence-analysis
