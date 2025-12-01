Predicting P2P Loan Default with Machine Learning
This repository contains the end-to-end data science pipeline for predicting loan defaults using the LendingClub dataset. The project implements three models (Logistic Regression, XGBoost, and CatBoost) to identify high-risk borrowers.

Champion Model: CatBoost Classifier (Accuracy: 66.85%, Recall: 68.7%)

Repository Structure
01_initial_data_inspection.ipynb: Loads raw data and performs initial inspection.
02_data_preprocessing.ipynb: Cleans data, handles missing values, and engineers features.
03_data_sampling_master.ipynb: Creates the balanced 50/50 "Master Sample" for training.
04_exploratory_data_analysis.ipynb: Visualizes key trends (Bivariate/Multivariate analysis).
5_1_data_prep_logistic_regression.ipynb: Prepares data (Scaling/One-Hot) specifically for the linear model.
6_1_logistic_regression.ipynb: Trains and evaluates the baseline model.
5_2_data_prep_xgboost.ipynb: Prepares data (Ordinal Encoding) for XGBoost.
6_2_xg_boost.ipynb: Trains the XGBoost model.
5_3_data_prep_catboost.ipynb: Prepares the "Full Feature" dataset (keeping int_rate/grade) for CatBoost.
6_3_categorical_boosting.ipynb: Trains the champion CatBoost model.
07_model_evaluation.ipynb: Generates the Combined ROC Curve and Final Leaderboard.

How to Reproduce Results
To reproduce the analysis and model results, please follow these steps strictly in order.

Prerequisites
A Google Account (to access Google Colab).

The raw dataset file: lc_accepted_loans_full_2007to2018.csv. 
Dataset link (https://www.kaggle.com/datasets/wordsforthewise/lending-club?select=accepted_2007_to_2018Q4.csv.gz)

Step-by-Step Instructions
1. Data Preparation (The Pipeline)

  Open 02_data_preprocessing.ipynb in Google Colab.

  Upload the raw dataset lc_accepted_loans_full_2007to2018.csv to the session.

  Run all cells. This will generate lc_loans_cleaned.csv. Download this file.

  Open 03_data_sampling_master.ipynb. Upload lc_loans_cleaned.csv.

  Run all cells to generate lc_loans_master_sample.csv. Download this file.

2. Model-Specific Data Prep

  For Logistic Regression: Open 5_1_data_prep_logistic_regression.ipynb. Upload lc_loans_master_sample.csv. Run all cells to generate X_train.csv, y_train.csv, etc.

  For XGBoost: Open 5_2_data_prep_xgboost.ipynb. Upload lc_loans_master_sample.csv. Run all cells to generate X_train_tree.csv, y_train_tree.csv, etc.

  For CatBoost: Open 5_3_data_prep_catboost.ipynb. Upload lc_loans_master_sample.csv. Run all cells to generate X_train_tree_full.csv, y_train_tree_full.csv, etc.

3. Running the Final Comparison To see the final results table and ROC curve immediately:

  Open 07_model_evaluation.ipynb.

  Upload ALL 12 generated CSV files from Step 2 (the standard, tree, and tree_full versions of train/test data).

  Run all cells.

  The notebook will output the Final Leaderboard and the ROC Curve comparison plot.

Dependencies
These notebooks are designed to run in Google Colab, which pre-installs most necessary libraries.
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost (Note: The notebooks include a !pip install catboost command to install this automatically).
