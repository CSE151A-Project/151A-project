# 151A-project


## About the Data Set
This project aims to predict the renting prices in different cities using various features from the dataset, such as property type, room type, bed type, cancellation policy, city, cleaning fee, host profile picture availability, host identity verification, and instant bookability. Our goal is to understand how these features influence renting prices and to build a predictive model to assist renters and landlords alike.


## Dataset Description

Number of Instances (records in the data set): __74111__

Number of Attributes (fields within each record, including the class): __29__

## Data Cleaning and Preprocessing Overview

In this analysis, we perform several steps to clean and preprocess the Airbnb dataset to prepare it for further analysis and modeling. The process involves handling missing values, encoding categorical variables, and ensuring data integrity. Below are the detailed steps undertaken:

1. Data Importation: The dataset originally in CSV format, is compressed into a .tar.gz file. This step is crucial for efficient file storage and management, enabling us to upload the dataset to GitHub without encountering issues related to file size limitations. 
2. Initial Data Exploration: We checked the shape of the dataset to get an idea of its size and remove the last row to clean up the data, ensuring that only relevant records are analyzed.
3. Data Cleaning: 
   1. We drop columns with excessive unique categories or missing data that would be impractical to one-hot encode or impute, such as 'thumbnail_url', 'zipcode', 'neighbourhood', 'first_review', and 'last_review'.
   2. We convert the id column to a numeric type and rename it to 'id' for clarity and ease of reference.
4. One-Hot Encoding: Categorical variables such as 'property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city', 'cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', and 'instant_bookable' are one-hot encoded. This process transforms these categorical variables into a format that can be used in machine learning algorithms, creating separate binary columns for each category.
5. Handling Missing Values: 
   1. A boxplot is drawn for the variables 'bathrooms', 'bedrooms', and 'beds' to visualize the distribution and identify outliers. Based on the presence of outliers, median values are chosen as a more reliable measure for imputing missing values. Missing values in 'bathrooms', 'bedrooms', and 'beds' are imputed with the median of each column, grouped by the 'accommodates' category, to maintain the integrity of the data.
   2. Missing values in 'host_response_rate' and 'review_scores_rating' are imputed with their respective median values. Before imputation, the 'host_response_rate' is converted from a percentage string to a float.

## Plots and Visualizations
Scatter plots, histograms, and box plots were used to visualize the data. ???

## Data Preprocessing
To prepare the data for modeling, we performed the following preprocessing steps:

Min-Max Normalization: Applied to numerical features to bring them to a scale between 0 and 1, improving model performance and stability.
Handling Missing Data: Missing values were addressed through imputation or removal, depending on the feature and the amount of missing data.
