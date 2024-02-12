# 151A-project


## About the Data Set
This project aims to predict the renting prices in different cities using various features from the dataset, such as property type, room type, bed type, cancellation policy, city, cleaning fee, host profile picture availability, host identity verification, and instant bookability. Our goal is to understand how these features influence renting prices and to build a predictive model to assist renters and landlords alike.


## Dataset Description

The dataset comprises listings with features that describe the property, host, and booking policies. Here are some key details:

Number of Instances (records in the data set): __74111__

Number of Attributes (fields within each record, including the class): __29__

Target Variable: The target variable is the renting price(log_price), which we aim to predict.

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
   3. We drop latitude and longitude column because it is difficult to process this kind of data and we already have City column.

## Plots and Visualizations
In this analysis, heatmap, pairplot, and box plots were used to visualize the data. Based on the visualized data, we will need to do data preprocessing on following collomns:

1. One-Hot Encoding
   1. We draw box plots for each colomns contains catergory data to illustrate the relationship between the data and Airbnb price. We found that there is certain relations between the catergory data and Airbnb price. Some catergory would be much higher Airbnb price than others. Thus, we will neet to One-Hot Encoding the columns to include them as features in the after machine learning process.
2. Normalization and Standardization:
   1. Our data need to be normalized since some feature have a large number, such as number_of_reviews. It is range from 0 - 600. These features are measured on different scales and have different ranges of values, so normalizing them would ensure that each feature contributes approximately proportionately to the final result. 
   2. Standardization is needed since accommodates, bathrooms, number_of_reviews, review_scores_rating, bedrooms, and beds are skewed. This would ensure that they all contribute equally to the analysis and that the model's performance is not inadvertently influenced by the natural variance in the dataset.
   3. For These 9 price and type plot,while log-transformed prices are valuable for analysis and modeling, presenting findings in log scale can be less intuitive for a general audience.Most people are not accustomed to thinking in terms of logarithmic scales in their daily lives.   Therefore, converting the log prices back to actual prices before presenting the results is often necessary. Actual prices give a direct, real-world interpretation of the costs involved.
 
## Data Preprocessing
To prepare the data for modeling, we performed the following preprocessing steps:

1. Min-Max Normalization: Applied to numerical features to bring them to a scale between 0 and 1, improving model performance and stability.

2. From the heatmap, number_of_reviews has a very low correlation with log_price (-0.01), suggesting that it has almost no linear relationship with the log of the price. Similarly, review_scores_rating also has a very low correlation with log_price (0.09), indicating a very weak linear relationship. So we might consider dropping number_of_reviews and review_scores_rating based on their low correlations with log_price. 

3. We might also use Bag of Words (BoW) or Term Frequency-Inverse Document Frequency (TF-IDF) methods to process the 'description' column.

4. Based on the bar chart, it appears that 'host_identity_verified' and 'instant_bookable' may have a minimal impact on pricing. So we might also consider dropping these 2 column.
