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
1. Creating a boxplot for the distribution of bathrooms, bedrooms, and beds to identify outliers.

2. Plotting histograms for host_response_rate and review_scores_rating to observe their distributions and filling missing values with the median.

3. Generating a histogram for log_price to visualize the price distribution.

4. Computing and visualizing a correlation matrix and heatmap for selected numeric features to understand relationships between variables.

5. Creating a pairplot for the same subset of numeric features to visualize distributions and pairwise relationships.

6. For These 9 price and type plot,while log-transformed prices are valuable for analysis and modeling, presenting findings in log scale can be less intuitive for a general audience.Most people are not accustomed to thinking in terms of logarithmic scales in their daily lives.   Therefore, converting the log prices back to actual prices before presenting the results is often necessary. Actual prices give a direct, real-world interpretation of the costs involved.
 
## Data Preprocessing
To prepare the data for modeling, we performed the following preprocessing steps:

1. Min-Max Normalization: Applied to numerical features to bring them to a scale between 0 and 1, improving model performance and stability.

2. From the heatmap, number_of_reviews has a very low correlation with log_price (-0.01), suggesting that it has almost no linear relationship with the log of the price. Similarly, review_scores_rating also has a very low correlation with log_price (0.09), indicating a very weak linear relationship. So we might consider dropping number_of_reviews and review_scores_rating based on their low correlations with log_price. 

3. We might also use Bag of Words (BoW) or Term Frequency-Inverse Document Frequency (TF-IDF) methods to process the 'description' column.

4. Based on the bar chart, it appears that 'host_identity_verified' and 'instant_bookable' may have a minimal impact on pricing. So we might also consider dropping these 2 column.

5. Standardization is needed since accommodates, bathrooms, number_of_reviews, review_scores_rating, bedrooms, and beds are skewed. This would ensure that they all contribute equally to the analysis and that the model's performance is not inadvertently influenced by the natural variance in the dataset.

## Conclusion

1. First model: Linear & Polynomial Regression
   
   We built our models using 1st, 2nd, and 3rd Polynomial Regression. 

   1. The linear regression model yields a Mean Squared Error (MSE) of approximately 0.1927, indicating a moderate level of error in prediction. While not ideal, this suggests that the model captures a reasonable amount of the variance in the data. Further analysis is needed to assess potential underfitting or overfitting and to refine the model accordingly.
   
   2. The second-degree polynomial regression model shows slight improvement over the linear model, with a slightly lower MSE of approximately 0.1868. Additionally, the model achieves a relatively high R2 score of approximately 0.6444, indicating that around 64.44% of the variance in the dependent variable is explained by the independent variables. This suggests that the model captures more of the underlying complexity in the data compared to the linear model.
   
   3. I didn't get the result of 3rd-degree polynomial regression model after runing it for 1 hour.

   In conclusion, while both models provide valuable insights, the second-degree polynomial regression model demonstrates slightly better performance in terms of predictive accuracy, as evidenced by its lower MSE. However, further analysis may be necessary to ensure that the model's performance is robust and generalizable to unseen data.
   To potentially improve the performance of our regression model, several strategies can be considered:

      1. Cross-validation: Consider techniques like k-fold cross-validation or leave-one-out cross-validation.
   
      2. Observe the dataset: Explore additional features or transformations of existing features that might better capture the underlying relationships in the data.
   
      3. Outlier Detection and Removal: Identify and remove outliers or influential data points that might disproportionately affect the model's performance. 