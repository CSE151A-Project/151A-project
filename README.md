# Project Title

## Introduction
(add more)
This project aims to predict the renting prices in different cities using various features from the dataset, such as property type, room type, bed type, cancellation policy, city, cleaning fee, host profile picture availability, host identity verification, and instant bookability. Our goal is to understand how these features influence renting prices and to build a predictive model to assist renters and landlords alike.

## Method

### Data Exploration

1. The boxplot reveals the presence of outliers, makes the median a more reliable measure than the mean for imputing missing values.
   1. ![Figure 1.](graphs/DataExploration1.png)

2. The histogram of the distribution of Airbnb price shows the data is a little left-skewed. 
   1. ![Figure 2.](graphs/DataExploration3.png)

3. The correlation matrix is calculated below.
   1. ![Figure 3.](graphs/DataExploration4.png)

***More investigations of other attributes can be found in the [Data PreProcessing Notebook](Data_preprocessing.ipynb), which are not shown due to the limit of space.***

### Data Preprocessing

1. Data Cleaning: 
   1. We drop columns with excessive unique categories or missing data that would be impractical to one-hot encode or impute, such as 'thumbnail_url', 'zipcode', 'neighbourhood', 'first_review', and 'last_review'.
   2. We convert the id column to a numeric type and rename it to 'id' for clarity and ease of reference.

2. Handling Missing Values: 
   1. A boxplot is drawn for the variables 'bathrooms', 'bedrooms', and 'beds' to visualize the distribution and identify outliers. Based on the presence of outliers, median values are chosen as a more reliable measure for imputing missing values. Missing values in 'bathrooms', 'bedrooms', and 'beds' are imputed with the median of each column, grouped by the 'accommodates' category, to maintain the integrity of the data.
   2. Missing values in 'host_response_rate' and 'review_scores_rating' are imputed with their respective median values. Before imputation, the 'host_response_rate' is converted from a percentage string to a float.
   3. We drop latitude and longitude column because it is difficult to process this kind of data and we already have City column.

3. Bag of Words (BOW) & Term Frequency-Inverse Document Frequency(TF-IDF)
   1. To utilize the 'description' and 'name' feature in training, we first need to transform each description into a vector and then discover the relationship between the descriptions and the log price. We use BOW and TF-IDF techniques, respectively, during the transformation process. We initially train the transformed vectors with the log price using a linear regression model, so that the model's theta learns the potential relationship between the description vector and the log price. We extract the theta of the LR model based on the words in each description. Then, we sum up the values of the corresponding theta and append the result as a new feature to our final training set.

4. Sentiment
   1.  It calculates the sentiment scores for the description and the name of a datapoint by summing up the sentiment scores of each word in the cleaned 'description' and 'name' respectively. If a word is not found in the sentiment_dict, its sentiment score is considered as 0.
         ```python
         punctuation = set(string.punctuation)

         def sentiment(d):
            sentimentScore = 0
            r = ''.join([c for c in d.lower() if not c in punctuation])
            for w in r.split():
               sentimentScore += sentiment_dict.get(w, 0)
            return sentimentScore
         def name(d):
            sentimentScore = 0
            r = ''.join([c for c in d.lower() if not c in punctuation])
            for w in r.split():
               sentimentScore += name_dict.get(w, 0)
            return sentimentScore
         ```

5. Fix perfect multicollinearity
   1. Perfect multicollinearity happens when one variable can be perfectly predicted from the others, causing issues in regression models by inflating the variance of the coefficient estimates, which can lead to a very large MSE. By setting drop_first=True, the function will drop the first level for each categorical variable. This effectively removes one dummy variable from each set of dummies derived from a categorical variable, thus eliminating the perfect multicollinearity that occurs when all dummy variables for a category are included.
      ```python
      df_encoded = pd.get_dummies(df, columns=['cleaning_fee','host_has_profile_pic', 'host_identity_verified', 'instant_bookable'], drop_first=True)
      ```

6. Encoding
   1. One-Hot Encoding:
      1. Categorical variables such as 'property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city', 'cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', and 'instant_bookable' were initially one-hot encoded. This process transforms categorical variables into a format that can be used in machine learning algorithms, creating separate binary columns for each category.
   2. KFold Target encoding:
      1. Due to the high dimensionality encountered with one-hot encoding, we implemented K-Fold target encoding to mitigate the issue. Target encoding is especially beneficial for neural network models. Where categorical features are replaced with the mean value of a target variable ('log_price') computed from each fold of the training data, to prevent data leakage.
   3. Leave One Out (LOO):
      1. We tried LOO after target encoding, but obtained a MSE of 0.003 with our final model, which was dramastically lower than the prior MSE. We attemptted to find the reason that caused the reduction of the MSE but failed. Therefore, we decided not to ultilize this encoding technique until we find the reason.
      2. Leave-One-Out (LOO) encoding is a form of target encoding that reduces overfitting by excluding the target value of the current row when calculating the category's mean target, thereby offering a more generalizable feature representation.
   
7. Norm & Standard
   1. Both normalization and standardization have their own advantages, and we first consider to use min-max because it makes us easy to interpret the data. However, We finally choose to use standardization for our project because we consider that it is less sensitivec to those outliers compared with min-max normalization. After the exploration of the data, we find that there are some outliers in the dataset which may affect the result greatly if we do not use standardization.


### Model 1: 2nd degree Polynomial Regression
First model: 2nd degree Polynomial Regression

### Model 2: Convolutional Neural Network

We employ hyperparameter tuning to optimize the configuration of our neural network model, leveraging the Keras Tuner for this purpose. The build_hp_model function dynamically constructs the model based on a range of hyperparameters, allowing for an exploration of different model architectures. It sets up a sequential model with a variable number of dense layers, each configured with a unit count ranging between 16 and 96 and using the 'leaky_relu' activation function to mitigate the vanishing gradient problem. The final layer, designed for regression, has a single unit. The learning rate for the Adam optimizer is also varied across a logarithmic scale from 1e-4 to 1e-2, enabling a comprehensive search across different magnitudes of learning rates.

Two Keras callbacks are utilized to enhance the training process:

Early Stopping: Monitors the validation loss and stops training if there hasn't been a significant decrease (less than 0.001) in the validation loss for 5 consecutive epochs. This prevents overfitting and ensures the model restores the weights from the epoch with the best performance.

Model Checkpoint: Saves the model at the filepath 'checkpoints' whenever a lower validation loss is observed. This ensures that the model configuration with the best validation performance is preserved, even if the model's performance degrades in subsequent epochs.

### Model 3: XGBoost
We performed K-fold cross-validation on our dataset using the XGBoost algorithm to predict Airbnb listing prices. This approach helps us to understand how well our model generalizes on unseen data by dividing the dataset into k distinct subsets (or folds), then iteratively training the model on k-1 subsets while using the remaining subset for validation. The process is repeated k times, with each subset serving as the validation set exactly once.

Cross-Validation Parameters
- Objective: Regression with squared error.
- Max Depth: 5 layers to control the complexity of the model.
- Eta (Learning Rate): 0.3 to control the model's learning rate.
- Evaluation Metric: Root Mean Squared Error (RMSE), a standard metric for regression tasks.


## Result
 
## Discussion

 
## Conclusion

## Collaboration
