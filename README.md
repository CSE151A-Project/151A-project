# 151A-project
This project aims to predict the renting prices in different cities using various features from the dataset, such as property type, room type, bed type, cancellation policy, city, cleaning fee, host profile picture availability, host identity verification, and instant bookability. Our goal is to understand how these features influence renting prices and to build a predictive model to assist renters and landlords alike.

## Dataset Description
The dataset comprises listings with features that describe the property, host, and booking policies. Here are some key details:
Number of Observations: xxxxxxxxxxxxxxxxx, which describe the total number of listings included.
Target Variable: The target variable is the renting price, which we aim to predict.
Features: The dataset includes binary features:
property_type
room_type
bed_type
cancellation_policy
city
cleaning_fee
host_has_profile_pic
host_identity_verified
instant_bookable
These features have been encoded as binary (0 or 1) to indicate the presence or absence of a particular attribute.

## Data Exploration
During the initial exploration, we analyzed the distributions, scales, and presence of missing data in our dataset. We find that there are great variances in some of the data, so we pick the median number.
### Standardize
These feature might need to be standardize, they have different unit with our target variavle, renting price.

amenities, property_type, room_type, bed_type, cancellation_policy, and cleaning_fee.

amenities is a list of category, Each observation will have a list of amenities.

property_type: category. Apartment', 'House', 'Condominium', 'Loft', etc.

room_type: category. Entire home/apt', 'Private room', 'Shared room

bed_type: category. Real Bed', 'Futon', 'Pull-out Sofa', 'Couch', 'Airbed'

cancellation_policy: category. 'strict', 'moderate', 'flexible', 'super_strict_30', 'super_strict_60'

cleaning_fee: boolean. 


## Plots and Visualizations
Scatter plots, histograms, and box plots were used to visualize the data. ???

## Data Preprocessing
To prepare the data for modeling, we performed the following preprocessing steps:

Min-Max Normalization: Applied to numerical features to bring them to a scale between 0 and 1, improving model performance and stability.
Handling Missing Data: Missing values were addressed through imputation or removal, depending on the feature and the amount of missing data.
