## [Milestone 4](Milestone4.ipynb)


### More data preprocessing

1. KFold Target encoding



### Hyper parameter tuning
-activation, random search, early stop, etc


### Plots and Visualizations

improvement after tuning, overfitting?

1. 

### Next Model

1.   XGBoost (Extreme Gradient Boosting):
XGBoost is a powerful and popular machine learning algorithm known for its performance in structured/tabular data and its ability to handle complex relationships within the data:

- XGBoost is robust to overfitting and can handle a large number of features, making it suitable for this dataset with 29 attributes.
- It can capture non-linear relationships between features and the target variable, which might not be effectively captured by linear or polynomial regression models.
- XGBoost often performs well in competitions and real-world applications, making it a reliable choice for predictive modeling tasks.
- It provides feature importance scores, which can help in understanding the relative importance of different features in predicting renting prices.


### Conclusion

1. 

   By leveraging hyperparameter tuning techniques, we meticulously optimized our predictive model, achieving a validation mean squared error (MSE) of approximately 0.182, showcasing the effectiveness of the chosen neural network architectures. Subsequent training with XGBoost further refined our model, resulting in a training MSE of 0.159 and a testing MSE of 0.181, indicative of its robust performance on unseen data. Additionally, K-fold cross-validation confirmed the consistency of our model's performance, with a mean MSE of approximately 0.186 across all folds, underscoring its reliability and generalization capability. This comprehensive approach, combining hyperparameter tuning with ensemble techniques, has yielded a model with promising predictive capabilities and potential for real-world applications. 
   
   Further refinements could involve exploring advanced ensemble strategies or incorporating additional features to enhance predictive accuracy and robustness.