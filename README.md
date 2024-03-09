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

1. Second Model: Neural Networks (NN) with Hyper Tuning

   The model development process involved initially building a neural network (NN) architecture with predetermined hyperparameters, yielding a Mean Squared Error (MSE) of 0.1935 on the test set. Subsequently, hyperparameter tuning was employed to optimize the model's performance, resulting in a reduction of the validation loss to 0.1883, indicating an improvement in predictive accuracy. Upon applying the best hyperparameters identified through tuning, the model achieved a lower MSE of 0.1905 on the test set. This iterative process demonstrates the effectiveness of hyperparameter tuning in fine-tuning the neural network's architecture and configuration to enhance its predictive capabilities. The model's performance, with a validation loss as low as 0.185, underscores its potential for accurately predicting Airbnb prices. However, it's essential to note that further experimentation and refinement could potentially yield even better results. 

   To further improve the model's performance in predicting Airbnb prices, several strategies can be considered:

   1. Explore More Complex Architectures

   2. Regularization Techniques

   3. Advanced Activation Functions

   Our neural network (NN) model, which achieved a Mean Squared Error (MSE) of approximately 0.1905, performed slightly worse in terms of prediction accuracy compared to the first model utilizing second-degree polynomial regression. The polynomial regression model had an MSE of approximately 0.1868, indicating a lower level of error in its predictions. However, it's important to consider that neural networks are generally more flexible and capable of capturing complex relationships in the data compared to polynomial regression models.

   