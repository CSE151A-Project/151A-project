## [Milestone 4](Milestone4.ipynb)


### More data preprocessing

1. KFold Target encoding

   Implementing K-Fold target encoding to mitigate the high dimensionality issue often encountered with one-hot encoding, especially beneficial for neural network models. Where categorical features are replaced with the mean value of a target variable ('log_price') computed from each fold of the training data, to prevent data leakage.

### Hyper parameter tuning
-activation, random search, early stop, etc

In our project, we employ hyperparameter tuning to optimize the configuration of our neural network model, leveraging the Keras Tuner for this purpose. The build_hp_model function dynamically constructs the model based on a range of hyperparameters, allowing for an exploration of different model architectures. It sets up a sequential model with a variable number of dense layers, each configured with a unit count ranging between 16 and 96 and using the 'leaky_relu' activation function to mitigate the vanishing gradient problem. The final layer, designed for regression, has a single unit. The learning rate for the Adam optimizer is also varied across a logarithmic scale from 1e-4 to 1e-2, enabling a comprehensive search across different magnitudes of learning rates.

Two Keras callbacks are utilized to enhance the training process:

Early Stopping: Monitors the validation loss and stops training if there hasn't been a significant decrease (less than 0.001) in the validation loss for 5 consecutive epochs. This prevents overfitting and ensures the model restores the weights from the epoch with the best performance.
Model Checkpoint: Saves the model at the filepath 'checkpoints' whenever a lower validation loss is observed. This ensures that the model configuration with the best validation performance is preserved, even if the model's performance degrades in subsequent epochs.


### Plots and Visualizations
#### Original Neural Network Loss: 
The mse of the orginal neural network is 0.1917, which is slightly lower than the mse of our previous polynomial model. 
The following plot shows the the change of loss as training epochs increase. The loss is decreasing as the model is trained with more epochs.
However, the plot also indicates a potential of overfitting. In order to improve the neural network, we did hyperparameter tuning and had the results shown in the second plot. 


![](graphs/Neural%20Network.png) 
#### Tuned Neural Network Loss: 
This following graph shows the relationship between loss and epochs for our tuned model. Although the plot still shows some overfitting, the loss of the tuned model is generally lower than the loss of the original model. The mse is 0.1904, indicating improvements in the tuned model. 
![](graphs/Best%20Model.png)

### Next Model
1.   XGBoost (Extreme Gradient Boosting):
XGBoost is a powerful and popular machine learning algorithm known for its performance in structured/tabular data and its ability to handle complex relationships within the data:
     - XGBoost is robust to overfitting and can handle a large number of features, making it suitable for this dataset with 29 attributes.
     - It can capture non-linear relationships between features and the target variable, which might not be effectively captured by linear or polynomial regression models.
     - XGBoost often performs well in competitions and real-world applications, making it a reliable choice for predictive modeling tasks.
     - It provides feature importance scores, which can help in understanding the relative importance of different features in predicting renting prices.

### Conclusion

1. Second Model: Neural Networks (NN) with Hyper Tuning

   The model development process involved initially building a neural network (NN) architecture with predetermined hyperparameters, yielding a Mean Squared Error (MSE) of 0.1917 on the test set. Subsequently, hyperparameter tuning was employed to optimize the model's performance, resulting in a reduction of the validation loss to 0.1877, indicating an improvement in predictive accuracy. Upon applying the best hyperparameters identified through tuning, the model achieved a lower MSE of 0.1905 on the test set. This iterative process demonstrates the effectiveness of hyperparameter tuning in fine-tuning the neural network's architecture and configuration to enhance its predictive capabilities. The model's performance, with a validation loss as low as 0.1877, underscores its potential for accurately predicting Airbnb prices. However, it's essential to note that further experimentation and refinement could potentially yield even better results. 

   To further improve the model's performance in predicting Airbnb prices, several strategies can be considered:

   1. Explore More Complex Architectures

   2. Regularization Techniques

   3. Advanced Activation Functions

   Our neural network (NN) model, which achieved a Mean Squared Error (MSE) of approximately 0.1905, performed slightly better in terms of prediction accuracy compared to the first model utilizing second-degree polynomial regression. The polynomial regression model had an MSE of approximately 0.2011, indicating a higher level of error in its predictions. That might because neural networks are generally more flexible and capable of capturing complex relationships in the data compared to polynomial regression models.

   
