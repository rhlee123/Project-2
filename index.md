# Project 2: Regularization Techniques
## Introduction
  In this project, I explored the use of five different regularization techniques: LASSO, Ridge, Elastic Net, SCAD (Smoothly-Clipped Absolute Deviation), and Square Root Lasso. For 3 different datasets, I fit multivariate linear models using the aforementioned regularization techniques and compared the MAE (Mean Absolute Error) of the predictions of these models to the MAE of a baseline multiple linear regression. Comparing the values of the loss function MAE between the different regularization methods and a baseline model allow the ability to draw conclusions about the effectiveness of certain regularization methods on a particular dataset. The regularized model that results in the lowest MAE would be the model that is most effective in minimizing error values. 
  
  The datasets explored in this project are: Boston Housing Price, Sale Price of Car, and a simulated dataset.
### Background 
  Regularization methods are used to determine the weights for features within a model, and depending on the regularization technique, features can be excluded altogether from the model by having a weight of 0. Further, regularization is the process of of regularizing the parameters that constrains or coefficients estimates towards zeros, in otherwords discouraging learning a too complex or too flexible model, and ultimately helping to reduce the risk of overfitting. Regularization helps to choose the preferred model complexity, so that the model is better at predicting. Regularization is essentially adding a penalty term to the objective function, and controlling the model complexity using that penalty term. Regularization ultmately attempts to reduce the variance of the estimator by simplifying it, something that will increase the bias, in such a way that will decrease the expected error of the model's predictions. Additionally, Regularization is useful in tackling issues of multicolinearity among features becauase it incorporates the need to learn additional information for our model other than the observations in the data set.
  Regularization techniques are especially useful in situations where there are large number of features or a low number of observations to number of features, where there is multicolinearity between the features, trying to seek a sparse solution, or accounting for for variable groupings in high dimensions. Regularization ultimately seeks to tackle the shortcomings of Ordinary Least Square models. 
### Ridge 
L2 regularization, or commonly known as ridge regularization, is a type of regularization that controls for the sizes of each coefficient or estimators. Not only does ridge regularization encorporate principles of OLS by reducing the sum of squared residuals, it also penalizes models with the regularization term of L2 Norm or
![\alpha \sum_{i=1}^p \beta_i^2](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%5Cbeta_i%5E2).

Ridge regularization is especially useful when there is multicolinearity within data, and further, ridge regularization seeks to ultimately minimze the cost function:

![\sum_{i=1}^N (y_i - \hat{y}_i)^2 + \alpha \sum_{i=1}^p |\beta_i| 
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Csum_%7Bi%3D1%7D%5EN+%28y_i+-+%5Chat%7By%7D_i%29%5E2+%2B+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%7C%5Cbeta_i%7C+%0A)

(α) in this instance is the hyperparameter that determines the strength of regularization, or the strength of the penalty on the model. 
### LASSO
L1 regularization, or commonly known as Least Absolute Shrinkage and Selection Operator (LASSO) regularization, determines the weight of features by penalizing the model with the regularization term of L1 Norm or ![\alpha \sum_{i=1}^p |\beta_i| ](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%7C%5Cbeta_i%7C+). 

Further, LASSO regularization seeks to ultimately minimze the cost function: 

![\sum_{i=1}^N (y_i - \hat{y}_i)^2 + \alpha \sum_{i=1}^p |\beta_i| ](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Csum_%7Bi%3D1%7D%5EN+%28y_i+-+%5Chat%7By%7D_i%29%5E2+%2B+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%7C%5Cbeta_i%7C+)

LASSO differs from ridge in that the way that LASSO penalizes high coefficients, as instead of squaring coefficients, LASSO takes the absolute value of the coefficients. Ultimately the weights of features can go to 0 using L1 norm, as opposed to L2 norm that ridge regularization in which weights can not go to 0. Ridge regularization will shrink the coefficients for least important features, very close to zero, however, will never make them exactly zero, resulting in the final model including all predictors. However, in the case of the LASSO, the L1 norm penalty has the eﬀect of forcing some of the coeﬃcient estimates to be exactly equal to zero when the tuning parameter (α) is suﬃciently large. Therefore, the lasso method, not only performs variable selection but is generally said to yield sparse models.
### Elastic Net
Elastic net regularization combines aspects of both ridge and LASSO regularization by including both L1 norm and L2 norm penalties. Elastic net determines the weights of features by minimizing the cost funciton: 

![\hat{\beta} = argmin_\beta \left\Vert  y-X\beta \right\Vert ^2 + \lambda_2\left\Vert  \beta \right\Vert ^2 + \lambda_1\left\Vert  \beta\right\Vert_1
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Chat%7B%5Cbeta%7D+%3D+argmin_%5Cbeta+%5Cleft%5CVert++y-X%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_2%5Cleft%5CVert++%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_1%5Cleft%5CVert++%5Cbeta%5Cright%5CVert_1%0A) 

Elastic net regularization is a good middle ground between the other techniques, ridge and lasso, because the model can learns weights that fit the multicolinearity and sparsity pattern within the data. 
 

