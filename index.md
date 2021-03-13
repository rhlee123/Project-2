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
Elastic net regularization combines aspects of both ridge and LASSO regularization by including both L1 norm and L2 norm penalties. Elastic net determines the weights of features by minimizing the cost funciton where λ between 0 and 1: 

![\hat{\beta} = argmin_\beta \left\Vert  y-X\beta \right\Vert ^2 + \lambda_2\left\Vert  \beta \right\Vert ^2 + \lambda_1\left\Vert  \beta\right\Vert_1
![jkiyutyfcgvhbkjnkihyguh](https://user-images.githubusercontent.com/55299814/111017541-3d28c000-8382-11eb-9681-03df13e00f9f.png)
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Chat%7B%5Cbeta%7D+%3D+argmin_%5Cbeta+%5Cleft%5CVert++y-X%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_2%5Cleft%5CVert++%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_1%5Cleft%5CVert++%5Cbeta%5Cright%5CVert_1%0A) 

Elastic net regularization is a good middle ground between the other techniques, ridge and LASSO, because the technique allow for the model to learns weights that fit the multicolinearity and sparsity pattern within the data. 
### SCAD
The Smoothly Clipped Absolute Deviation regularization attempts to address issues of multicolinearity and encourage sparse solutions to ordinary least squares, while at the same time allowing for large (β) Values. 

The SCAD penalty is genearlly defined by it first derivative:

![p'_\lambda(\beta) = \lambda \left\{ I(\beta \leq \lambda) + \frac{(a\lambda - \beta)_+}{(a - 1) \lambda} I(\beta > \lambda) \right\}
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+p%27_%5Clambda%28%5Cbeta%29+%3D+%5Clambda+%5Cleft%5C%7B+I%28%5Cbeta+%5Cleq+%5Clambda%29+%2B+%5Cfrac%7B%28a%5Clambda+-+%5Cbeta%29_%2B%7D%7B%28a+-+1%29+%5Clambda%7D+I%28%5Cbeta+%3E+%5Clambda%29+%5Cright%5C%7D%0A)

with the penalty function represented by the piecewise function: 

![\begin{cases} \lambda & \text{if } |\beta| \leq \lambda \\ \frac{(a\lambda - \beta)}{(a - 1) } & \text{if } \lambda < |\beta| \leq a \lambda \\ 0 & \text{if } |\beta| > a \lambda \\ \end{cases}
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Bcases%7D+%5Clambda+%26+%5Ctext%7Bif+%7D+%7C%5Cbeta%7C+%5Cleq+%5Clambda+%5C%5C+%5Cfrac%7B%28a%5Clambda+-+%5Cbeta%29%7D%7B%28a+-+1%29+%7D+%26+%5Ctext%7Bif+%7D+%5Clambda+%3C+%7C%5Cbeta%7C+%5Cleq+a+%5Clambda+%5C%5C+0+%26+%5Ctext%7Bif+%7D+%7C%5Cbeta%7C+%3E+a+%5Clambda+%5C%5C+%5Cend%7Bcases%7D%0A)

The cost function ultimately looks like: 

![jkiyutyfcgvhbkjnkihyguh](https://user-images.githubusercontent.com/55299814/111017670-d7890380-8382-11eb-84e0-7e6908fb891a.png)

### Square Root LASSO: 

Square Root LASSO slightly adjusts the LASSO method, in which it takes the square root of the LASSO cost function. It is important to note that L1 norm is still used for its penalty. Ultimately, lasso weights features by minimizing the cost function: 

![\sqrt{\frac{1}{n}\sum\lim_{i=1}^{n}(y_i-\hat{y}_i)^2} +\alpha\sum\lim_{i=1}^{p}|\beta_i|
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum%5Clim_%7Bi%3D1%7D%5E%7Bn%7D%28y_i-%5Chat%7By%7D_i%29%5E2%7D+%2B%5Calpha%5Csum%5Clim_%7Bi%3D1%7D%5E%7Bp%7D%7C%5Cbeta_i%7C%0A)

# Applying on Different Data Sets
All data used in comparing the regularization techniques are standardized, and will be evaluated based on Mean Absolute Error. 

(α) values cannot be too large or too small due to the various penalties of the different regularization techniques. Optimal (α) values were found by observing the (α) value that produces the lowest MAE for the regularized model.

## Code for finding the MAE of different regularized linear models and implementing different reguralization techniques in python for boston housing dataset: 
Basic imports:
```python ![jkiyutyfcgvhbkjnkihyguh](https://user-images.githubusercontent.com/55299814/111017606-90027780-8382-11eb-8f78-034ea50ce823.png)

![jkiyutyfcgvhbkjnkihyguh](https://user-images.githubusercontent.com/55299814/111017611-9bee3980-8382-11eb-84d6-7deb79defcb4.png)
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from sklearn.model_selection import KFold
```
Initializing data:
```python
df_boston = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/Boston Housing Prices(1) (1).csv')
df_boston
features = ['crime','rooms','residential','industrial','nox','older','distance','highway','tax','ptratio','lstat']
X = np.array(df_boston[features])
y = np.array(df_boston['cmedv']).reshape(-1,1)
datb = np.concatenate([X,y], axis=1)
datb_train, datb_test = tts(datb, test_size=0.3, random_state=1234)
ss=SS()
```
Ridge: 
```python
lr.fit(ss.fit_transform(datb_train[:,:-1]),datb_train[:,-1])
yhat_lr = lr.predict(ss.transform(datb_test[:,:-1]))
mae_lr = mean_absolute_error(datb_test[:,-1], yhat_lr)
print("Standardized MAE Ridge Model = ${:,.2f}".format(1000*mae_lr))
```
LASSO:
```python
ls = Lasso(alpha=0.16)
ls.fit(ss.fit_transform(datb_train[:,:-1]),datb_train[:,-1])
yhat_ls = ls.predict(ss.transform(datb_test[:,:-1]))
mae_ls = mean_absolute_error(datb_test[:,-1], yhat_ls)
print("MAE Lasso (standardized) Model = ${:,.2f}".format(1000*mae_ls))
```
SCAD:
Intializing SCAD: 
```python
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
def scad(beta):
  beta = beta.flatten()
  beta = beta.reshape(-1,1)
  n = len(y)
  return 1/n*np.sum((y-X.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))
  
def dscad(beta):
  beta = beta.flatten()
  beta = beta.reshape(-1,1)
  n = len(y)
  return np.array(-2/n*np.transpose(X).dot(y-X.dot(beta))+scad_derivative(beta,lam,a)).flatten()
```
Applying SCAD:
```python 
p = X.shape[1]
b0 = np.random.normal(1,1,p)
lam = 1
a = 2
output = minimize(scad, b0, method = 'L-BFGS-B', jac=dscad,options={'gtol':1e-8, 'maxiter': 50000, 'maxls': 25, 'disp':True})
output.x
yhat_test_scad = datb_test[:,:-1].dot(output.x)
mae_scad = mean_absolute_error(datb_test[:,-1],yhat_test_scad)
print("MAE Scad Model = ${:,.2f}".format(1000*mae_scad))
```
Square Root Lasso: 
```python 
slasso = sm.OLS(y,X)
slasso = slasso.fit_regularized(method='sqrt_lasso', alpha=.01)
slasso.params
yhat_sl = slasso.predict(datb_test[:,:-1])
mae_sl = mean_absolute_error(datb_test[:,-1], yhat_sl)
print("MAE Square Root Lasso Regression Model = ${:,.2f}".format(1000*mae_sl))
```
## Boston Housing Data 
To take a look at the boston housing data set, here is a heatmap showing the correlations between features in the dataset in which we will be using to predict housing prices: 

![project1](https://user-images.githubusercontent.com/55299814/111015985-6133d380-8379-11eb-9c51-925a01550166.png)

Clearly, there is strong correlation between the features, which indicates that regularization might be helpful effectively estimating coefficients despite the multicolinearity present. Further, it indicates that regularization might reduce the MAE of a model that looks to predict price using these features. 

Below are the MAE of multivariate linear regression models predicing boston housing prices regularized using ridge, LASSO, Elastic Net, SCAD, Square Root LASSO compared to a baseline linear model. The optimal (α) value found are also listed.

| Model                          | MAE (Standardized)| Optimal α| 
|--------------------------------|-------------------|----------|
| MAE Linear Model               | $3,629.76         |          |                                
| MAE Ridge Regression Model     | $3,443.23         | 45       |             
| MAE LASSO Model                | $3,499.16         | .16      |             
| MAE Elastic Net Model          | $3,445.53         | .15      | 
| MAE SCAD Model                 | $3,270.01         | 2        |            
| MAE Square Root LASSO          | $3,257.50         | .01      |            

It seems that all of the regularization methods outperformed the baseline linear model. All the regularization techniques performed better at predicting coefficients that would result in predictions that produce lower MAE's because the regularization techniques were configured to take into account data with multicolinearity present. Square Root LASSO regularization with a MAE of $3,257.50 performed the best at minimizing MAE when applied to a multivariate linear model, especially because the Square Root LASSO accounts for multicolinearity and penalizes in a way that produces sparse solutions. 

## Car Price Data Set
To take a look at the car price housing data set, here is a heatmap showing the correlations between features in the dataset in which we will be using to predict car prices: 

![hbjjknhgvhvjbkj](https://user-images.githubusercontent.com/55299814/111017691-f7b8c280-8382-11eb-8060-7c8acb872cd8.png)

It looks like there is slightly less correlation between the features in this dataset as compared to the boston housing dataset. There is still some multicolinearity, so regularization appears like it might still be helpful in reducing the MAE of a model that looks to predice price of a car using the features in the heatmap above. 

I repeated the same process for the boston housing dataset on this car price data set to find the MAE of multivariate linear regression models predicting car prices regularized using LASSO, Elastic Net, SCAD, Square Root LASSO compared to a baseline linear model. The results are listed below with the corresponding optimal (α) value listed.

| Model                          | MAE (Standardized)| Optimal α| 
|--------------------------------|-------------------|----------|
| MAE Linear Model               | $1,683.92         |          |                                
| MAE Ridge Regression Model     | $1,612.29         | 15.25    |             
| MAE LASSO Model                | $1,602.30         | 63       |             
| MAE Elastic Net Model          | $1,612.52         | .14      | 
| MAE SCAD Model                 | $1,696.68         | 2        |            
| MAE Square Root LASSO          | $1,619.19         | 3        |   

It seems that all the regularization methods outperformed the baseline linear model except for the SCAD model which marginally underperformed. This shows that regularization models must be picked according to the data, as some regularized models can actually make worse predictions and result in a higher MAE than the baseline model. However, excluding SCAD, it is apparent that regularization techniques that take into consideration the possible multicolinearity of data and/or encourages sparse solutions can be helpful in reducing the MAE of model that tries to predict using features that show slightly less correlation among themselves compared to the boston housing dataset. Regularization can be seen to clearly help models make better predictions. For models predicting car price, the LASSO regularized model performed the best at making predictions because the predictions yielded the lowest MAE at $1602.30. The Ridge and Elastic Net regularized models also produced competitived predictions with MAE's of $1,612.29 and $1,612.52 respectively. 

Lets Take a closer look at how different alpha values affect the MAE of the three top performing models predicting car price mentioned above.

LASSO:
![image](https://user-images.githubusercontent.com/55299814/111018356-57fd3380-8386-11eb-9e14-d76b5e20d45a.png)

Ridge: 
![image](https://user-images.githubusercontent.com/55299814/111018372-7105e480-8386-11eb-8f4c-5716c4df2c03.png)

Elastic Net:
![image](https://user-images.githubusercontent.com/55299814/111018410-a3174680-8386-11eb-803a-cdc78669dd96.png)

The combination of LASSO and Ridge in Elastic Net by the use of L1 and L2 norm can be evidently seen in the above graphs.

Ultimately, this dataset only has 48 rows and 15 features. The lower MAE values from regularization show that regularization can help prevent overfitting which would have otherwise been a huge risk when using this dataset for prediction because of the low number of observations and relatively high number of features. 

## Synthetic Data

Here I initialize synthetic data:
```python
def make_correlated_features(num_samples,p,rho):
  vcor = [] 
  for i in range(p):
    vcor.append(rho**i)
  r = toeplitz(vcor)
  mu = np.repeat(0,p)
  X = np.random.multivariate_normal(mu, r, size=num_samples)
  return X
```
```python
n = 200
p = 50
X = make_correlated_features(200,p,0.8)
beta = np.array([-2,0,0,0,1,1,1,5,2,-3])
beta = beta.reshape(-1,1)
betas = np.concatenate([beta,np.repeat(0,p-len(beta)).reshape(-1,1)],axis=0)
n = 200
sigma = 2
y = X.dot(betas) + sigma*np.random.normal(0,1,n).reshape(-1,1)
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3,random_state=1234)
```
I performed similar evaluations of the different regularization techniques as the other two datasets for my synthetic data. Below I show the output

| Model                          | MAE (Standardized)| Optimal α| 
|--------------------------------|-------------------|----------|
| MAE Linear Model               | 1.94              |          |                                
| MAE Ridge Regression Model     | 1.90              | 2        |             
| MAE LASSO Model                | 1.72              | .075     |             
| MAE Elastic Net Model          | 1.73              | .1       | 
| MAE SCAD Model                 | 1.85              | 2        |            
| MAE Square Root LASSO          | 1.43              | 2.       |  

Above shows another instance where all the regularization techniques produced predictions that effectively outperformed the baseline linear model. In this instance, the Square Root LASSO regularized multivariate linear model performed the best as its predictions resulted in the lowest MAE of 1.43. This instance shows an example of a situation where Square Root LASSO regularization is much more effective in reducing MAE when compared to normal LASSO regularization. This might indicate that Square Root LASSO might be more advantageous than normal LASSO in this situation because Square Root LASSO's data-driven optimal (α) is independent of the unkown error variance under homoskedasticity. It should be noted that the normal LASSO that takes into multicolinearity was actually also very effective at reducing the MAE of the multivariate linear model and actually had the second lowest MAE of 1.72. 

# Conclusion
  It can be seen that regularization can effectively improve model predictions, evident by the regularization techniques effectively producing predictions that result in lower MAE's when compared to a baseline multivariate linear model. It can be seen as well through analysis of the above datasets that regularization can be extremely helpful when multicolinearity is present in the data, and can be extremely effective at producing sparse solutions. The regularization techniques showed that they were also effective at helping models reduce the risk of overfitting, as shown by the consistently lower MAEs for the cars dataset that had a lower number of observations and high number of features. It should be noted that regularization will not always result in better predictions, and can sometimes even result in worse predictions when compared to a baseline linear model. This was seen in the SCAD regularize model performing worse than the baseline linear model in the car price dataset, indicating that the selection of the penalty is important for regularization. Additionaly, the selection of an optimal (α) is extremely important as some (α) values can produce models with higher MAE. There should be especially careful thought in the selection of hyperparameters and penalty as regularization do not always produce better predictions.
  Ultimately, regularization can be extremely powerful in enhancing the accuracy of predictions for multivariate linear models and reducing MAE. Regularization can make models more effective because they can take into consideration multicolinearity within the data, produces sparse solution, and reduce overfitting. 

