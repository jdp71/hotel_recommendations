## Introduction  
This project is based on a recent Kaggle competition where participants were asked to help Expedia predict where their users will book their next hotel stay. Currently, Expedia uses search parameters to adjust their hotel recommendations, but there aren't enough customer specific data to personalize them for each user. In this project, I will contextualize customer data and predict the likelihood a user will stay at 100 different hotel groups.  
**NOTE:  Many of these cells take a long time to run.  For example, the Logistic Regression portion takes over 60 minutes.  Be patient.**

## Data Sources Used  
The data for this project can be found on the Kaggle website at this link [Expedia Hotel Prediction Data]( https://www.kaggle.com/c/expedia-hotel-recommendations/data).  

## Technologies Used  
* Python 3+  
* Jupyter Notebook 5.7.8  
* R Studio 1.1.447  

## Required Packages  
**Python Packages**
```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
```  
**R Libraries**
```R
library(ggplot2)
library(reshape2)
library(scales)
library(reticulate)
library(ggthemes)
```  

## Analysis Methods Used  
* Exploratory Data Analysis (EDA)  
* Using Python and R within the same Jupyter Notebook  
* Graphic analysis  
* Correlation analysis  
* K-Nearest Neighbor (KNN)  
* Logistic regression  
* Random Forest Classifier  
* Naive Bayes Classifier  

## Model Deployment  
Four models were used to see which one was the most accurate in predicting the correct hotel cluster.  
1. **KNN**:  This is a non-parametric model, so the thought was that it would do well for our non-normally distributed data.  The intent was to teach the model to predict the correct hotel cluster based on the number of nearest neighbors.  
2. **Logistic regression**:  The correlation matrix suggested that a linear prediction model might not be the best choice for this dataset.  While logistic regression is considered a linear method, it performs better than a linear model when the dependent variable is categorical.  In our case, the category is essentially either a yes or a no as to whether a specific hotel will belong to a certain cluster.  
3. **Random Forest Classifier**:  This is also a non-parametric classification method and therefore, the accuracy value should be close to what we saw with the KNN model.  Because this type of method is prone to overfitting, the code also included cross validation.  
4. **Naive Bayes Classifier**:  This model was chosen because it is usually quick to compute and is fundamentally different from the previous three models.  

## Summary of Results  
The table below provides an accuracy summary for each model.  
| Model | Accuracy Score |  
| ----- | :-----: |  
| KNN | 25.6% |  
| Logistic Regression | 30.4% |  
| Random Forest | 25.0% |  
| Naive Bayes | 10.3% |  

The logistic regression model was the most accurate.  
The KNN and Random Forest algorithms performed about the same.  
The Naive Bayes algorithm was the least accurate.  
