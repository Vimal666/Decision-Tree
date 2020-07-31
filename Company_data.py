# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 22:01:48 2020

@author: Vimal PM
"""
#importing the neccesary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn import preprocessing
#Loading the dataset using pd.read_csv()
company_data=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//Work done//Decison tree//Company_Data.csv")

#loading the label encoder to convert the categorical variables to numerical
Le=preprocessing.LabelEncoder()
company_data.columns        
#Index(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
      # 'ShelveLoc', 'Age', 'Education', 'Urban', 'US']
#shelveLoc,Urban,US, are the categoiacal variables inside my dataset
company_data["Shelveloc"]=Le.fit_transform(company_data["ShelveLoc"])
company_data["urban"]=Le.fit_transform(company_data["Urban"])
company_data["Us"]=Le.fit_transform(company_data["US"])
#removing the categorical variables from my dataset
company_data=company_data.drop("ShelveLoc",axis=1)
company_data=company_data.drop("Urban",axis=1)
company_data=company_data.drop("US",axis=1)
#creating a bin lisy and converting my sales data's to categorical format
bins=[-1,6,12,18]
#creating a new variable called "sales_status" and converting them into categorical based on my sales variables
company_data["sales_status"]=pd.cut(company_data["Sales"],bins,labels=["low","medium","high"])
#checking the missing values
company_data.isna().sum()
#Sales           0
#CompPrice       0
#Income          0
#Advertising     0
#Population      0
#Price           0
#Age             0
#Education       0
#Shelveloc       0
#urban           0
#Us              0
#sales_status    0
################no null values found#############
predictors=company_data.iloc[:,1:11]
#next I would like to normalize the dataset to make data unitless and scalefree
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
#applying the normalization to my dataset
predictors_norm=norm_fun(predictors.iloc[:,0:7])    
target=pd.DataFrame(company_data.iloc[:, 11])
predictors_norm.describe()
       # CompPrice      Income  Advertising  ...       Price         Age   Education
#count  400.000000  400.000000   400.000000  ...  400.000000  400.000000  400.000000
#mean     0.489541    0.481389     0.228793  ...    0.549671    0.514955    0.487500
#std      0.156475    0.282687     0.229323  ...    0.141776    0.294551    0.327566
#min      0.000000    0.000000     0.000000  ...    0.000000    0.000000    0.000000
#25%      0.387755    0.219697     0.000000  ...    0.455090    0.268182    0.250000
#50%      0.489796    0.484848     0.172414  ...    0.556886    0.536364    0.500000
#75%      0.591837    0.707071     0.413793  ...    0.640719    0.745455    0.750000
#max      1.000000    1.000000     1.000000  ...    1.000000    1.000000    1.000000
predictors_train,predictors_test,target_train,target_test = train_test_split(predictors_norm,target,test_size=0.2, random_state=0)
#Building the model
model=DecisionTreeClassifier(criterion="entropy")
model.fit(predictors_train,target_train)
pred=model.predict(predictors_test)
pd.crosstab(target_test.sales_status,pred)
pd.Series(pred).value_counts()
#medium    45
#low       33
#high       2
np.mean(pred==target_test.sales_status) #0.5625(accuracy)

