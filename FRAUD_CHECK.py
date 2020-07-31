# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:49:25 2020

@author: Vimal PM
"""

#importing neccesary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
#loading the dataset using pd.read_csv()
Fraud=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\Work done\Decison tree\Fraud_check.csv")
Fraud.columns
#Index(['Undergrad', 'Marital.Status', 'Taxable.Income', 'City.Population',
       #'Work.Experience', 'Urban']
#checking the missing values       
Fraud.isna().sum()
#Undergrad          0
#Marital.Status     0
#Taxable.Income     0
#City.Population    0
#Work.Experience    0
#Urban              0

#converting categorical data's to numerical 
fraud=Fraud
Le=preprocessing.LabelEncoder()
Fraud['Undergrad']=Le.fit_transform(fraud.iloc[:, 0])
Fraud["Marital.Status"]=Le.fit_transform(fraud.iloc[:,1])
Fraud["Urban"]=Le.fit_transform(fraud.iloc[:,5])
#craeting a categories based on "Taxable.Income"
bins=[10000,30000,100000]
labels=["risky","good"]
Fraud["status"]=pd.cut(Fraud["Taxable.Income"],bins,labels=["risky","good"])
Frauds=Fraud.drop(['Taxable.Income'],axis=1)
#next I would like to normalize the dataset to make data unitless and scalefree
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
#applying normalizations to my dataset    
predictors=Frauds.iloc[:,0:5]
predictor_norm=norm_fun(predictors.iloc[:,:])
predictor_norm.describe()

        #Undergrad  Marital.Status  City.Population  Work.Experience       Urban
#count  600.000000      600.000000       600.000000       600.000000  600.000000
#mean     0.520000        0.523333         0.476832         0.518611    0.503333
#std      0.500017        0.410979         0.286496         0.294738    0.500406
#min      0.000000        0.000000         0.000000         0.000000    0.000000
#25%      0.000000        0.000000         0.236713         0.266667    0.000000
#50%      1.000000        0.500000         0.463879         0.500000    1.000000
#75%      1.000000        1.000000         0.714575         0.800000    1.000000
#max      1.000000        1.000000         1.000000         1.000000    1.000000
target=pd.DataFrame(Frauds.iloc[:,5])        
#gettine the train test data's
predictors_train,predictors_test,target_train,target_test=train_test_split(predictor_norm,target,test_size=0.2,random_state=0)
#building the model
model=DecisionTreeClassifier(criterion="entropy")
model.fit(predictors_train,target_train)
pred=model.predict(predictors_test)
pd.crosstab(target_test.status,pred)
#         good  risky
#status             
#risky     18      7
#good      73     22
pd.Series(pred).value_counts()
#good     91
#risky    29
#from the above analysis, I can say risky are very less(29) compared to good(91)
#getting the accuray
np.mean(pred==target_test.status)
#0.6666666666666666  (66%)
