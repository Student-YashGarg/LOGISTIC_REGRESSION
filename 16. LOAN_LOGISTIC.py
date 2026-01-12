# LOAN APPROVAL PREDICTION MODEL
# BINARY CLASSIFICATION PROBLEM
# using logistic regression

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("E:\Desktop\ML\LOGISTIC_REGRESSION\loan_dataset.csv")
print(df)

#-------------DROP MISSING VALUES-------------------
# Before drop missing value
print(df.isnull().sum())
print(df.shape) # 614,13

# After drop missing value
df=df.dropna()
print(df.isnull().sum())
print(df.shape) # 480,13

#-------------UNDERSTAND LOAN APPROVAL VIA CATEGORY---------------
# plot bw education and loan
# ax=sns.countplot(x='Education',hue="Loan_Status",data=df)
# for container in ax.containers:
#     ax.bar_label(container)
#     plt.show()
# # plot bw dependents and loan
# ax=sns.countplot(x='Dependents',hue="Loan_Status",data=df)
# for container in ax.containers:  
#     ax.bar_label(container)
#     plt.show()
# # plot bw married and load
# ax=sns.countplot(x='Married',hue="Loan_Status",data=df)
# for container in ax.containers:
#     ax.bar_label(container)
#     plt.show()
# # plot bw Property_Area and loan
# ax=sns.countplot(x='Property_Area',hue="Loan_Status",data=df)
# for container in ax.containers:
#     ax.bar_label(container)
#     plt.show()

#------------CHECK CATEGORIAL VALUES---------------
print(df.Education.value_counts())
print(df.Married.value_counts())
print(df.Gender.value_counts())

# check dependent
print(df.Dependents.value_counts()) 
# Dependents
# 0     274
# 2      85
# 1      80
# 3+     41

# replace 3+ with 4
df.replace({'Dependents':{'3+':4}},inplace=True)
# df.replace(to_replace='3+',value=4,inplace=True) # Replaces '3+' everywhere in the DataFrame

print(df.Dependents.value_counts()) 
# Dependents
# 0     274
# 2      85
# 1      80
# 4     41

# repalce gender,married,loan_status and education ....with numaric values
df.replace({'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
df.replace({'Married':{'Yes':1,'No':0}},inplace=True)
df.replace({'Gender':{'Male':1,'Female':0}},inplace=True)
df.replace({'Loan_Status':{'Y':1,'N':0}},inplace=True)
df.replace({'Self_Employed':{'Yes':1,'No':0}},inplace=True)

print(df.Property_Area.value_counts()) 
# Property_Area
# Semiurban    191
# Urban        150
# Rural        139

# repalce urban,rural,semiurban....using OHE
area_dummy=pd.get_dummies(df.Property_Area,dtype=int,drop_first=True)
print(area_dummy)

new_df=pd.concat([df,area_dummy],axis='columns')
new_df.drop(['Property_Area'],axis='columns',inplace=True)

print(new_df)

#########################################################################
#---------------------FEATURES---------------------------
x=new_df.drop(['Loan_ID','Loan_Status'],axis='columns') # Independent
y=new_df['Loan_Status'] # Dependent

#--------SPLIT DATA-------------------------------------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)

#-------TRAIN MODEL-----------------------------------------
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)
print("Model is Trained Successfully!")

#---------TRAIN-PREDICTION---------------------------------
y_train_pred=model.predict(x_train)
print(y_train_pred)
print(y_train)
print("TRAIN Model Score(Accuracy):", model.score(x_train,y_train))
#-------TEST-PREDICTION------------------------------------------
y_test_pred=model.predict(x_test)
print("TEST Model Score(Accuracy):", model.score(x_test,y_test))


