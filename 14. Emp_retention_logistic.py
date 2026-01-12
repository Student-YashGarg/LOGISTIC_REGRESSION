# LOGISTIC REGRESSION
# EMPLOYEE RETENTION PREDICTION
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("E:\Desktop\ML\LOGISTIC_REGRESSION\HR_comma_sep.csv")

#-----------------LEFT----------------------
left=df[df.left==1]
print(left.shape) # row,col

#------------------RETAIN-------------------------
retain=df[df.left==0]
print(retain.shape) # row,col

#-------------CO-RELATION-----------------------
# SALARY VS LEFT
pd.crosstab(df.salary,df.left).plot(kind='bar')
plt.show()

# using seaborn
# ax=sns.countplot(data=df,x='salary',hue='left')
# for containers in ax.containers:
#     ax.bar_label(containers)
# plt.show()

# DEPARTMENT VS LEFT
pd.crosstab(df.Department,df.left).plot(kind='bar')
plt.show()

#----------------------FILTER DATA--------------------------------------------------

# we predict retention on 5 columns only
subdf=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
# convert low,med,high sal to numbers ....using OHE concept
salary_dummie=pd.get_dummies(subdf.salary,dtype=int,drop_first=True) #drop_first...avoid dummy var trap
# merge both dataframes
new_subdf=pd.concat([subdf,salary_dummie],axis='columns')
# drop salary column
new_subdf.drop(['salary'],axis='columns',inplace=True)
# Now data consist all numeric values only
print(new_subdf)

#----------FEATURES----------------------------------------------------------------------------
x=new_subdf  # independent
y=df.left    # dependent

#-----------SPLIT DATASET------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0) # 30% train
print(len(x_train))
print(len(x_test))

#------------TRAIN MODEL--------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)

#------------PREDICTION ON TEST DATASET-----------------------------------------------------------
y_test_pred=model.predict(x_test)

# Evaluation
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

print("Accuracy score:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall   :", recall_score(y_test, y_test_pred))
print("F1 Score :", f1_score(y_test, y_test_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_test_pred))

# plot confusion matrix
cm=confusion_matrix(y_test, y_test_pred)
import seaborn as sns
mydict={'fontsize':10,'fontstyle':'italic','color':'k','weight':0.5,'verticalalignment':'center'}
sns.heatmap(cm, annot=True,cmap='coolwarm',annot_kws=mydict,linewidths=2,linecolor='y', yticklabels=['Not Purchased', 'Purchased'],xticklabels=['Not Purchased', 'Purchased'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.show()

# cofficiet matrix
# [[3237  225]
#  [ 770  268]]
# Although the model achieves high overall accuracy,
# it performs poorly in identifying employees who are likely to leave, 
# as indicated by low recall and a high number of false negatives. 
# This limits its usefulness for employee retention strategies





