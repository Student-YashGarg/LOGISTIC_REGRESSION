# LOGISTIC_REGRESSION
# INSURANCE PURCHASE PREDICTION 
# BASED ON AGE...

# “Built a logistic regression model to predict insurance purchase behavior using age, 
# evaluated using accuracy, precision, recall, F1-score,
# and confusion matrix with probability-based interpretation.”

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

# Data
df=pd.read_csv("E:\Desktop\ML\LOGISTIC_REGRESSION\insurance_data.csv")
# print(df.head(10))

# Features
x=df[['age']] # drop bought...2D
y=df.bought_insurance

# # Graph
# plt.scatter(x,y)
# plt.show()

# Split Data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


#---------------Train Model----------------------------
model=LogisticRegression()
model.fit(x_train,y_train)
print("Model is Train Successfully!")

#----------predicted_probabilty----------
print(x_test)
print("Predicted probability:", model.predict_proba(x_test) )
print("model prob.prediction:", model.predict(x_test))

#---------Trained Prediction------------------------------
y_train_pred=model.predict(x_train)
# print("Train Accuracy Score: ",accuracy_score(y_train,y_train_pred))
print("Tain model score: ",model.score(x_train,y_train)) # model score = accuracy score


#-----------Test Prediction------------------------------------
y_test_pred=model.predict(x_test)
print("Test model score: ",model.score(x_test,y_test)) # model scoree = accuracy score

# Evaluation
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
print("Accuracy score:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall   :", recall_score(y_test, y_test_pred))
print("F1 Score :", f1_score(y_test, y_test_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_test_pred))

# plot confusion matrix
cm=confusion_matrix(y_test, y_test_pred)
import seaborn as sns
mydict={'fontsize':33,'fontstyle':'italic','color':'k','weight':0.5,'verticalalignment':'center'}
sns.heatmap(cm, annot=True, cmap='coolwarm',annot_kws=mydict,linewidths=2,linecolor='y', yticklabels=['Not Purchased', 'Purchased'],xticklabels=['Not Purchased', 'Purchased'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.show()

#-----------PREDICATION---------------------------
age=int(input("Enter age for Purchase prediction: "))
result=model.predict([[age]])[0]
if result==1:
    print(f"for {age} age : Purchased")
else:
    print(f"for {age} age : Not Purchased")

