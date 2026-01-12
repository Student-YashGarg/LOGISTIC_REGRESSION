# LOGISTIC REGRESSION
# (MULTI MODEL CLASSIFICATION)

# DIGIT PREDICTION MODEL
# “A multiclass classification project using the Digits dataset to train a Logistic Regression model 
# that predicts handwritten digits from 8×8 pixel images.”

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
# Digits dataset teaches how a machine converts images into numbers 
# and learns patterns to classify digits.

digits=load_digits()
# print(digits)
# print(dir(digits)) # shows Data + attribut + methods....
print(digits.keys()) # shows only DATA

print(digits.images[9])

# plt.gray()
plt.matshow(digits.images[9],cmap='hot')
plt.show()

#--------------FEATURES--------------
x=digits.data # independent
y=digits.target # dependent

#--------SPLIT DATASET--------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#---------------TRAIN MODEL----------------------------
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
print("Model is Trained Successfully!")

#-------------PREDICTION ON TEST--------------------
y_test_pred=model.predict(x_test)
print("Testing Data Prediction")
print(y_test_pred[:10])
print("Actual Prediction")
print(y_test[:10])

print("Model_Score:",model.score(x_test,y_test))

#-----------CONFUSION MATRIX--------------
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_test_pred)
print("Confusion Matrix")
print(cm)


sns.heatmap(cm,annot=True)
plt.xlabel('Predication')
plt.ylabel('Truth')
plt.title("Confusion Matrix - Digits")
plt.show()