# Admission Chance Predictor Based on Test Scores and Profile
# This project uses a Logistic Regression model to predict
# whether a student will be admitted (1) or not admitted (0) based on:
# GRE score , TOEFL score ,SOP (Statement of Purpose) score, CGPA, Research experience

import pandas as pd 

#------------LOAD DATASET---------------------------------
df=pd.read_csv(r"E:\Desktop\ML\LOGISTIC_REGRESSION\admission_data.csv")

#-------------FEATURES---------------------------------
x=df.drop(['Admitted'],axis='columns') # Independent
y=df['Admitted'] # Dependent

#-----------SPLIT DATASET-----------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#---------------TRAIN MODEL-------------------------------
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)
# print("Model is Train Successfully!")

#----------TEST PREDICTION-----------------------------
y_test_pred=model.predict(x_test)
# accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_test_pred)
print(accuracy)

#---------MANUAL PREDICTION-----------------------------------
# print("\n--- Predict Admission ---")
# gre = float(input("Enter GRE Score: "))
# toefl = float(input("Enter TOEFL Score: "))
# sop = float(input("Enter SOP Score (1-5): "))
# cgpa = float(input("Enter CGPA (out of 10): "))
# research = int(input("Research Experience (1 = Yes, 0 = No): "))

# model_pred=model.predict([[gre,toefl,sop,cgpa,research]])[0]
# prob_pred=model.predict_proba([[gre,toefl,sop,cgpa,research]])[0][1]

# # results
# print("Prediction:","Admitted" if model_pred==1 else 'Not Admitted')
# print(f"Probabilty (Chances): {prob_pred*100:.2f}%") 
# print(f"Model Accuracy: {accuracy*100:.2f}%")

#############################################################################################
#--------GUI-------------------------------------
import tkinter as tk 
from tkinter import messagebox

app = tk.Tk()
app.title("Suspicious Login Detection")
app.geometry("500x500")
app.configure(bg="#f0f0f0")

# Entry labels and fields
fields = {
    "GRE Score": None,
    "TOEFL Score": None,
    "SOP Score (1-5)": None,
    "CGPA (out of 10)": None,
    "Research Experience (1=Yes,0=No)": None
}

tk.Label(app, text="Admission Prediction", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=10)
frame = tk.Frame(app, bg="#f0f0f0")
frame.pack()

for i, label in enumerate(fields):
    tk.Label(frame, text=label, font=("Arial", 12), bg="#f0f0f0").grid(row=i, column=0, pady=8, padx=10, sticky="w")
    entry = tk.Entry(frame, font=("Arial", 12), width=20)
    entry.grid(row=i, column=1, pady=8, padx=10)
    fields[label] = entry

# Prediction function
def predict_loan():
    try:
        gre = float(fields["GRE Score"].get())
        toefl = float(fields["TOEFL Score"].get())
        sop = float(fields["SOP Score (1-5)"].get())
        cgpa = float(fields["CGPA (out of 10)"].get())
        research=int(fields["Research Experience (1=Yes,0=No)"].get())
        features = [[gre,toefl,sop,cgpa,research]]

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        # Above line calculates the probability that the loan will be approved.
        # Returns the probability of each class (0 or 1) as a list.
        # For example: [[0.10, 0.90]] 10% chance of Not Approved (class 0) 90% chance of Approved (class 1)

        msg = f"Prediction: {'Admitted' if prediction == 1 else 'Not Admitted'}\n" \
              f"Admission Probability: {prob*100:.2f}%\nModel Accuracy: {accuracy*100:.2f}%"
        messagebox.showinfo("Result", msg)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numerical values.")

# Button
tk.Button(app, text="Predict", command=predict_loan,
          font=("Arial", 12), bg="#4caf50", fg="white", padx=10, pady=5).pack(pady=20)

app.mainloop()