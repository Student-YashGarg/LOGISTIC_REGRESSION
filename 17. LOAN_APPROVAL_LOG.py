# Loan Approval Classifier for Microfinance Institutions Using Logistic Regression
# Project is to predict whether a loan application should be approved or rejected
# based on some parameters

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

#----------LOAD DATASET-------------------------------------
df = pd.read_csv("E:\Desktop\ML\LOGISTIC_REGRESSION\loan_data.csv")
# print(df.isnull().sum())
# print((df.shape))
#---------FEATURES-------------------------------------------------
X = df[['Age', 'Income', 'LoanAmount', 'Creditscore', 'PreviousLoans']]  # Indpended variables
y = df['Approved'] # Depended Variables
#----------SPLIT DATASET--------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(len(X_train))
# print(len(X_test))
#----------TRAIN MODEL------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)
#print(model)
#----------TEST PREDICTION------------------------------------------
y_pred=model.predict(X_test)
#print(y_pred)
accuracy = accuracy_score(y_test,y_pred)
#print(accuracy)
###################################################################################
#--------GUI-------------------------------------
app = tk.Tk()
app.title("Loan Approval Predictor - Microfinance")
app.geometry("420x500")
app.configure(bg="#f0f0f0")

# Entry labels and fields
fields = {
    "Age": None,
    "Income (₹)": None,
    "Loan Amount (₹)": None,
    "Credit Score": None,
    "Previous Loans Count": None
}

tk.Label(app, text="Loan Approval Predictor", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=10)
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
        age = float(fields["Age"].get())
        income = float(fields["Income (₹)"].get())
        loan_amt = float(fields["Loan Amount (₹)"].get())
        credit = float(fields["Credit Score"].get())
        prev_loans = int(fields["Previous Loans Count"].get())
        features = [[age, income, loan_amt, credit, prev_loans]]

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        # Above line calculates the probability that the loan will be approved.
        # Returns the probability of each class (0 or 1) as a list.
        # For example: [[0.10, 0.90]] 10% chance of Not Approved (class 0) 90% chance of Approved (class 1)

        msg = f"Prediction: {'Loan Approved' if prediction == 1 else 'Loan Rejected'}\n" \
              f"Approval Probability: {prob*100:.2f}%\nModel Accuracy: {accuracy*100:.2f}%"
        messagebox.showinfo("Result", msg)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numerical values.")

# Button
tk.Button(app, text="Predict Loan Approval", command=predict_loan,
          font=("Arial", 12), bg="#4caf50", fg="white", padx=10, pady=5).pack(pady=20)

app.mainloop()