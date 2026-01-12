# Suspicious Login Detection Using Logistic Regression
# Detect whether a login attempt is normal or suspicious
# based on parameter like login time, location, device type, and previous failed attempts.

import pandas as pd

df=pd.read_csv("E:\Desktop\ML\LOGISTIC_REGRESSION\login_data.csv")

# Features
x=df.drop(['Suspicious'],axis='columns')
y=df['Suspicious']

# split dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)
# print("Model is trained successfully!")

# Test Prediction 
y_test_pred=model.predict(x_test)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_test_pred)
# print(accuracy)

################################################################3
#--------GUI-------------------------------------
import tkinter as tk 
from tkinter import messagebox

app = tk.Tk()
app.title("Suspicious Login Detection")
app.geometry("500x500")
app.configure(bg="#f0f0f0")

# Entry labels and fields
fields = {
    "Login Time": None,
    "Login Location (1=Known,0=Unknown)": None,
    "Device Type": None,
    "Failed Attempts": None,
}

tk.Label(app, text="Suspicious Login Detection", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=10)
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
        time = float(fields["Login Time"].get())
        location = float(fields["Login Location (1=Known,0=Unknown)"].get())
        device = float(fields["Device Type"].get())
        attempts = float(fields["Failed Attempts"].get())
        features = [[time,location,device,attempts]]

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        # Above line calculates the probability that the loan will be approved.
        # Returns the probability of each class (0 or 1) as a list.
        # For example: [[0.10, 0.90]] 10% chance of Not Approved (class 0) 90% chance of Approved (class 1)

        msg = f"Prediction: {'Suspecious Login Detected' if prediction == 1 else 'Login is Normal'}\n" \
              f"Login Probability: {prob*100:.2f}%\nModel Accuracy: {accuracy*100:.2f}%"
        messagebox.showinfo("Result", msg)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numerical values.")

# Button
tk.Button(app, text="Predict", command=predict_loan,
          font=("Arial", 12), bg="#4caf50", fg="white", padx=10, pady=5).pack(pady=20)

app.mainloop()