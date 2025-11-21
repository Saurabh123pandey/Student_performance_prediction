# 📘 Student Performance Prediction Model

This project is a **Machine Learning-based Student Performance Prediction System** that uses multiple study-related factors to predict a student's final exam score. The model is built using **Linear Regression** and provides insights into how different inputs affect academic performance.

---

## 🚀 Project Overview

The goal of this project is to **predict the Final Score of a student** based on:

* Study hours per week
* Attendance percentage
* Number of assignments completed
* Daily sleep hours
* Internet usage hours per day

This helps in understanding how study habits and lifestyle factors contribute to academic results.

---

## 📂 Dataset Details (`students_data.csv`)

The dataset used should contain the following columns:

| Column Name                    | Description                               |
| ------------------------------ | ----------------------------------------- |
| `study_hours_per_week`         | Total study hours in a week               |
| `attendance_percent`           | Attendance percentage (%)                 |
| `assignments_completed`        | Number of assignments finished            |
| `sleep_hours`                  | Daily sleep duration (hours)              |
| `internet_usage_hours_per_day` | Daily internet usage (hours)              |
| `final_score`                  | Actual final exam score (target variable) |

Make sure all columns exist in the CSV file before running the code.

---

## 🧠 Model Used

We use **Linear Regression**, a supervised learning algorithm that finds a best-fit line through the data.

### Why Linear Regression?

* Simple and easy to interpret
* Works well with continuous output values (like scores)
* Fast training

---

## 📌 Installation & Requirements

Install the required libraries:

```bash
pip install pandas scikit-learn matplotlib numpy
```

---

## 🧪 Code Explanation

### **1. Importing Libraries**

The project uses:

* `pandas` for data handling
* `LinearRegression` for the ML model
* `matplotlib` for graphs
* `numpy` for numeric operations
* `metrics` for model evaluation

### **2. Loading the Dataset**

```python
data = pd.read_csv("students_data.csv")
```

### **3. Selecting Features and Target**

```python
X = data[['study_hours_per_week',
          'attendance_percent',
          'assignments_completed',
          'sleep_hours',
          'internet_usage_hours_per_day']]

y = data['final_score']
```

### **4. Training the Model**

```python
model = LinearRegression()
model.fit(X, y)
```

### **5. Predicting**

```python
pred_score = model.predict(X)
print("Predicted score sample:", round(pred_score[0], 2))
```

---

## 📊 Model Evaluation Metrics

The following metrics help measure model accuracy:

| Metric       | Meaning                                  |
| ------------ | ---------------------------------------- |
| **MAE**      | Average absolute error                   |
| **MSE**      | Squared error (penalizes large mistakes) |
| **RMSE**     | Root of MSE (easy interpretation)        |
| **R² Score** | How well data fits the model (0 to 1)    |

Example:

```python
mae = mean_absolute_error(y, pred_score)
mse = mean_squared_error(y, pred_score)
rmse = np.sqrt(mse)
r2 = r2_score(y, pred_score)
```

---

## 📈 Visualizations

### **1. Histogram of Final Scores**

Shows distribution of student marks.

### **2. Scatter Plot (Study Hours vs Score)**

Shows relationship between study time and performance.
A **trend line** is added for better visualization.

---

## 🎯 Testing the Model with User Input

The script allows users to enter data manually:

```python
hour = int(input("Enter Study Hour Per Week: "))
attendance = int(input("Enter Attendance Percent: "))
assignment = int(input("Enter Assignments Completed: "))
sleep_hours = int(input("Enter Sleep Hour: "))
internet_use = int(input("Enter Internet Usage Hours Per Day: "))

new_input = [[hour, attendance, assignment, sleep_hours, internet_use]]
new_predic = model.predict(new_input)
print(f"Predicted Final Score = {round(new_predic[0], 2)}")
```

---

## 📌 Key Features

✔ Simple and easy-to-understand code
✔ Uses real-life features
✔ Shows graphs for better analysis
✔ User can test the model with custom values
✔ Beginner-friendly ML project

---

## 📁 Project Structure

```
├── students_data.csv
├── student_performance_model.ipynb  (or .py file)
├── README.md
└── plots/ (optional)
```

---

## 🌟 Future Improvements

* Add more ML models (Random Forest, Gradient Boosting)
* Build a GUI or Web App (Streamlit / Flask)
* Add data cleaning and preprocessing
* Add feature selection techniques

---

## 🙌 Author

**Saurabh Pandey**
Machine Learning Beginner | AIML Enthusiast

If you like this project, don't forget to ⭐ star the repository!

---

## 📬 Contact

For any queries:
📧 *[saurabh12cs@gmail.com](mailto:saurabh12cs@gmail.com)*

---

> This is a simple but powerful supervised learning project perfect for students and beginners in Machine Learning.
