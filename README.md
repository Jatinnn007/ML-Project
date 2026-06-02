Titanic Survival Prediction 🚢

A Machine Learning project that predicts whether a passenger survived the Titanic disaster using Logistic Regression.

This project uses data preprocessing and classification techniques with Scikit-learn.

---

📌 Project Overview

The model is trained on the Titanic dataset and predicts passenger survival based on features like:

* Passenger Class
* Gender
* Age
* Fare
* Embarkation Port

The project demonstrates:

* Data cleaning
* Missing value handling
* Label encoding
* One-hot encoding
* Feature scaling
* Logistic Regression classification

---

🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

---

📂 Dataset

Dataset files used:

* train.csv
* test.csv

Dataset source:
Titanic dataset from Kaggle.

---

⚙️ Features Used

The following features were used for prediction:

* Pclass
* Sex
* Age
* Fare
* Embarked

Unused columns removed:

* Cabin
* SibSp
* Name
* Ticket
* Parch
* PassengerId

---

🔍 Data Preprocessing

The following preprocessing steps were applied:

1 --->  Missing Value Handling

* Age → filled using median
* Embarked → filled using mode
* Fare → filled using median

2 ---> Encoding

* Label Encoding for Gender
* One-Hot Encoding for Embarked column

3 ---> Feature Scaling

* StandardScaler used for normalization

---

🤖 Machine Learning Model

Model Used:
LogisticRegression()

The dataset was split into:
* 80% Training Data
* 20% Testing Data

---

🧪 Sample Prediction
person = np.array([[2, 1, 35, 12.35, 0, 1, 0]])

Prediction Output:
This person is likely alive.

          OR
          
This person is likely dead.
