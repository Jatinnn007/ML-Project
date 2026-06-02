import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

df = pd.read_csv(r"C:\Users\jatin\Desktop\mlpractice\data\train.csv")
df1 = pd.read_csv(r"C:\Users\jatin\Desktop\mlpractice\data\test.csv")


df = df.drop(['Cabin','SibSp','Name','Ticket','Parch','PassengerId'],axis=1)
df1 = df1.drop(['Cabin','SibSp','Name','Ticket','Parch','PassengerId'],axis=1)


df['Age'] = df['Age'].fillna(df['Age'].median())
df1['Age'] = df1['Age'].fillna(df1['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df1['Fare'] = df1['Fare'].fillna(df['Fare'].median())

encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df1['Sex'] = encoder.transform(df1['Sex'])


embark_train = pd.get_dummies(df['Embarked'], prefix='Embarked')
embark_test = pd.get_dummies(df1['Embarked'],prefix='Embarked')

df = pd.concat([df,embark_train],axis=1)
df1 = pd.concat([df1,embark_test],axis=1)

df = df.drop('Embarked',axis=1)
df1 = df1.drop('Embarked',axis=1)




X = df.drop('Survived',axis=1)
y = df['Survived']

X ,df1 = X.align(df1,join='left',axis=1)

X_train , X_test , y_train , y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

df1 = scaler.transform(df1)

model = LogisticRegression()
model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)
print(f"Accuracy: {accuracy * 100 :.2f}")


person = np.array([[2, 1, 35, 12.35, 0, 1, 0]])
person = scaler.transform(person)
prediction = model.predict(person)


if prediction == 1:
    print("This person is likely alive.")
else:
    print("This person is likely dead")