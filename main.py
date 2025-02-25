import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("LoanApprovalPrediction.csv")

print("\nShape :", data.shape)
print("\nNoise Data Count Prediction: ", data.isnull().sum())
print("\nMean= ",data['Dependents'].mean() )
print("Median= ",data['Dependents'].median())
print("Mode= ",data['Dependents'].mode())

result1 = data['Dependents'].fillna(data['Dependents'].median())
print(result1)
plt.title('Dependents')
plt.hist(result1)
plt.show()
x=np.arange(0,598,1)
plt.scatter(x,result1)
plt.show()
plt.boxplot(result1)
plt.show()

print("\nMean= ",data['LoanAmount'].mean())
print("Median= ",data['LoanAmount'].median())
print("Mode= ",data['LoanAmount'].mode())

result2 = data['LoanAmount'].fillna(data['LoanAmount'].mean())
print(result2)
plt.title('LoanAmount')
plt.hist(result2)
plt.show()
x=np.arange(0,598,1)
plt.scatter(x,result2)
plt.show()
plt.boxplot(result2)
plt.show()

print("\nMean= ",data['Loan_Amount_Term'].mean())
print("Median= ",data['Loan_Amount_Term'].median())
print("Mode= ",data['Loan_Amount_Term'].mode())

result3 = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
print(result3)
plt.title('Loan_Amount_Term')
plt.hist(result3)
plt.show()
x=np.arange(0,598,1)
plt.scatter(x,result3)
plt.show()

print("\nMean= ",data['Credit_History'].mean())
print("Median= ",data['Credit_History'].median())
print("Mode= ",data['Credit_History'].mode())
result4 = data['Credit_History'].fillna(data['Credit_History'].median())
print(result4)
plt.title('Credit_History')
plt.hist(result4)
plt.show()
x=np.arange(0,598,1)
plt.scatter(x,result4)
plt.show()

data['Dependents'] = result1
data['LoanAmount'] = result2
data['Loan_Amount_Term'] = result3
data['Credit_History'] = result4


import seaborn as sns
fig, ax = plt.subplots(2,2, figsize=(10,10))

sns.countplot(x='Loan_Status', data = data, ax=ax[0][0])
sns.countplot(x='Gender', data = data, ax=ax[0][1])
sns.countplot(x="Education", hue = "Loan_Status", data = data, palette="RdBu", ax=ax[1][0])
sns.countplot(x = "Gender" , hue = "Loan_Status", data= data, palette= "rocket", ax=ax[1][1])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#KNN Algorithm
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# Making predictions
y_pred = knn.predict(X_test)
# Evaluating model
accuracy = accuracy_score(y_test, y_pred)



#SVM Algorithm
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)
# Predict the target variable for test data
y_pred = svm_model.predict(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
# Test the model
y_pred = model.predict(X_test)
# Evaluate the model
acCuracy = accuracy_score(y_test, y_pred)


#Random Forest
# Initialize the Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
rfc.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = rfc.predict(X_test)
# Evaluate the model
aCcuracy = accuracy_score(y_test, y_pred)


#Decision Tree
# Create decision tree classifier object
clf = DecisionTreeClassifier()
# Train the classifier on the training data
clf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = clf.predict(X_test)
# Calculate accuracy score
accuracY = accuracy_score(y_test, y_pred)

# Calculate accuracy score
Accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of KNN :",accuracy*100.0)
print("Accuracy of SVM : ",Accuracy*100.0)
print("Accuracy of Logistic Regression : ", acCuracy * 100.0)
print("Accuracy of Random Forest :", aCcuracy*100.0)
print("Accuracy of Decision Tree :", accuracY*100.0)


models = ['KNN', 'SVM', 'Logistic Regression', 'Random Forest', 'Decision Tree']
accuracies = [accuracy*100.0, Accuracy*100.0, acCuracy*100.0, aCcuracy*100.0, accuracY*100.0]
colors = ['red', 'blue', 'green', 'purple', 'orange']
plt.bar(models, accuracies, color=colors, width = 0.4)
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Scores of Machine Learning Models')
plt.show()






