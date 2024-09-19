# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages. 
2. Print the present data and placement data and salary data.
3.Using logistic regression find the predicted values of accuracy confusion matrices.
4.Display the results.
```
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Oviya N 
RegisterNumber:212223040140  
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1.head()
x=data1.iloc[:,:-1]
print(x) #allocate the -1 column for x
y=data1["status"]
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
#classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
```

## Output:

Placement Data:

![image](https://github.com/user-attachments/assets/0974ffdd-650d-4bbe-a042-bbb87614d3d1)

Checking the null() function:

![image](https://github.com/user-attachments/assets/ce722f9a-4add-46d0-b7a1-d04df43b07c1)

Data Duplicate:

![image](https://github.com/user-attachments/assets/372b189b-a454-43b9-bf6d-04cbbbe5ead9)

Print Data:

![image](https://github.com/user-attachments/assets/9b38beca-7ea4-462c-bb21-aa72a9d2e073)

Y_prediction array:

![image](https://github.com/user-attachments/assets/9d25b586-8dd3-4985-9f16-76d953da50f2)

Accuracy value:

![image](https://github.com/user-attachments/assets/482b0f56-8ac6-4918-9049-e6b5f07cf9e3)

Confusion array:

![image](https://github.com/user-attachments/assets/f1da7069-e57b-4026-8bb7-dd16694adf68)

Classification Report:

![image](https://github.com/user-attachments/assets/d5466b91-1b14-4d37-8a8d-70d07439fcc2)

Prediction of LR:

![image](https://github.com/user-attachments/assets/65d7f36a-18c7-4b25-97f7-78f33d652598)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
