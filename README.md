Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
1.Import Libraries and Load Dataset. 
2. Preprocess the Data. 
3. Split the Dataset.
4. Train the Decision Tree Classifier.
5. Make Predictions and Evaluate the Model

Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn. 
Developed by: KISHORE A
RegisterNumber: 212223220091
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data
```
```
data["left"].value_counts()
``
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data.head()
data["salary"]=le.fit_transform(data["salary"])
data
x=data[["satisfaction_level","last_evaluation","number_project","time_spend_company"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,2,9]])
Output:


Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.


