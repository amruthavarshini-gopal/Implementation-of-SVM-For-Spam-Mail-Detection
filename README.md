# Implementation-of-SVM-For-Spam-Mail-Detection

## Aim:

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import required libraries for encoding detection, data handling, text processing, machine learning, and evaluation.

2.Open the dataset file (spam.csv) in binary mode and detect its character encoding using chardet.

3.Load the CSV file into a pandas DataFrame using the detected encoding (e.g., 'windows-1252').

4.Display the dataset using head(), info(), and check for any missing values with isnull().sum().

5.Assign the label column (v1) to variable x and the message text column (v2) to variable y.

6.Split the dataset into training and testing sets using an 80-20 split with train_test_split().

7.Convert the text data into numerical format using CountVectorizer to prepare for model training.

8.Initialize the Support Vector Machine classifier (SVC) and train it on the vectorized training data.

9.Predict the labels of the test set using the trained SVM model.

10.Evaluate the model’s performance by calculating and printing the accuracy score using accuracy_score().

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Amruthavarshini Gopal
RegisterNumber: 212223230013 
*/
import chardet
file="spam.csv"
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding="windows-1252")
data.head()
data.info()
data.describe()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
```

## Output:

### Head
![Screenshot 2025-05-23 221650](https://github.com/user-attachments/assets/eb9a86c4-2b46-4fa9-b378-5a58bafe7b9b)

### data.info()
![Screenshot 2025-05-23 221717](https://github.com/user-attachments/assets/15e247d8-2f4b-4345-87b0-041877d24414)

### data.describe()
![Screenshot 2025-05-23 221738](https://github.com/user-attachments/assets/d7574454-49b9-44a7-81c5-bfc5ece2a91f)

### data.isnull().sum()
![Screenshot 2025-05-23 221751](https://github.com/user-attachments/assets/1faa62ad-f407-45b9-a6bf-4f48683efbac)

### y predicted 
![Screenshot 2025-05-23 221805](https://github.com/user-attachments/assets/99c45c8c-9d4d-4fb0-8de7-9db453284461)

### Accuracy
![Screenshot 2025-05-23 221828](https://github.com/user-attachments/assets/f37145f5-32b6-4a79-bf2c-dab6c69dda10)


## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
