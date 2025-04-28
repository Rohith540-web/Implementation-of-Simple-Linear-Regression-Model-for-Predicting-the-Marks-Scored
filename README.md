# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import Required Libraries

Import libraries for numerical computation (numpy), data handling (pandas), plotting (matplotlib), and performance metrics from sklearn.

Step 2: Load Dataset

Read the CSV file student_scores.csv using pandas.read_csv().

x stores the input features (hours studied).

y stores the target variable (exam scores).

Step 3: Split Data into Training and Testing Sets

Use train_test_split() to split x and y into:

x_train, y_train: for training

x_test, y_test: for evaluation

Use a test size of 1/3 and a fixed random_state for reproducibility.

Step 4: Train the Linear Regression Model

Create a LinearRegression model instance.

Fit the model to the training data using regressor.fit(x_train, y_train).

Step 5: Make Predictions

Use regressor.predict() on x_test to get predicted scores y_pred.

Step 6: Evaluate the Model

Calculate evaluation metrics:

MSE (Mean Squared Error)

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Print the values of MSE, MAE, and RMSE.

Step 7: Visualize the Results

Training Set Plot:

Scatter plot of actual training data (x_train, y_train).

Regression line based on predictions from training data.

Test Set Plot:

Scatter plot of actual test data (x_test, y_test).

Regression line based on predicted test data (y_pred).


## Program:

```
Developed by: ROHITH V
RegisterNumber: 212224220083 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
mse=mean_squared_error(y_test,y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="green")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


```

## Output:
Head:


![Screenshot 2025-04-28 154418](https://github.com/user-attachments/assets/d942b7a6-c34d-42e4-97e4-b69a0e26ba75)


Tail:


![Screenshot 2025-04-28 154425](https://github.com/user-attachments/assets/40fb3d4c-a451-49f3-9391-6ac48d74eeaf)


X value:


![Screenshot 2025-04-28 154434](https://github.com/user-attachments/assets/581baf79-de1c-4abe-a1aa-dce878d2bfc2)


Y value:


![Screenshot 2025-04-28 154440](https://github.com/user-attachments/assets/03b11c37-80ec-4f0d-8ac3-29c5ce41658b)


y predict:


![Screenshot 2025-04-28 154450](https://github.com/user-attachments/assets/397cc32c-30ba-4865-9111-bafb4710866c)


y test:


![Screenshot 2025-04-28 154457](https://github.com/user-attachments/assets/9b45f225-d3f2-4263-8df1-f42bb8de1adf)



![Screenshot 2025-04-28 154506](https://github.com/user-attachments/assets/f6a730ea-27ea-465c-bf24-d1bb5f77c6cc)


Graph:

![Screenshot 2025-03-09 200155](https://github.com/user-attachments/assets/9bb15c6d-7f3c-42ef-8f53-295e931f6268)


![Screenshot 2025-03-09 200211](https://github.com/user-attachments/assets/f0e5bcbf-c648-4178-a1da-933bd732dc16)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
