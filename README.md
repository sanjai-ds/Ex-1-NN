<H3>NAME : Sanjai S</H3>
<H3>REGISTER NO: 212223230186</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 17/05/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
Import Libraries
```py

from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```
Read the dataset
```py
df=pd.read_csv("Churn_Modelling.csv")

```
Checking Data
```py
df.head()
df.tail()
df.columns
```
Check the missing data
```py
df.isnull().sum()
```

Check for Duplicates
```
df.duplicated()
```
Assigning Y
```
y = df.iloc[:, -1].values
print(y)
```
Check for duplicates
```py
df.duplicated()
```
Check for outliers
```py
df.describe()
```
Dropping string values data from dataset
```py
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()
```
Normalize the dataset
```py
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
Split the dataset
```py
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```
Training and testing model
```py
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```


## OUTPUT:
Data checking:

![output](1.png)

Missing Data:

![output](4.png)

Duplicates identification:

![output](6.png)


Values of 'Y':

![output](3.png)


Outliers:

![output](7.png)

Checking datasets after dropping string values data from dataset:

![output](8.png)

Normalize the dataset:

![output](9.png)

Split the dataset:

![output](10.png)

Training and testing model:

![output](11.png)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


