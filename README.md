<H3>R. SANJAI S</H3>
<H3>212223230186</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 14/05/2026     </H3>
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
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
Customer_detail=pd.read_csv("Churn_Modelling.csv")
Customer_detail.head()
Customer_detail.info()
Customer_detail.dtypes
Customer_detail.nunique()
print(Customer_detail.columns)
Customer_detail.drop(["CustomerId","Surname","Age","Geography","Gender"],axis=1,inplace=True)   
Customer_detail.head()
Customer_detail.describe().round()
scaler = StandardScaler()
Detail2= pd.DataFrame(scaler.fit_transform(Customer_detail)).round(2)
Detail2
x=Detail2.iloc[:,:-1].values
x
y=Detail2.iloc[:,-1].values
y
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```



## OUTPUT:

### DATASET:
![alt text](image.png)

### DROPING THE UNWANTED DATASET:
![alt text](image-1.png)

### CHECKING FOR NULL VALUES
![alt text](image-2.png)

### CHECKING FOR DUPLICATED VALUES
![alt text](image-3.png)

### DESCRIBING THE DATASET
![alt text](image-4.png)

### SCALING THE DATASET
![alt text](image-5.png)

### X FEATURES
![alt text](image-6.png)

### Y FEATURES
![alt text](image-7.png)

### SPLITING THE DATASET INTO TRAINING AND TESTING
![alt text](image-8.png)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


