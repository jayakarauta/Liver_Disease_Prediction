#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import python libraries
# visualizing data


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# import csv file


# In[4]:


dataset = pd.read_csv(r"J:\1.ML_PROJECT\2.ML_PROJECTS\2.LIVER\indian_liver_patient.csv")


# In[5]:


# checking the first 5 rows and colums 


# In[6]:


dataset.head()


# In[7]:


#The describe function in the pandas library is used to generate descriptive statistics of a DataFrame or Series. 
#When applied to a DataFrame, it provides summary statistics for each numerical column, 
#including count, mean, standard deviation, minimum, and maximum values. 
#The output of describe is a comprehensive summary that aids in understanding the basic statistical properties of the data.


# In[8]:


dataset.describe()


# In[9]:


# i have to check only columns means it will show only columns


# In[10]:


dataset.columns


# In[11]:


#data cleaning 
#checking duplicate tuple,if any will be removed


# In[13]:


dataset.duplicated ()


# In[14]:


# above any value is repeating it will show true .in this data its not showing true  


# In[15]:


dataset.duplicated ().sum()


# In[16]:


#here i can see 13 rows are duplicated


# In[17]:


# now i am deleting repeated values


# In[21]:


dataset=dataset.drop_duplicates()
print (dataset.shape)


# In[22]:


#i can see clearly here 570 rows and 11 columns


# In[23]:


#checking the null values in the data set


# In[24]:


dataset.isna().sum()


# In[25]:


# maximum all featuers having a null vallues except one featuer in dataset Albumin_and_Globulin_Ratio 4


# In[26]:


# i have to fill the missing letters in dataset .i can use three methods finding mean are median are mode i can use any method 
#to find the value
#mean ehen i will use means outliears are not there i can use mean method 
# first i have to find the outliears 


# In[27]:


sns.boxplot(data=dataset,x='Albumin_and_Globulin_Ratio')


# In[28]:


#above i used seborn for box plot finding the outliear visualiztion
# i can see clerly outliears  are there in this data
# i can't use mean here because outliears are there
# i can use median are mode to fill the missing values


# In[29]:


# i am finding mode ,median & mean


# In[31]:


dataset['Albumin_and_Globulin_Ratio'].mode()


# In[32]:


dataset['Albumin_and_Globulin_Ratio'].median()


# In[33]:


dataset['Albumin_and_Globulin_Ratio'].mean()


# In[34]:


# i compare with mean & median here the values are almost similar 
# some time mean value will come very high are low is totaly depends on the outliears.
# in mean outilears values impact more so maximum try to avoid outliers to get best value 
# here for me mean & median is similar


# In[35]:


# now i am filling the mising values with median 


# In[37]:


dataset['Albumin_and_Globulin_Ratio'] = dataset['Albumin_and_Globulin_Ratio'].fillna(dataset['Albumin_and_Globulin_Ratio'].median())


# In[38]:


dataset.isna().sum()


# In[39]:


# here i can see all values are in null 


# In[40]:


#now i am checking how many men patients & how mant female patients .will i am using countploy


# In[43]:


import seaborn as sns
sns.countplot(data=dataset, x='Gender', label='count')


# In[44]:


# i can see male patients graph is high  compare to female patients graph


# In[45]:


# now i have to check how many male and female patients are there 


# In[48]:


Male,Female=dataset['Gender'].value_counts()
print('Number of patients that are Male:',Male)
print('Number of patients that are Female:',Female)


# In[49]:


#here in my data set gender is catagericol variable .
#now i have to change to numerical .why mean ml will take only numerical variables


# In[50]:


# encoding the gender column
#lable male as 1 and female as 0


# In[51]:


def partition (x):
    if x=='male':
        return 1
    return 0
dataset['Gender']=dataset['Gender'].map(partition)


# In[53]:


dataset


# In[54]:


# i can all the variable converted to interger formate .now i can model properly in ml


# In[55]:


#converting output column 'dataset' to 0 and 1


# In[56]:


#dataset output value has 1for liver disease and 2 for no liver disease  so let's make it 0 forno disease to maeke it coninient


# In[57]:


def partition (x):
    if x== 2:
        return 0
    return 1
dataset['Dataset']=dataset['Dataset'].map(partition)


# In[58]:


dataset['Dataset']


# In[59]:


# all converterd into 0&1


# In[60]:


# i am finding corrleation between all the features in dataset


# In[62]:


plt.figure(figsize=(10,10))
sns.heatmap(dataset.corr())
plt.show()


# In[63]:


# above plot we can see in diagonally all in white color 
# white color represents 100% two variable are equal
# black color represents negatively corelated here its showing .so its a reverse
# we can say that one variable is increasing one variable is decreasing
# one of the featuer slection matrix


# In[64]:


#data preparation
# creating  feature matrix and dependent variable vector
# all independent variablle i am converting into onwe array


# In[65]:


X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values


# In[ ]:


#here X is featuer matrix x is independent colums coverting into one array
#here y is dependent variable vector in target colums variables converted into one array
#my target colum is last colum (dataset)


# In[66]:


# to slove this i have split the data into train data & test data


# In[73]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[74]:


#featuer scaling 


# In[77]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[78]:


# what i did here means i trained into standard scaler method 
# X_train & X_test 


# In[79]:


# i completed data preprocing up to now 


# In[82]:


# meachine learning models


# In[83]:


#logistic Regression


# In[86]:


from sklearn.linear_model import LogisticRegression

log_classifier = LogisticRegression(random_state=0)
log_classifier.fit(X_train, y_train)

    


# In[87]:


#perfect trained in logistic regression


# In[89]:


# predicting the output
log_y_pred = log_classifier.predict(X_test)


# In[90]:


# now i have to train model in confusion matrix


# In[93]:


from sklearn.metrics import confusion_matrix
log_cm = confusion_matrix(y_test, log_y_pred)
sns.heatmap(log_cm, annot=True)
plt.show()


# In[94]:


#True Positive (TP): The number of observations that were correctly predicted as the positive class.
#True Negative (TN): The number of observations that were correctly predicted as the negative class.
#False Positive (FP): The number of observations that were incorrectly predicted as the positive class (Type I error).
#False Negative (FN): The number of observations that were incorrectly predicted as the negative class (Type II error).


# In[95]:


#according to above data flase negative is more so we have to decrease the fn value is type 2 error


# In[96]:


# finding the accuracy_score,precision_score


# In[98]:


from sklearn.metrics import accuracy_score,precision_score
print(accuracy_score(y_test,log_y_pred))
print(precision_score(y_test, log_y_pred))


# In[99]:


# right now i am tacking KNN model k nearest neighbors algorithum


# In[100]:


X_train.shape


# In[102]:


# for checking how many observations in data .i taken shape for refferance 
# square root no of observation in traning data 
# in x place we will replace the k value so we have to do square root for observation 


# In[109]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=21, metric='minkowski')
knn_classifier.fit(X_train, y_train)


# In[106]:


#The Minkowski distance is a generalization of various distance metrics, including Euclidean and Manhattan distances. 
#It is controlled by a parameter p, where:


# In[107]:


# now i have to check the model perfoming good are not
# will using prediction method


# In[108]:


knn_y_pred = knn_classifier.predict(X_test)


# In[110]:


from sklearn.metrics import confusion_matrix
knn_cm = confusion_matrix(y_test, knn_y_pred)
sns.heatmap(knn_cm, annot=True)
plt.show()


# In[111]:


# finding the accuracy_score,precision_score


# In[115]:


from sklearn.metrics import accuracy_score,precision_score
print(accuracy_score(y_test,knn_y_pred))
print(precision_score(y_test, knn_y_pred))


# In[116]:


# Support Vector Machines


# In[126]:


from sklearn.svm import SVC

svm_classifier = SVC(kernel='rbf', random_state=0)
svm_classifier.fit(X_train, y_train)


# In[130]:


svm_y_pred = svm_classifier.predict(X_test)


# In[131]:


from sklearn.metrics import confusion_matrix
svm_cm = confusion_matrix(y_test, svm_y_pred)
sns.heatmap(svm_cm , annot=True)    


# In[132]:


from sklearn.metrics import accuracy_score,precision_score
print(accuracy_score(y_test,svm_y_pred))
print(precision_score(y_test, svm_y_pred))


# In[ ]:




