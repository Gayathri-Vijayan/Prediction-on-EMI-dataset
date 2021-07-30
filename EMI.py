#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')


# # Loading the Dataset

# In[2]:


df=pd.read_csv("EMI_dataset.csv")
df.head()


# # Data Preprocessing

# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


print(df['organization'].unique())


# In[7]:


df_model=df


# In[8]:


df['E13']=df['E13'].replace(0,'Paid')
df['E13']=df['E13'].replace(1,'Not paid')


# In[9]:


df


# In[10]:


cols = df.columns
cols


# # Explor̥atory Data Analysis

# In[11]:


plt.subplots(figsize=(10,6))
sns.countplot(df['country'],palette='cool')


# In[12]:


sns.countplot(data=df, x="country", hue="E13")


# The country X has more loan applicants and considering the repayment country Z has much more paid customers

# In[13]:


plt.subplots(figsize=(10,6))
sns.distplot(df["interest_rate"])


# In[14]:


plt.subplots(figsize=(10,6))
sns.distplot(df["unpaid_principal_bal"],color="g")


# In[15]:


plt.subplots(figsize=(10,6))
sns.distplot(df["borrower_credit_score"],color="r")


# In[16]:


df1 = pd.DataFrame(df.groupby("country")["E13"].value_counts())
df1


# In[17]:


sns.boxplot(x="E13",y="interest_rate",data=df)


# Interest rate ranging between 4-4.5 are less for repayment

# In[18]:


df['loan_term'].value_counts()[:5].plot(kind='pie', autopct = "%.2f%%", figsize=(10,7), shadow=True, startangle=135,cmap='Oranges')
plt.ylabel('Loan Term',fontsize='15')
plt.title('Plot of duration of loans')


# Majority of the loan duration is 360 days

# In[19]:


plt.subplots(figsize=(16,8))
ax = sns.barplot(x = df['organization'].value_counts()[:10].index, y = df['organization'].value_counts()[:10],palette='cool')
plt.title('Organization',fontsize=25)
plt.xticks(rotation=90,fontsize=14)
plt.show()


# In[20]:


plt.subplots(figsize=(16,8))
sns.countplot(x="organization",hue="E13",data=df)
plt.xticks(rotation=90,fontsize=12)


# The company belonging to others have relatively more number of paid and non-paid customers

# # Model training and Evaluation

# In[21]:


cols = df_model.columns[df_model.dtypes == object]
cols


# In[22]:


df_model.head()


# In[23]:


le = LabelEncoder()
df_model['country'] = le.fit_transform(df_model['country'])
df_model['organization'] = le.fit_transform(df_model['organization'])
df_model['origination_date'] = le.fit_transform(df_model['origination_date'])
df_model['first_payment_date'] = le.fit_transform(df_model['first_payment_date'])
df_model['loan_purpose'] = le.fit_transform(df_model['loan_purpose'])
df_model['E13'] = le.fit_transform(df_model['E13'])


# In[24]:


X = df_model.drop(['candidate_id','E13'],axis=1)
y = df_model['E13']


# In[25]:


X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size=0.2)


# In[26]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
print(accuracy_score(y_test,model.predict(X_test)))
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
f1_score(y_test,y_pred)


# In[27]:


from sklearn.naive_bayes import BernoulliNB
NB= BernoulliNB()
NB.fit(X_train, y_train)
print(accuracy_score(y_test,NB.predict(X_test)))
y_pred = NB.predict(X_test)
print(classification_report(y_test,y_pred))
f1_score(y_test,y_pred)


# In[28]:


from sklearn.tree import DecisionTreeClassifier
DT= DecisionTreeClassifier()
DT.fit(X_train, y_train)
print(accuracy_score(y_test,DT.predict(X_test)))
y_pred = DT.predict(X_test)
print(classification_report(y_test,y_pred))
f1_score(y_test,y_pred)


# In[29]:


from sklearn.ensemble import  AdaBoostClassifier
AB = AdaBoostClassifier()
AB.fit(X_train, y_train)
print(accuracy_score(y_test,AB.predict(X_test)))
y_pred = AB.predict(X_test)
print(classification_report(y_test,y_pred))
f1_score(y_test,y_pred)


# In[30]:


from sklearn.svm import SVC
S= SVC()
S.fit(X_train, y_train)
print(accuracy_score(y_test,S.predict(X_test)))
y_pred = S.predict(X_test)
print(classification_report(y_test,y_pred))
f1_score(y_test,y_pred)


# In[31]:


from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()
lreg.fit(X_train,y_train)
print(accuracy_score(y_test,lreg.predict(X_test)))
y_pred = lreg.predict(X_test)
print(classification_report(y_test,y_pred))
f1_score(y_test,y_pred)


# In[32]:


from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train,y_train)
print(accuracy_score(y_test,KNN.predict(X_test)))
y_pred = KNN.predict(X_test)
print(classification_report(y_test,y_pred))
f1_score(y_test,y_pred)


# Majority of the models could classify the data giving an accuracy approximately 99% and high F-score. 
# Comparatively Random Forest Classifier has higher performance and could be chosen for the classification purpose

# In[ ]:




