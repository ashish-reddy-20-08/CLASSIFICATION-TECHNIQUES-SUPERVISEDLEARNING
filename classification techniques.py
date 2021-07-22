import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data= pd.read_csv("train.csv")


# In[3]:


data.head()


# In[4]:


#dropping name, passengerId and ticket column 
data=data.drop(['Name'],axis=1)
data=data.drop(['Ticket'],axis=1)
data=data.drop(['PassengerId'],axis=1)


# In[5]:


data.head()


# In[6]:


#making survived column to first place
data= data[['Survived','Pclass',"Sex","Age",'SibSp','Parch','Fare','Cabin','Embarked']]


# In[7]:


data.head()


# In[8]:


data.isna().sum()


# In[9]:


data['Cabin'].value_counts()


# In[10]:


#data visulaisation
sns.heatmap(data.corr())


# In[11]:


# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
data['Cabin']= label_encoder.fit_transform(data['Sex']) 
print(data.head())


# In[12]:


data.Cabin.mode()


# In[13]:


data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])


# In[14]:


data.isna().sum()


# In[15]:


# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
data['Embarked']= label_encoder.fit_transform(data['Embarked']) 
print(data.head())


# In[16]:


data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])


# In[17]:


data.isna().sum()


# In[18]:


data.Age.mean()


# In[19]:


#mean imputation for the age column 
mean_value=data['Age'].mean()
data.fillna(value=mean_value,inplace=True)


# In[20]:


data.isna().sum()


# In[21]:


data.head()


# In[22]:


# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
data['Sex']= label_encoder.fit_transform(data['Sex']) 
print(data.head())


# In[23]:


x=data.iloc[:,1:].values


# In[24]:


y=data.iloc[:,:1].values


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=0)


# In[26]:


from sklearn.metrics import accuracy_score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # random forest

# In[ ]:


#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)   


# In[ ]:


#Fitting Decision Tree classifier to the training set  
from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(x_train, y_train)  


# In[ ]:


#Predicting the test set result  
y_pred= classifier.predict(x_test)  


# In[ ]:


#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  


# In[ ]:


cm


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy=accuracy_score(y_test,y_pred)
accuracy*100


# In[ ]:





# # knn

# In[ ]:


#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)  


# In[ ]:


y_pred= classifier.predict(x_test)


# In[ ]:


#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred) 


# In[ ]:


cm


# In[ ]:


accuracy=accuracy_score(y_test,y_pred)
accuracy*100


# In[ ]:





# In[ ]:





# # naive bayes

# In[27]:


# Fitting Naive Bayes to the Training set  
from sklearn.naive_bayes import GaussianNB  
classifier = GaussianNB()  
classifier.fit(x_train, y_train)  


# In[28]:


# Predicting the Test set results  
y_pred = classifier.predict(x_test)  


# In[29]:


# Making the Confusion Matrix  
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)  


# In[30]:


cm


# In[31]:


accuracy=accuracy_score(y_test,y_pred)
accuracy*100


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logisticRegr= LogisticRegression()

logisticRegr.fit(x_train,y_train)






# In[ ]:


y_pred=logisticRegr.predict(x_test)

y_pred


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy=accuracy_score(y_test,y_pred) 
accuracy*100


# In[ ]:





# In[ ]:





# # test data

# In[32]:


test_data= pd.read_csv("test.csv")


# In[33]:


#dropping name, passengerId and ticket column 
test_data=test_data.drop(['Name'],axis=1)
test_data=test_data.drop(['Ticket'],axis=1)
test_data=test_data.drop(['PassengerId'],axis=1)


# In[34]:


test_data.head()


# In[35]:


test_data.isna().sum()


# In[36]:


# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
test_data['Sex']= label_encoder.fit_transform(test_data['Sex']) 
print(test_data.head())


# In[37]:


# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
test_data['Embarked']= label_encoder.fit_transform(test_data['Embarked']) 
print(test_data.head())


# In[ ]:





# In[38]:


# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
test_data['Cabin']= label_encoder.fit_transform(test_data['Sex']) 
print(test_data.head())


# In[39]:


test_data.Cabin.mode()


# In[40]:


data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode()[0])


# In[41]:


test_data.head()


# In[42]:


test_data.isna().sum()


# In[43]:


#mean imputation for the age column 
mean_value=test_data['Age'].mean()
test_data.fillna(value=mean_value,inplace=True)


# In[45]:


test_data.isna().sum()


# In[46]:


test_pred=classifier.predict(test_data)


# In[47]:


test_pred


# In[48]:


test_data1= pd.read_csv("test.csv")


# In[49]:


output = pd.DataFrame({'PassengerId': test_data1.PassengerId, 'Survived': test_pred})
output.to_csv('submission3.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




