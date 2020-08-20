#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
df=pd.read_csv("C:/Users/Dell/Downloads/Social_Network_Ads.csv")
df 


# In[5]:


X=df.iloc[:,:-1].values 
y=df.iloc[:,-1].values 
X 
y 


# In[10]:


from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0) 
print(X_train)
print(X_test) 
print(y_train)
print(y_test)


# In[17]:


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler() 
X_train=sc.fit_transform(X_train) 
X_test=sc.fit_transform(X_test) 
X_train 


# In[19]:


from sklearn.neighbors import KNeighborsClassifier 
classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2) 
classifier.fit(X_train,y_train)


# In[23]:


classifier.predict(sc.transform([[49,36000]])) 


# In[26]:


y_pred=classifier.predict(X_test) 
y_pred 
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[31]:


from sklearn.metrics import confusion_matrix ,accuracy_score 
cm=confusion_matrix(y_test,y_pred) 
print(cm) 
accuracy_score(y_test,y_pred)


# In[32]:



from matplotlib.colors import ListedColormap
plt.figure(figsize=(10,5))
# X_set,y_set repersent train_X and Train_y
X_set, y_set = X_train, y_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
   np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('k', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()


# In[33]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('k', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

