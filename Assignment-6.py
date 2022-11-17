#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing required libraries for assignment 5 here
import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")


# In[4]:


data = pd.read_csv("/Users/jagadeeshreddy/Desktop/CC GENERAL.csv")
data.head()


# In[5]:


data.isnull().any()


# In[6]:


data.fillna(data.mean(), inplace=True)
data.isnull().any()


# In[7]:


x = data.drop('CUST_ID', axis = 1)
print(x)


# In[8]:


#Scaling
scaler = StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)


# In[9]:


#Normalizing the data
X_normalized = normalize(X_scaled_array)
X_normalized = pd.DataFrame(X_normalized)


# In[10]:


#Reducing the dimensionality of the Data
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
principalDf =  pd.DataFrame(data = X_principal, columns = ['principal component1', 'principal component2'])
finalDf = pd.concat([principalDf, data[['TENURE']]], axis = 1)
finalDf.head()


# In[11]:


ac2 = AgglomerativeClustering(n_clusters = 2)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['principal component1'], principalDf['principal component2'],
           c = ac2.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[12]:


ac3 = AgglomerativeClustering(n_clusters = 3)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['principal component1'], principalDf['principal component2'],
           c = ac3.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[13]:


ac4 = AgglomerativeClustering(n_clusters = 4)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['principal component1'], principalDf['principal component2'],
           c = ac4.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[14]:


ac5 = AgglomerativeClustering(n_clusters = 5)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['principal component1'], principalDf['principal component2'],
           c = ac5.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[15]:


k = [2, 3, 4, 5]
 
# Appending the silhouette scores of the different models to the list
silhouette_scores = []
silhouette_scores.append(
        silhouette_score(principalDf, ac2.fit_predict(principalDf)))
silhouette_scores.append(
        silhouette_score(principalDf, ac3.fit_predict(principalDf)))
silhouette_scores.append(
        silhouette_score(principalDf, ac4.fit_predict(principalDf)))
silhouette_scores.append(
        silhouette_score(principalDf, ac5.fit_predict(principalDf)))


# In[16]:


# Plotting a bar graph to compare the results
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 10)
plt.ylabel('Silhouette_scores', fontsize = 10)
plt.show()


# In[ ]:




