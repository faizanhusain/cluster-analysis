#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Sparks Foundation
# Data science and Business Analytics Intern 
# Task 2: predict the optimum number of clusters and represent it visually.
# 

# # Impoting all necessery libraries

# In[3]:


#importing all the necessary modules
import pandas as pd #reading the csv file and creating a dataframe
import numpy as np
import matplotlib.pyplot as plt     #for plotting data from url and trained data
import seaborn as sns


# In[14]:


data= pd.read_csv("C:\\Users\\harishcomputer1\\Downloads\\Iris.csv")
data.head()


# In[16]:


data.tail()


# # Step2: Data visualization

# In[17]:


data.isnull()


# In[20]:


#we can also see, whether there any null value or not with the help of heatmap
sns.heatmap(data.isnull(), yticklabels= False, cbar= False)


# In[21]:


data.corr()


# In[22]:


#Visualize the correlation by creating a heat map
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot= True, fmt='.0%')


# # Step 3: Preprocessing

# In[25]:


x=data.drop(columns=['Id','Species'], axis=1)
y= data.Species
from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
y=encoder.fit_transform(y)
y


# In[26]:


x.head()


# let's see here how the k means clustering works:
# 
# Kmeans itself means that 'K' no of centroids. The steps to implement are:
# 
# step1: Iniyially take some k no of centroids
# Step2: Initialize them
# Step3: create cluster by grouping all the nearest points to these centroids respectively
# Step4: Now get the mean of the each cluster created and update the centroids to mean value till no future movements of centroid possible
# Step5: Get the optimized/best value of k perform elbow method. where the plot is between wcss(with cluster sum of squres) and k value. from this plot the k value where the abrupt decrease/ elbow shape starts that is taken as optimum value of k

# In[27]:


from sklearn.cluster import KMeans
wcss=[]


# In[29]:


for i in range(1,10):
    kmeans= KMeans(n_clusters = i, random_state=1)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    


# In[30]:


sns.set_style("darkgrid")
plt.plot(range(1,10), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')  #with cluter sum of square
plt.show()


# here we can see that the elbow curve start at k=3 and therefore optimum number of k is 3

# In[31]:


model=KMeans(n_clusters=3, random_state=1)
y_pred = model.fit_predict(x)
x=x.values


# In[36]:


# visualising the cluster on the first two columns
plt.scatter(x[y_pred==0,0],x[y_pred==0,1], s= 100, c='magenta', label="Iris-setosa")
plt.scatter(x[y_pred==1,0], x[y_pred==1,1], s=100, c= 'blue', label= 'Iris-versicolour')
plt.scatter(x[y_pred==2,0], x[y_pred==2,1], s=100, c= 'green',label='Iris-verginica' )

#ploting the centroids of the clusters
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=100, c='black',label='Centroids')

plt.legend()
plt.show()


# In[ ]:




