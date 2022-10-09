#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/wikibook/machine-learning/master/data/csv/basketball_stat.csv")


# In[2]:


df.head()


# In[5]:


df.Pos.value_counts()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df.drop(['2P','AST','STL'], axis=1, inplace=True)


# In[9]:


df.head()


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


train, test = train_test_split(df, test_size=0.2)


# In[13]:


train.shape[0]


# In[14]:


test.shape[0]


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[16]:


max_k_range = train.shape[0] // 2
k_list = []
for i in range(3, max_k_range, 2):
    k_list.append(i)

cross_validation_scores = []
x_train = train[['3P', 'BLK' , 'TRB']]
y_train = train[['Pos']]

# 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train.values.ravel(),
                             cv=10, scoring='accuracy')
    cross_validation_scores.append(scores.mean())

cross_validation_scores


# In[17]:


plt.plot(k_list, cross_validation_scores)
plt.xlabel('the number of k')
plt.ylabel('Accuracy')
plt.show()


# In[18]:


# find best k
cvs = cross_validation_scores
k = k_list[cvs.index(max(cross_validation_scores))]
print("The best number of k : " + str(k) )


# In[ ]:




