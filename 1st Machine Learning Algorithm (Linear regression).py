#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


# # 1. Dataset

# In[2]:


x, y = make_regression(n_samples = 100, n_features = 1 , noise = 10)


# In[3]:


plt.scatter(x,y)


# In[6]:


print(x.shape)
y = y.reshape(y.shape[0], 1)
print(y.shape)


# In[11]:


#matrice X
X = np.hstack((x, np.ones(x.shape)))
X.shape


# In[12]:


X


# In[13]:


theta = np.random.randn(2,1)
theta


# # 2. Modele

# In[17]:


def model(X,theta):
    return X.dot(theta)


# In[18]:


model(X,theta)


# In[21]:


plt.scatter(x,y)
plt.plot(x,model(X,theta), c='r')


# # 3. Fonction coût

# In[22]:


def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta)-y)**2)


# In[23]:


cost_function(X, y, theta)


# # 4. Gradient et Descente de Gradient

# In[31]:


def Gradient(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)   


# In[32]:


Gradient(X, y, theta)


# In[43]:


def Gradient_descent(X, y, theta, learning_rate, n_iteration):
    cost_history = np.zeros(n_iteration)
    for i in range(0, n_iteration):
        theta = theta - learning_rate*Gradient(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
        
    return theta, cost_history
    


# # Machine learning !

# In[44]:


theta_final, cost_history = Gradient_descent(X, y, theta, learning_rate=0.01, n_iteration=1000)
theta_final


# In[48]:


predictions = model(X, theta_final)
plt.scatter(x, y)
plt.plot(x, predictions, c="r")


# In[46]:


plt.plot(range(1000), cost_history)


# In[47]:


def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - (u/v)


# In[49]:


coef_determination(y, predictions)


# # Conclusion : notre algorithme a un score de 89,75 % et un taux d'apprentissage qui tend vers 0 à partir de la 300 ième itération 

# In[ ]:




