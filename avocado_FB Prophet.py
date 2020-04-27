#!/usr/bin/env python
# coding: utf-8

# In[20]:


from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv("C:/Users/nikhi/Desktop/python/avocado.csv")


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.describe(include='O')


# In[13]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.iloc[:,10] = le.fit_transform(df.iloc[:,10])
df.head(2)


# In[12]:


df['Date'] = pd.to_datetime(df['Date'])


# In[13]:


df = df.sort_values("Date")


# In[14]:


df['Date'].head(),df['Date'].tail()


# In[17]:


plt.figure(figsize=(22,7))
plt.plot(df['Date'], df['AveragePrice'])


# In[27]:


plt.figure(figsize=(15,5))
plt.title("Distribution Price")
ax = sns.distplot(df["AveragePrice"], color = 'green')


# In[28]:


plt.figure(figsize=[8,5])
sns.countplot(x = 'year', data = df)


# In[26]:


X= df[['Date','Total Volume', '4046', '4225', '4770',
       'Small Bags', 'Large Bags', 'XLarge Bags', 'type']]
y= df.iloc[:,1]


# In[33]:


from sklearn.feature_selection import mutual_info_regression
y=df['AveragePrice']


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)


# In[28]:


X_train.head(2)


# In[34]:



train_dataset= pd.DataFrame()
train_dataset['ds'] = pd.to_datetime(X["Date"])
train_dataset['y']=y
train_dataset.head(2)


# In[35]:


prophet_basic = Prophet()
prophet_basic.fit(train_dataset)


# In[36]:


future= prophet_basic.make_future_dataframe(periods=300)
future.tail(2)


# In[37]:


forecast=prophet_basic.predict(future)


# In[39]:


df['yhat'] = forecast['yhat']
df['AveragePrice'].plot(legend=True)
df['yhat'].plot(legend= True)


# In[40]:



fig1 =prophet_basic.plot(forecast)


# In[41]:


fig1 = prophet_basic.plot_components(forecast)


# In[ ]:




