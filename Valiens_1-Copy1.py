#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:,.2f}'.format

from scipy.stats import linregress

import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[21]:


df=pd.read_csv("D:\\Export\VA_21Nov.csv",encoding='latin-1')
df1=pd.read_csv("D:\\Export\VA1.csv",encoding='latin-1')


# In[22]:


df.head()


# In[23]:


df1.head()


# In[27]:


df.describe()


# In[28]:


df1.describe()


# In[26]:


df.UID=df.UID.apply(str)
df1.UID=df1.UID.apply(str)


# In[57]:


df.columns


# In[63]:


df['TOTAL_TIME_SPENT_ON_LEVELS_SEC'].sum()


# In[30]:


linregress(df.WINS,df.D1)


# In[32]:


linregress(df1.WINS,df1.DAY1)


# In[36]:


linregress(df1.ATTEMPTS,df1.DAY1)


# In[38]:


linregress(df1.TIME_MIN,df1.DAY1)


# In[47]:


#Mixed Lineer Model
md = smf.mixedlm("DAY1 ~ TIME_MIN", df1, groups=df1["UID"])
mdf=md.fit()
print(mdf.summary())


# In[66]:


from pandas import DataFrame
from sklearn import linear_model
import statsmodels.api as sm
X = df1[['ATTEMPTS',
       'TIME_MIN', 'EXTRA_MOVES_USED', 'EXTRA_MOVES_BOUGHT',
       'BOOSTERS_SELECTED', 'TOOLS_USED', 'RESHUFFLE', 'OBJECTIVES_LEFT',
       'OBJECTIVES_LEFT_AT_1ST_EXTRAMOVES_PURCHASE',
       'OBJECTIVES_LEFT_AT_2ND_EXTRAMOVES_PURCHASE',
       'OBJECTIVES_LEFT_AT_3RD_EXTRAMOVES_PURCHASE']]
Y = df1['DAY1']
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[67]:


model = sm.OLS(Y,X).fit()
predictions = model.predict(X)
print_model = model.summary()
print(print_model)


# In[44]:


categorical_df = df.select_dtypes(include=['object']).copy()
print(categorical_df.head())

print(categorical_df.isnull().sum())

print( max(df.EVENT_DATE), min(df.EVENT_DATE))


# In[34]:


df1.UID.nunique()


# In[35]:


df1.UID.count()


# In[19]:


df["WIN_RATE"]=df.WIN_COUNT/(df.WIN_COUNT+df.LOST_COUNT+df.QUIT_COUNT)
df.WIN_RATE.describe()


# In[22]:


df.WIN_RATE.median()

