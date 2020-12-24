#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("D:/Quarter3/QAM/Project/My project")
from math import sqrt


# In[2]:


Industry = pd.read_csv('10_Industry_Portfolios.csv',skiprows=11)
Industry = Industry.iloc[0:1126,:]

FF_mkt = pd.read_csv('F-F_Research_Data_Factors.csv',skiprows =3)
FF_mkt = FF_mkt.iloc[0:1126,:]


# In[3]:


# Cleaning Industry data
ind_data = Industry.copy()
ind_data['Date'] = pd.to_datetime((ind_data.iloc[:,0].astype(str) + '01'),format="%Y%m%d") 
ind_data['Year'] = pd.DatetimeIndex(ind_data.loc[:,'Date']).year
ind_data['Month'] = pd.DatetimeIndex(ind_data.loc[:,'Date']).month
ind_data = ind_data.loc[((ind_data['Year']>=1989) & (ind_data['Year']<=2019)),:].reset_index(drop=True)
ind_data = ind_data.iloc[8:372,:].reset_index(drop=True)
ind_data


# In[4]:


# Cleaning Market data 
ff_data = FF_mkt.iloc[:,[0,1,4]].copy()
ff_data['Date'] =  pd.to_datetime((ff_data.iloc[:,0].astype(str) + '01'),format="%Y%m%d")
ff_data['Year'] = pd.DatetimeIndex(ff_data.loc[:,'Date']).year
ff_data['Month'] = pd.DatetimeIndex(ff_data.loc[:,'Date']).month
ff_data = ff_data.loc[((ff_data['Year']>=1989) & (ff_data['Year']<=2019)),:].reset_index(drop=True)
ff_data = ff_data.iloc[8:372,:].reset_index(drop=True)
ff_data


# In[5]:


data = pd.merge(ind_data, ff_data, on=["Unnamed: 0","Date","Year","Month"])
data = data.loc[~((data['NoDur'] == '-99.99') & (data['Durbl'] == '-99.99') & (data.Manuf == '-99.99') 
             & (data.Enrgy == '-99.99') & (data.HiTec == '-99.99') & (data.Telcm == '-99.99')
             & (data.Shops == '-99.99') & (data['Hlth '] == '-99.99') & (data.Utils == '-99.99')
             & (data.Other == '-99.99') & (data['Mkt-RF'] == '-99.99') & (data.RF == '-99.99') ),:]
data = data.iloc[:,[0,11,12,13,1,2,3,4,5,6,7,8,9,10,14,15]]
data_means = data.iloc[:,4:16]
data_means = data_means.apply(pd.to_numeric)

# New temp
temp = pd.DataFrame({"ExNoDur": data_means['NoDur'] - data_means['RF'],
                     "ExDurbl": data_means['Durbl'] - data_means['RF'], 
                     "ExManuf": data_means['Manuf'] - data_means['RF'],
                     "ExEnrgy": data_means['Enrgy'] - data_means['RF'], 
                     "ExHiTec": data_means['HiTec'] - data_means['RF'],
                     "ExTelcm": data_means['Telcm'] - data_means['RF'], 
                     "ExShops": data_means['Shops'] - data_means['RF'],
                     "ExHlth": data_means['Hlth '] - data_means['RF'], 
                     "ExUtils": data_means['Utils'] - data_means['RF'],
                     "ExOther": data_means['Other'] - data_means['RF'], 
                     "Mkt-RF":data_means['Mkt-RF']})

temp['Mean_rows'] = temp.mean(axis=1)

data = pd.concat([data.loc[:,["Date","Year","Month"]], (temp/100)],axis=1)


# In[6]:


#EW 
mean_total_ew = data['Mean_rows'].mean()
sd_total_ew = data['Mean_rows'].std()
Sharpe_Ratio_total_ew =  mean_total_ew/sd_total_ew
CEQ_total_ew = mean_total_ew - 0.5*((sd_total_ew)**2)
turnover = []
turnover.insert(0,0)
temp_outsample = data.iloc[:,3:14]
for i in range(1,temp_outsample['ExNoDur'].count()):
    turnover.append(sum(abs((1/11) - ((1/11) * (1+temp_outsample.iloc[i,:]))/ (1 + sum((1/11)* temp_outsample.iloc[i,:])))))
Turnover_ew = np.mean(turnover)


# In[7]:


#VW 
mean_total_vw = data['Mkt-RF'].mean()
sd_total_vw = data['Mkt-RF'].std()
Sharpe_Ratio_total_vw = mean_total_vw/sd_total_vw
CEQ_total_vw = mean_total_vw - 0.5*(sd_total_vw**2)
Turnover_vw = 0


# In[8]:


#MVE in sample
sigma = np.cov(temp_outsample.T)
mu = temp_outsample.mean()
xt = np.matmul(np.linalg.inv(sigma),mu)
weights = xt/sum(xt)
ret = pd.DataFrame(weights*temp_outsample).sum(axis=1)
mean_MVE_in = ret.mean()
sd_MVE_in = ret.std()
Sharpe_MVE_in = mean_MVE_in/sd_MVE_in
CEQ_MVE_in = mean_MVE_in - 0.5*(sd_MVE_in**2)
Turnover_MVE_in = np.nan


# In[9]:


#MVE Out of sample test
ret = []
w = 0
turnover = []
for i in range(60,temp_outsample['ExNoDur'].count()):
    sigma = np.cov(temp_outsample.iloc[i-60:i,:].T)
    mu = temp_outsample.iloc[i-60:i,:].mean()   
    xt = np.matmul(np.linalg.inv(sigma),mu)
    weights = xt/sum(xt)
    ret.append(weights * temp_outsample.iloc[i,:].values)
    if  (i-60 == 0):
        turnover.append(0)
    else:
        turnover.append(sum(abs(weights - (w * (1+temp_outsample.iloc[i-1,:]))/ (1 + sum(weights * temp_outsample.iloc[i-1,:])))))
    w = weights 
ret = pd.DataFrame(ret,columns = temp_outsample.columns)
ret_MVE_out = ret.sum(axis=1)
mean_MVE_out = ret_MVE_out.mean()
sd_MVE_out = ret_MVE_out.std()
Sharpe_MVE_out =  mean_MVE_out/sd_MVE_out
CEQ_MVE_out =  mean_MVE_out - 0.5*(sd_MVE_out**2)
Turnover_MVE_out = np.mean(turnover)


# In[10]:


#Shrunk MVE Out of sample test
ret_shrink = []
w=0
tunrover= []
for i in range(60,temp_outsample['ExNoDur'].count()):
    returns = temp_outsample.iloc[i-60:i,:]
    T = returns['ExNoDur'].count()
    
    # Shrink covariance matrix 
    S = np.cov(returns.T)
    target= np.mean(np.diag(S))*np.eye(S.shape[1]) 
    f = lambda row: (((returns.iloc[row,:] @ returns.iloc[row,:].T) - S)**2).sum()
    omega2 = np.nanmean([f(x) for x in range(T)])/(T-1)
    total_var = sum(sum((S - target)**2))
    delta2 = total_var - omega2
    beta = max((delta2/total_var), 0)
    Sigma_hat = (1 - beta) * target + beta * S
    sigma = Sigma_hat
    
    # Shrink returns
    m_i = returns.mean()
    shrinkage_target = m_i.mean()
    #beta_denominator =  m_i.var()
    #beta_numerator = (beta_denominator - np.mean((returns.std()/sqrt(returns.shape[0]))**2)) if (beta_denominator - np.mean((returns.std()/sqrt(returns.shape[0]))**2))<0 else 0
    omega2 = (returns.var()/T).mean() if (returns.var()/T).mean()>0 else 0
    total_var = m_i.var()
    delta2 = total_var - omega2
    beta = max((delta2/total_var), 0 )
    mu = (1-beta)*shrinkage_target + beta*m_i
    xt = np.matmul(np.linalg.inv(sigma),mu)
    weights = xt/sum(xt)
    ret_shrink.append(weights * temp_outsample.iloc[i,:].values)
    if  (i-60 == 0):
        turnover.append(0)
    else:
        turnover.append(sum(abs(weights - (w * (1+temp_outsample.iloc[i-1,:]))/ (1 + sum(weights * temp_outsample.iloc[i-1,:])))))
    w = weights 
ret_shrink = pd.DataFrame(ret_shrink,columns = temp_outsample.columns)
shrink_ret_MVE_out = ret_shrink.sum(axis=1)
mean_shrink_MVE_out = shrink_ret_MVE_out.mean()
sd_shrink_MVE_out = shrink_ret_MVE_out.std()
Sharpe_shrink_MVE_out =  mean_shrink_MVE_out/sd_shrink_MVE_out
CEQ_shrink_MVE_out =  mean_shrink_MVE_out - 0.5*(sd_shrink_MVE_out**2)
Turnover_shrink_MVE_out = np.mean(turnover)


# In[11]:


#Risk Parity
ret = []
w = 0
turnover = []
for i in range(60,temp_outsample['ExNoDur'].count()):
    sigma = np.cov(temp_outsample.iloc[i-60:i,:].T)
    diag_sigma = np.diag(sigma)  
    xt = 1/diag_sigma
    weights = xt/sum(xt)
    ret.append(weights * temp_outsample.iloc[i,:].values)
    if  (i-60 == 0):
        turnover.append(0)
    else:
        turnover.append(sum(abs(weights - (w * (1+temp_outsample.iloc[i-1,:]))/ (1 + sum(weights * temp_outsample.iloc[i-1,:])))))
    w = weights 
ret = pd.DataFrame(ret,columns = temp_outsample.columns)
ret_rp = ret.sum(axis=1)
mean_rp = ret_rp.mean()
sd_rp = ret_rp.std()
Sharpe_rp =  mean_rp/sd_rp
CEQ_rp =  mean_rp - 0.5*(sd_rp**2)
Turnover_rp = np.mean(turnover)


# In[12]:


total = pd.DataFrame(np.array([[mean_total_ew,mean_MVE_in, mean_MVE_out, mean_shrink_MVE_out ,mean_total_vw, mean_rp],
                               [sd_total_ew, sd_MVE_in, sd_MVE_out, sd_shrink_MVE_out,sd_total_vw, sd_rp],
                               [Sharpe_Ratio_total_ew, Sharpe_MVE_in,Sharpe_MVE_out,Sharpe_shrink_MVE_out, Sharpe_Ratio_total_vw, Sharpe_rp],
                               [CEQ_total_ew, CEQ_MVE_in, CEQ_MVE_out,CEQ_shrink_MVE_out, CEQ_total_vw,CEQ_rp],
                               [Turnover_ew, Turnover_MVE_in, Turnover_MVE_out, Turnover_shrink_MVE_out, Turnover_vw, Turnover_rp]]),
                    columns = ["EW","MVE Insample","MVE Outsample","Shrunk MVE Outsample","VW","RP"])#,
                    #rows = ["Mean","SD","Sharpe Ratio","CEQ","Turnover"])


# In[13]:


total = total.T


# In[14]:


total.columns = ["Mean","SD","Sharpe Ratio","CEQ","Turnover"]


# In[15]:


total


# In[ ]:





# In[ ]:




