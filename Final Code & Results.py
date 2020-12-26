#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd, YearEnd
import os
os.chdir("D:/Quarter3/QAM/Project/My project")


# In[2]:


Industry = pd.read_csv('10_Industry_Portfolios.csv',skiprows=11)
Industry = Industry.iloc[0:1126,:]

FF_mkt = pd.read_csv('F-F_Research_Data_Factors.csv',skiprows =3)
FF_mkt = FF_mkt.iloc[0:1126,:]


# In[3]:


# Record CRISP unkowns
unknowns = ["-66.0", "-77.0", "-88.0", "-99.0", "-99.99", "-999", "A", "B", "C", "D", "E", "S", "T", "P"]

# Create function to convert CRISP unkowns to np.nan
convert_unknows = np.vectorize(lambda x: np.nan if x in unknowns else x)

# Convert to decimal
Industry.iloc[:, 1:] = Industry.iloc[:, 1:].apply(convert_unknows, axis = 0).astype(float).div(100)
FF_mkt.iloc[:, 1:] = FF_mkt.iloc[:, 1:].apply(convert_unknows, axis = 0).astype(float).div(100)

# Rename data columns
Industry.rename(columns = {"Unnamed: 0":"date"}, inplace = True)
FF_mkt.rename(columns = {"Unnamed: 0":"date"}, inplace = True)

# Convert date column
Industry['date'] = pd.to_datetime(Industry['date'], format = "%Y%m")
FF_mkt['date'] = pd.to_datetime(FF_mkt['date'], format = "%Y%m")

Industry['Year'] = pd.DatetimeIndex(Industry.loc[:,'date']).year
Industry['Month'] = pd.DatetimeIndex(Industry.loc[:,'date']).month
FF_mkt['Year'] = pd.DatetimeIndex(FF_mkt.loc[:,'date']).year
FF_mkt['Month'] = pd.DatetimeIndex(FF_mkt.loc[:,'date']).month


# In[4]:


Industry = Industry.loc[((Industry['Year']>=1989) & (Industry['Year']<=2019)),:].reset_index(drop=True)
Industry = Industry.iloc[8:372,:].reset_index(drop=True)
FF_mkt = FF_mkt.loc[((FF_mkt['Year']>=1989) & (FF_mkt['Year']<=2019)),:].reset_index(drop=True)
FF_mkt = FF_mkt.iloc[8:372,:].reset_index(drop=True)


# In[5]:


FF_mkt


# In[13]:


data = pd.merge(Industry, FF_mkt, on=["date","Year","Month"])
data = data.iloc[:,[0,11,12,1,2,3,4,5,6,7,8,9,10,13,16]]
data_means = data.iloc[:,3:15]
data_means = data_means.apply(pd.to_numeric)
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
data = pd.concat([data.loc[:,["date","Year","Month"]], temp],axis=1)
data


# In[59]:


def get_info(data, M):
    
    # define X
    X = data.iloc[:,3:15]
    #X = data.loc[:,  data.columns != "date"]
    
    # define N
    N = len(X.columns)
    
    # Define table
    table = pd.DataFrame(0, columns = ['mean', 'std', 'Sharpe', 'ceq', 'turn_over'], index = ['ew', 'vw', 'mve_in', 'mve_out', 'shrink_mve_out', 'rp'])
    
    # ==== Equal weighting ====
    table.loc['ew', 'mean'] = X.apply(np.mean, axis = 1).mean()
    table.loc['ew', 'std'] = X.apply(np.mean, axis = 1).std()
    table.loc['ew', 'Sharpe'] = table.loc['ew', 'mean']/table.loc['ew', 'std']
    table.loc['ew', 'ceq'] = table.loc['ew', 'mean'] - 0.5 * table.loc['ew', 'std']**2

    # Get turn_over
    turn_over = np.zeros(len(X))
    for i in range(len(X)):
        if i == 0:
            turn_over[i] = 1
        else:
            R = 1 + X.iloc[i - 1, :]
            turn_over[i] = 1/N * np.absolute(1 - R/R.mean()).sum()
    
    table.loc['ew', 'turn_over'] = turn_over.mean()
    
    
    # === Value weighting ===
    table.loc['vw', 'mean'] = X['Mkt-RF'].mean()
    table.loc['vw', 'std'] = X['Mkt-RF'].std()
    table.loc['vw', 'Sharpe'] = table.loc['vw', 'mean']/table.loc['vw', 'std']
    table.loc['vw', 'turn_over'] = 0
    table.loc['vw', 'ceq'] = table.loc['vw', 'mean'] - 0.5 * table.loc['vw', 'std']**2
    
    # === MVE in-sample === 
    Sigma = X.cov()
    mu = X.mean()
    w = np.linalg.pinv(Sigma) @ mu/(np.ones(N).T @ np.linalg.pinv(Sigma) @ mu)
    
    if w @ mu > 0:
        table.loc['mve_in', 'mean'] = w @ mu
        table.loc['mve_in', 'std'] = np.sqrt(w.T @ Sigma @ w)
        table.loc['mve_in', 'Sharpe'] = w @ mu/np.sqrt(w @ Sigma @ w.T)
    else:
        table.loc['mve_in', 'mean'] = -w @ mu
        table.loc['mve_in', 'std'] = np.sqrt(w.T @ Sigma @ w)
        table.loc['mve_in', 'Sharpe'] = -w @ mu/np.sqrt(w @ Sigma @ w.T)
            
    table.loc['mve_in', 'turn_over'] = np.nan
    table.loc['mve_in', 'ceq'] = table.loc['mve_in', 'mean'] - 0.5 * table.loc['mve_in', 'std']**2
    
    # === MVE out-sample ===
    outsample = np.zeros(len(X) - M)
    turn_over = np.zeros(len(X) - M)
    w = np.zeros(N)

    for i in range(M, len(X)):
        Sigma = X.iloc[(i - M):i, :].cov()
        mu = X.iloc[(i - M):i, :].mean()
        w_new = np.linalg.pinv(Sigma) @ mu/(np.ones(N).T @ np.linalg.pinv(Sigma) @ mu)
        R = 1 + X.iloc[i - 1, :]
        turn_over[i - M] = np.sum(np.absolute(w_new - (w * R)/(w @ R)))
        w = w_new
        if w @ mu > 0:
            outsample[i - M] = X.iloc[i, :] @ w
        else:
            outsample[i - M] = -X.iloc[i, :] @ w
    
    table.loc['mve_out', 'mean'] = np.mean(outsample)
    table.loc['mve_out', 'std'] = np.std(outsample)
    table.loc['mve_out', 'Sharpe'] = np.mean(outsample)/np.std(outsample)
    table.loc['mve_out', 'turn_over'] = turn_over.mean()
    table.loc['mve_out', 'ceq'] = table.loc['mve_out', 'mean'] - 0.5 * table.loc['mve_out', 'std']**2
    
    # === Shrink MVE out-of-sample === 
    outsample = np.zeros(len(X) - M)
    turn_over = np.zeros(len(X) - M)
    w = np.zeros(N)

    for i in range(M, len(X)):     
        # Record sample covariance matrix
        S = X.iloc[(i - M):i, :].cov()
        
        # Define target matrix
        target = np.mean(np.diag(S)) * np.eye(N)
        
        # Define function to help compute omega2
        f = lambda row: ((X.iloc[row, :] @ X.iloc[row, :].T - S)**2).sum()
  
        # Compute non-idiosyncratic variance of variance
        omega2 = np.nanmean([f(x) for x in range(M + 1)])/(M - 1)
  
        # Calculate total variation of variance
        total_var = ((S - target)**2).sum().sum()
  
        # Calculate idiosyncratic variance of variance
        delta2 = total_var - omega2
  
        # Compute shrinkage parameter
        beta = np.max([delta2/total_var, 0])
  
        # Get Sigma_hat
        Sigma = (1 - beta) * target + beta * S
        
        # Record sample mean
        m = X.iloc[(i - M):i, :].mean()
        
        # Record target
        target = np.mean(m)
        
        # Calculate variance of mean estimate
        omega2 = (X.iloc[(i - M):i, :].var()/M).mean()
        
        # Calculate total variance
        total_var = np.mean(m.sub(target)**2)
        
        # Calculate idiosyncratic variance
        delta2 = total_var - omega2
        
        # Compute shrinkage parameter
        beta = np.max([delta2/total_var, 0])
        
        # Compute mu estimate
        mu = (1 - beta) * target + beta * m
    
        w_new = np.linalg.pinv(Sigma) @ mu/(np.ones(N).T @ np.linalg.pinv(Sigma) @ mu)
        R = 1 + X.iloc[i - 1, :]
        turn_over[i - M] = np.sum(np.absolute(w_new - (w * R)/(w @ R)))
        
        w = w_new
        if w @ mu > 0:
            outsample[i - M] = X.iloc[i, :] @ w
        else:
            outsample[i - M] = -X.iloc[i, :] @ w
    
    table.loc['shrink_mve_out', 'mean'] = np.mean(outsample)
    table.loc['shrink_mve_out', 'std'] = np.std(outsample)
    table.loc['shrink_mve_out', 'Sharpe'] = np.mean(outsample)/np.std(outsample)
    table.loc['shrink_mve_out', 'turn_over'] = turn_over.mean()
    table.loc['shrink_mve_out', 'ceq'] = table.loc['shrink_mve_out', 'mean'] - 0.5 * table.loc['shrink_mve_out', 'std']**2
      
    # === RP ===
    returns = np.zeros(len(X) - M)
    turn_over = np.zeros(len(X) - M)
    w = np.zeros(N)
    
    for i in range(M, len(X)):
        sigma = np.sqrt(np.diag(X.iloc[(i - M):i, :].cov()))
        w_new = (1/sigma)/np.sum(1/sigma)
        returns[i - M] = w_new.T @ X.iloc[i, :]
        R = 1 + X.iloc[i - 1, :]
        turn_over[i - M] = np.sum(np.absolute(w_new - (w * R)/(w @ R)))
        w = w_new
        
    table.loc['rp', 'mean'] = np.mean(returns)
    table.loc['rp', 'std'] = np.std(returns)
    table.loc['rp', 'Sharpe'] = np.mean(returns)/np.std(returns)
    table.loc['rp', 'turn_over'] = turn_over.mean()
    table.loc['rp', 'ceq'] = table.loc['rp', 'mean'] - 0.5 * table.loc['rp', 'std']**2
    
    return(table)


# In[60]:


result_Industry = get_info(data, 60)


# In[61]:


result_Industry


# # Charles Code

# In[57]:


os.chdir("D:/Quarter3/QAM/Project")
# FF25 portfolio
FF25 = pd.read_csv("FF_25_Portfolios.csv", nrows = 1127)

## Fama-French 3 factor
FF3 = pd.read_csv("FF3.csv")

## Fama-French mom
FFmom = pd.read_csv("FF_Mom.csv")


# Record CRISP unkowns
unknowns = ["-66.0", "-77.0", "-88.0", "-99.0", "-99.99", "-999", "A", "B", "C", "D", "E", "S", "T", "P"]

# Create function to convert CRISP unkowns to np.nan
convert_unknows = np.vectorize(lambda x: np.nan if x in unknowns else x)

# Convert to decimal
FF25.iloc[:, 1:] = FF25.iloc[:, 1:].apply(convert_unknows, axis = 0).astype(float).div(100)
FF3.iloc[:, 1:] =FF3.iloc[:, 1:].apply(convert_unknows, axis = 0).astype(float).div(100)
FFmom.iloc[:, 1] = FFmom.iloc[:, 1].apply(convert_unknows).div(100)

# Rename data columns
FF25.rename(columns = {"Unnamed: 0":"date"}, inplace = True)
FF3.rename(columns = {"Unnamed: 0":"date"}, inplace = True)
FFmom.rename(columns = {"Unnamed: 0":"date"}, inplace = True)

# Convert date column
FF25['date'] = pd.to_datetime(FF25['date'], format = "%Y%m")
FF3['date'] = pd.to_datetime(FF3['date'], format = "%Y%m") 
FFmom['date'] = pd.to_datetime(FFmom['date'], format = "%Y%m") 

# Remove the top 5 ME quintiles
ME = []
for i in range(len(FF25.columns)):  
    if FF25.columns[i][:3] != 'ME5':  
        ME.append(i)

FF25 = FF25.iloc[:, ME]
FF25.drop(['BIG LoBM', 'BIG HiBM'], axis = 1, inplace = True)

col4 = FF3.drop('RF', axis = 1)
col5 = FF25.merge(FF3[['date', 'Mkt-RF', 'RF']], on = 'date')
col6 = FF25.merge(FF3, on = 'date' ).merge(FFmom, on = 'date')

col5.iloc[:, 1:21] = col5.iloc[:, 1:21].sub(col5['RF'], axis = 0)
col6.iloc[:, 1:21] = col6.iloc[:, 1:21].sub(col6['RF'], axis = 0)
    
col5.drop('RF', axis = 1, inplace = True)
col6.drop('RF', axis = 1, inplace = True)

start = pd.to_datetime("19899", format = "%Y%m")
end = pd.to_datetime("201912", format = "%Y%m")

col4 = col4.loc[(col4.date >= start) & (col4.date <= end)]
col5 = col5.loc[(col5.date >= start) & (col5.date <= end)]
col6 = col6.loc[(col6.date >= start) & (col6.date <= end)]


# In[58]:


result_mkt_smb_hml = get_info(col4, 60)
result_ff1 = get_info(col5, 60)
result_ff4 = get_info(col6, 60)


# In[62]:


result_Industry


# In[63]:


result_mkt_smb_hml


# In[64]:


result_ff1


# In[65]:


result_ff4


# # Tables

# In[66]:


table = pd.DataFrame(0, columns = ['Industry Portfolios', 'Mkt/SMB/HML', 'FF 1-factor', 'FF 4-factor'], index = ['1/N', 'mve_in', 'mve_out', 'shrink_mve_out', 'vw','rp'])


# In[70]:


tickers = [result_Industry, result_mkt_smb_hml, result_ff1, result_ff4 ]


# In[71]:


tickers


# In[87]:


expected_returns = pd.DataFrame([result_Industry.iloc[:,0] , result_mkt_smb_hml.iloc[:,0],  result_ff1.iloc[:,0],  result_ff4.iloc[:,0]]).T
expected_returns.index = ['1/N', 'mve_in', 'mve_out', 'shrink_mve_out', 'vw','rp']
expected_returns.columns = ['Industry Portfolios', 'Mkt/SMB/HML', 'FF 1-factor', 'FF 4-factor']
expected_returns


# In[86]:


std_dev = pd.DataFrame([result_Industry.iloc[:,1] , result_mkt_smb_hml.iloc[:,1],  result_ff1.iloc[:,1],  result_ff4.iloc[:,1]]).T
std_dev.index = ['1/N', 'mve_in', 'mve_out', 'shrink_mve_out', 'vw','rp']
std_dev.columns = ['Industry Portfolios', 'Mkt/SMB/HML', 'FF 1-factor', 'FF 4-factor']
std_dev


# In[88]:


sharpe = pd.DataFrame([result_Industry.iloc[:,2] , result_mkt_smb_hml.iloc[:,2],  result_ff1.iloc[:,2],  result_ff4.iloc[:,2]]).T
sharpe.index = ['1/N', 'mve_in', 'mve_out', 'shrink_mve_out', 'vw','rp']
sharpe.columns = ['Industry Portfolios', 'Mkt/SMB/HML', 'FF 1-factor', 'FF 4-factor']
sharpe


# In[89]:


ceq = pd.DataFrame([result_Industry.iloc[:,3] , result_mkt_smb_hml.iloc[:,3],  result_ff1.iloc[:,3],  result_ff4.iloc[:,3]]).T
ceq.index = ['1/N', 'mve_in', 'mve_out', 'shrink_mve_out', 'vw','rp']
ceq.columns = ['Industry Portfolios', 'Mkt/SMB/HML', 'FF 1-factor', 'FF 4-factor']
ceq


# In[90]:


turnover = pd.DataFrame([result_Industry.iloc[:,4] , result_mkt_smb_hml.iloc[:,4],  result_ff1.iloc[:,4],  result_ff4.iloc[:,4]]).T
turnover.index = ['1/N', 'mve_in', 'mve_out', 'shrink_mve_out', 'vw','rp']
turnover.columns = ['Industry Portfolios', 'Mkt/SMB/HML', 'FF 1-factor', 'FF 4-factor']
turnover


# In[ ]:




