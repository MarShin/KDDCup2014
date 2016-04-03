# IPython log file

get_ipython().magic('logstart')
import pandas as pd
import numpy as np
import csv
essays = pd.read_csv('../COMP4332_Dataset/essays.csv')
projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
outcomes = pd.read_csv('../COMP4332_Dataset/outcomes.csv')
projects = projects.sort_values('projectid')
outcomes = outcomes.sort_values('projectid')
essays = essays.sort_values('projectid')
dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]
labels = np.array(outcomes.is_exciting)
print (labels)
print (labels.shape)
print (train_idx.shape)
for i in range(len(train_idx)):
    train_dat = pd.append(projects[i])
    
for i in range(len(train_idx)):
    train_dat = pd.DataFrame.append(projects[i])
    
for i in range(len(train_idx)):
    train_dat[i] = pd.DataFrame.append(projects[i])
    
train_dat = DataFrame(columns=('school_latitude', 'school_longitude','fulfillment_labor_materials','total_price_excluding_optional_support','total_price_including_optional_support', 'projectid', 'teacher_acctid', 'schoolid', 'school_ncesid',list(set(projects.columns).difference(set(projects_numeric_columns)).difference(set(projects_id_columns)).difference(set(['date_posted'])))))
res = DataFrame(columns=('ab'))
train_dat = pd.DataFrame(columns=('school_latitude', 'school_longitude','fulfillment_labor_materials','total_price_excluding_optional_support','total_price_including_optional_support', 'projectid', 'teacher_acctid', 'schoolid', 'school_ncesid',list(set(projects.columns).difference(set(projects_numeric_columns)).difference(set(projects_id_columns)).difference(set(['date_posted'])))))
projects.header_list
projects.columns.values
list(projects.columns.values)
res = DataFrame(columns= list(projects.columns.values))
res = pd.DataFrame(columns= list(projects.columns.values))
res.columns.values
train_dat = pd.DataFrame(columns = list(projects.columns.values))
test_dat = pd.DataFrame(columns = list(projects.columns.values))
for i in train_idx:
    train_dat.append(projects[train_idx])
    
for i in train_idx:
    train_dat.append(projects[i])
    
projects[0]
projects[0]
for i in train_idx:
    train_dat.add(projects[i])
    
projects.loc[0]
for i in train_idx:
    train_dat.append(projects.loc[train_idx])
    
print (train_dat)
for i in train_idx:
    train_dat.append(projects.loc[i])
    
print(train_dat)
for i in train_idx:
    train_dat.append(projects.loc[i])
    
