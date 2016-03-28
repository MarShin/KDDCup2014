# coding=utf-8
import csv
import pandas as pd
import numpy as np

# load the data
print('loading the data...')
projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
print('complete..')

# sort the data based on id
projects = projects.sort_values('projectid')

# split the training data and testing data
dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

# fill the missing data
#projects = projects.fillna(method='pad') # fill the missing hole with the previous observation data


# separate train / test dataset
projects_train = projects.loc[train_idx]
projects_test = projects.loc[test_idx]

#check data size
print("projects_train size: ", projects_train.shape)
print("projects_test size: ", projects_test.shape)

#write to file
projects_train.to_csv('projects_train.csv',encode='utf-8',index=False)
projects_test.to_csv('projects_test.csv',encode='utf-8',index=False)
