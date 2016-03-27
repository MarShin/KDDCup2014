# coding: utf-8
projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
import pandas as pd
projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
outcomes = pd.read_csv('../COMP4332_Dataset/outcomes.csv')
outcomes.head(20)
projects = projects.sort('projectid')
outcomes = outcomes.sort('projectid')
essays = pd.read_csv('../COMP4332_Dataset/essays.csv')
totalCount = projects.shape[0]
print  totalCount
print(totalCount)
for i in range(1,projects.shape[1]):
    nullcount = projects[projects[projects.columns[i]].isnull()].shape[0]
    percentage = float(nullcount)/float(totalCount) *100
    if(percentage>0):
        print(projects.columns[i],percentage,'%')
        
for i in range(1,projects.shape[1]):
    if (nullcount = projects[projects[projects.columns[i]].isnull()].shape[0]):
for i in range(1,projects.shape[1]):
    if (projects[projects[projects.columns[i]].isnull()].shape[0]):
        print(projects[projects.columns[i]])
        
projects.head(30)
get_ipython().magic('history ')
get_ipython().magic('history > history_for_print.txt')
get_ipython().magic('history ')
get_ipython().magic('save history')
get_ipython().magic('save')
