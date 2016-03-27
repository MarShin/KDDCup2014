import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression

print('loading data..')
# Both Train / Test
projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
resources = pd.read_csv('../COMP4332_Dataset/resources.csv')
essays = pd.read_csv('../COMP4332_Dataset/essays.csv')

# Only Train
outcomes = pd.read_csv('../COMP4332_Dataset/outcomes.csv')
donations = pd.read_csv('../COMP4332_Dataset/donations.csv')

print('complete')

print 'projects', projects.shape
print 'essays', essays.shape
print 'resources', resources.shape
print 'donations', donations.shape
print 'outcomes', outcomes.shape

#projects = projects.sort('projectid')
#essays = essays.sort('projectid')
#resources = resources.sort('projectid')

#projects.to_csv('../COMP4332_Dataset/projects_sorted.csv')
#essays.to_csv('../COMP4332_Dataset/essays_sorted.csv')
#projects.to_csv('projects_sorted.csv')
#outcomes = outcomes.sort('projectid')
