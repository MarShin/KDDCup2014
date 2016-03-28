# coding=utf-8
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# load the data
print('loading the data...')
projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
outcomes = pd.read_csv('../COMP4332_Dataset/outcomes.csv')
essays = pd.read_csv('../COMP4332_Dataset/essays.csv')
print('complete..')

# sort the data based on id
projects = projects.sort_values('projectid')
outcomes = outcomes.sort_values('projectid')
essays = essays.sort_values('projectid')

# split the training data and testing data
dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

# set the target labels
labels = np.array(outcomes.is_exciting)

#print missing fields
totalCount = essays.shape[0]
for i in range(essays.shape[1]):
    nullcount = essays[essays[essays.columns[i]].isnull()].shape[0]
    percentage = float(nullcount) / float(totalCount)*100
    if(percentage>0):
        print(essays.columns[i],percentage,'%')

#fill missing field with 'null'
essays = essays.fillna(value='null')

#feature extraction using word Counts
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(essays['short_description'].values)


'''
#preprocessing the data based on different types of attr
essays.columns['short_description']
essays['short_description'][0]
essays['short_description'][664097]
essays['short_description'][664097].split()
essays['parsed'][0] = essays['short_description
essays['parsed'][0] = essays['short_description'][0].split()
essays['new'][0] = essays['short_description'][0].split()
essays['parsed_sd'][0] = essays['short_description'][0].split()
essays['parsed_sd'] = essays['short_description'].split()
essays['parsed_sd'] = essays['short_description']
for i in essays['short_description']:
    essays['parsed_sd'] = i.split()

for i in range(len(essays['short_description'])):
    essays['parsed_sd'][i] = essays['short_description'][i].split()

essays['short_description'][0].split()
essays['short_description'][1].split()
essays['short_description'][10].split()
for i in range(len(essays['short_description'])):
    print (i)
    essays['parsed_sd'][i] = essays['short_description'][i].split()

essays['short_description'][4000].split()
essays['short_description'][4000]
totalCount = essays.shape[0]
for i in range(essays.shape[1]):
    nullcount = essays[essays[essays.columns[i]].isnull()].shape[0]
    percentage = float(nullcount)/float(totalCount) *100
    if(percentage>0):
        print(essays.columns[i],percentage,'%')


#Predicting
train = projects_data[train_idx]
test = projects_data[test_idx]
print('shape of test', test.shape)
clf = LogisticRegression()


clf.fit(train, labels=='t')
preds = clf.predict_proba(test)[:,1]
# preds = clf.predict(test)

#Save prediction into a file
sample['is_exciting'] = preds
sample.to_csv('predictions.csv', index = False)



def test():
    pass


if __name__ == '__main__':
    test()
'''
