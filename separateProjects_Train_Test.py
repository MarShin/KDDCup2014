# coding=utf-8
import csv
import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.linear_model import LogisticRegression


# load the data
print('loading the data...')
projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
#outcomes = pd.read_csv('../COMP4332_Dataset/outcomes.csv')
#essays = pd.read_csv('../COMP4332_Dataset/essays.csv')
print('complete..')

# sort the data based on id
projects = projects.sort_values('projectid')
#outcomes = outcomes.sort_values('projectid')
#essays = essays.sort_values('projectid')

# split the training data and testing data
dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

# fill the missing data
#projects = projects.fillna(method='pad') # fill the missing hole with the previous observation data

# set the target labels
#labels = np.array(outcomes.is_exciting)


# separate train / test dataset
projects_train = pd.DataFrame(columns = list(projects.columns.values))
projects_test = pd.DataFrame(columns = list(projects.columns.values))

#essays_train = pd.DataFrame(columns = list(essays.columns.values))
#essays_test =pd.DataFrame(columns = list(essays.columns.values))


count=0
for i in train_idx:
    projects_train.loc[count] = projects.loc[i]
    count+=1
    print (count / len(train_idx) *100, "% train_dat done")


count=0
for i in test_idx:
    projects_test.loc[count] = projects.loc[i]
    count+=1
    print (count / len(test_idx) *100, "% test_dat done")


projects_train.to_csv('projects_train.csv')
projects_test.to_csv('projects_test.csv')

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
