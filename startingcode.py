# coding=utf-8
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# load the data
print('loading the data...')
projects = pd.read_csv('./data/projects.csv')
outcomes = pd.read_csv('./data/outcomes.csv')
sample = pd.read_csv('./data/sampleSubmission.csv')
print('complete..')

# sort the data based on id
projects = projects.sort('projectid')
sample = sample.sort('projectid')
outcomes = outcomes.sort('projectid')

# split the training data and testing data
dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

# fill the missing data
projects = projects.fillna(method='pad') # fill the missing hole with the previous observation data

# set the target labels
labels = np.array(outcomes.is_exciting)

#preprocessing the data based on different types of attr
projects_numeric_columns = ['school_latitude', 'school_longitude',
                            'fulfillment_labor_materials',
                            'total_price_excluding_optional_support',
                            'total_price_including_optional_support']

projects_id_columns = ['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid']
projects_categorial_columns = np.array(list(set(projects.columns).difference(set(projects_numeric_columns)).difference(set(projects_id_columns)).difference(set(['date_posted']))))

projects_categorial_values = np.array(projects[projects_categorial_columns])

print(projects_categorial_columns)
print(projects_categorial_columns.shape)
print(projects_categorial_values[:, 0].shape)

# encode the category value and reform the original data
label_encoder = LabelEncoder()
projects_data = label_encoder.fit_transform(projects_categorial_values[:,0])

for i in range(1, projects_categorial_values.shape[1]):
    label_encoder = LabelEncoder()
    projects_data = np.column_stack((projects_data, label_encoder.fit_transform(projects_categorial_values[:,i])))


projects_data = projects_data.astype(float)
print('The shape of the project data', projects_data.shape)

# One hot encoding
print('one hot encoding...')
enc = OneHotEncoder()
enc.fit(projects_data)
projects_data = enc.transform(projects_data)
print('The shape of the project data after one hot encoding', projects_data.shape)



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
