# coding=utf-8
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# load the data
print('loading the data...')
projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
outcomes = pd.read_csv('../COMP4332_Dataset/outcomes.csv')
sample = pd.read_csv('../COMP4332_Dataset/sampleSubmission.csv')
print('complete..')

# sort the data based on id
projects = projects.sort('projectid')
sample = sample.sort('projectid')
outcomes = outcomes.sort('projectid')

# split the training data and testing data
dates = np.array(projects.date_posted)
train_idx = (np.where(dates < '2014-01-01') and (np.where(dates >= '2010-04-14') ))[0]
test_idx = np.where(dates >= '2014-01-01')[0]



# fill the missing data
projects = projects.fillna(method='pad') # fill the missing hole with the previous observation data


###Only to get labels for the valid train_idx
dataset = pd.merge(projects, outcomes, how='left', on='projectid')
projects_train = dataset.loc[train_idx]
projects_train = projects_train.sort_values('projectid')
projects_train = projects_train.reset_index(drop=True)
#projects_train= projects_train.dropna()

# set the target labels
labels = np.array(projects_train.is_exciting)

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

clf = DecisionTreeClassifier()

clf.fit(train, labels=='t')

preds = clf.predict_proba(test)[:,1]
# preds = clf.predict(test)

#Save prediction into a file
sample['is_exciting'] = preds
sample.to_csv('DT_NoBalance_predictions.csv', index = False)

def test():
    pass


if __name__ == '__main__':
    test()
