# coding=utf-8
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def get_length(string):
    return len(string)


# load the data
print('loading the data...')
projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
essays = pd.read_csv('../COMP4332_Dataset/essays.csv')
outcomes = pd.read_csv('../COMP4332_Dataset/outcomes.csv')
sample = pd.read_csv('../COMP4332_Dataset/sampleSubmission.csv')
print('complete..')

# sort the data based on id
projects = projects.sort('projectid')
sample = sample.sort('projectid')
outcomes = outcomes.sort('projectid')
essays = essays.sort_values('projectid')

#print("sorted:")
#print("projects: ")
#rint(projects.head(3))
#print("sample: ")
#print(sample.head(3))
#print("outcomes: ")
#print(outcomes.head(3))
#print("essays: ")
#print(essays.head(3))

essays = essays.fillna(value='null')

print("adding text_length features..")
for i in ['need_statement', 'essay']:
    projects[i+'_length'] = essays[i].apply(get_length)

#add text proba
print("loading and adding text_proba...")
text_proba =  pd.read_csv('../COMP4332_Dataset/text_predictions.csv')
projects = pd.merge(projects, text_proba, how='left', on='projectid')

print("Filling missing data..")
# fill the missing data
#projects = projects.fillna(method='pad') # fill the missing hole with the previous observation data

projects.fulfillment_labor_materials = projects.fulfillment_labor_materials.fillna(projects.fulfillment_labor_materials.mean())
projects.school_metro= projects.school_metro.fillna("urban")
projects.school_ncesid = projects.school_ncesid.fillna(1.709930e+11)
projects.school_zip = projects.school_zip.fillna(10456.0)
projects.school_district = projects.school_district.fillna("New York City Dept of Ed")
projects.school_county = projects.school_county.fillna("Los Angeles")
projects.teacher_prefix = projects.teacher_prefix.fillna("Mrs.")
projects.primary_focus_subject = projects.primary_focus_subject.fillna("Literacy")
projects.primary_focus_area = projects.primary_focus_area.fillna("Literacy & Language")
projects.grade_level.value_counts()
projects.grade_level = projects.grade_level.fillna("Grade PreK-2")
projects.students_reached = projects.students_reached.fillna(projects.students_reached.mean())
projects.secondary_focus_area = projects.secondary_focus_area.fillna(projects.primary_focus_area)
projects.secondary_focus_subject = projects.secondary_focus_subject.fillna(projects.primary_focus_subject)
projects.resource_type.value_counts()
projects.resource_type = projects.resource_type.fillna("Supplies")



dates = np.array(projects.date_posted)
train_idx = (np.where(dates < '2014-01-01') and (np.where(dates >= '2010-04-14') ))[0]
test_idx = np.where(dates >= '2014-01-01')[0]

###Only to get labels for the valid train_idx
dataset = pd.merge(projects, outcomes, how='left', on='projectid')
projects_train = dataset.loc[train_idx]
projects_train = projects_train.sort_values('projectid')
projects_train = projects_train.reset_index(drop=True)
#projects_train= projects_train.dropna()

# set the target labels
labels = np.array(projects_train.is_exciting)

new_projects = projects


print('train size: '+str(train_idx.shape))
print('test size: '+str(test_idx.shape))
print('new_projects shape: ' +str(new_projects.shape))
print('label shape: '+str(labels.shape))

print("Preprocessing Data done!")

#preprocessing the data based on different types of attr
projects_numeric_columns = ['school_latitude', 'school_longitude',
                            'fulfillment_labor_materials','students_reached',
                            'total_price_excluding_optional_support',
                            'total_price_including_optional_support', 'need_statement_length', 'essay_length',
                            'short_description_proba', 'need_statement_proba', 'essay_proba']

projects_id_columns = ['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid']
projects_categorial_columns = np.array(list(set(new_projects.columns).difference(set(projects_numeric_columns)).difference(set(projects_id_columns)).difference(set(['date_posted']))))

projects_categorial_values = np.array(new_projects[projects_categorial_columns])

#print("categorical columns: ")
#print(projects_categorial_columns)
#print(projects_categorial_columns.shape)
#print(projects_categorial_values[:, 0].shape)

projects_selected_columns = np.array(['students_reached','total_price_excluding_optional_support','need_statement_length', 'essay_length', 'short_description_proba', 'need_statement_proba', 'essay_proba'])
projects_selected_values = np.array(new_projects[projects_selected_columns])

print("Encoding...")
# encode the category value and reform the original data
label_encoder = LabelEncoder()
projects_data = label_encoder.fit_transform(projects_categorial_values[:,0])

for i in range(1, projects_categorial_values.shape[1]):
    label_encoder = LabelEncoder()
    projects_data = np.column_stack((projects_data, label_encoder.fit_transform(projects_categorial_values[:,i])))


projects_data = projects_data.astype(float)
print('The shape of the project data', projects_data.shape)
project_data = np.concatenate([projects_data,projects_selected_values], axis=1)

'''
# One hot encoding
print('one hot encoding...')
enc = OneHotEncoder()
enc.fit(projects_data)
projects_data = enc.transform(projects_data)
print('The shape of the project data after one hot encoding', projects_data.shape)
'''


#Predicting
train = projects_data[train_idx]
test = projects_data[test_idx]
print('shape of test', test.shape)

#clf = LogisticRegression()
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)

print('classifying...')
clf.fit(train, labels=='t')

preds = clf.predict_proba(test)[:,1]
# preds = clf.predict(test)

#Save prediction into a file
sample['is_exciting'] = preds
sample.to_csv('GB_predictions.csv', index = False)

def test():
    pass


if __name__ == '__main__':
    test()
