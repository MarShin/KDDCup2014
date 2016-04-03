# coding=utf-8
import csv
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

def process_text(string):
    string = string.lower()
    string = re.sub(r'\\n', '', string)
    string = re.sub(r'\r','', string)
    string = re.sub(r'\t','', string)
    string = string.strip()
    return string

def get_length(string):
    return len(string)

#if __name__ == "__main__":

# load the data
print('loading the data...')
projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
outcomes = pd.read_csv('../COMP4332_Dataset/outcomes.csv')
essays = pd.read_csv('../COMP4332_Dataset/essays.csv')
print('complete..')

# sort the data based on id
print('sorting data according to projectid..')
projects = projects.sort_values('projectid')
outcomes = outcomes.sort_values('projectid')
essays = essays.sort_values('projectid')


# split the training data and testing data
dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

#check missing fields
print('Empty fields: ')
totalCount = essays.shape[0]
for i in range(essays.shape[1]):
    nullcount = essays[essays[essays.columns[i]].isnull()].shape[0]
    percentage = float(nullcount) / float(totalCount)*100
    if(percentage>0):
        print(essays.columns[i],percentage,'%')

# fill missing data with null
print("filling missing data..")
essays = essays.fillna(value='null')

#pre-process text fields (to lowercase, erase whitespace, add length attribute)
print("preprocessing text fields..")
for i in ['title', 'short_description','need_statement', 'essay']:
    essays[i]=essays[i].apply(process_text)
    essays[i+'_length'] = essays[i].apply(get_length)


#separate train / test
essays_train = essays.loc[train_idx]
essays_test = essays.loc[test_idx]

#check dataset size
print("essays_train size: ", essays_train.shape)
print("essays_test size: ", essays_test.shape)

#check if in correct order
print("Essays_Train: ")
print(essays_train.head())
print("Essays_Test: ")
print(essays_test.head())

#sort again
essays_train = essays_train.sort_values('projectid')
essays_test = essays_test.sort_values('projectid')


#reset index
print('resetting index..')
essays_train = essays_train.reset_index(drop=True)
essays_test = essays_test.reset_index(drop=True)

# set target labels
labels = pd.DataFrame(outcomes.is_exciting)

#concat label to data
essays_train = pd.concat([essays_train,labels], axis=1)

#feature extraction using word Counts
print('Extracting Features from [text]..')

'''
############Set Target Train Text #################
#title, short_description, need_statement, essay
#title_length, short_description_length, need_statement_length, essay_length
train_data = 'short_description'
target_label = 'is_exciting'
###################################################

count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(essays_train[train_data].values)

classifier = MultinomialNB()
targets = essays_train[target_label].values
classifier.fit(counts, targets)
'''

print('Pipelining..')
pipeline = Pipeline([ ('vectorizer', CountVectorizer(ngram_range=(1,2))), ('classifier', MultinomialNB()) ])

'''
pipeline.fit(essays_train[train_data].values, essays_train[target_label].values)

 k_fold = KFold(n=len(essays_train), n_folds=6)
 scores = []
 confusion = np.array([[0, 0], [0, 0]])
 for train_indices, test_indices in k_fold:
    train_text = essays_train.iloc[train_indices][train_data].values
    train_y = essays_train.iloc[train_indices][target_label].values

    test_text = essays_train.iloc[test_indices][train_data].values
    test_y = essays_train.iloc[test_indices][target_label].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label='t')
    scores.append(score)

print(train_data+': ')
print('Total text classified:', len(essays_train))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)
# Total emails classified: 619326
# Score: 0.0178885877784
# Confusion matrix:
# [[21660   178]
#  [ 3473 30015]]
'''
#'title', 'short_description',
for i in ( 'need_statement', 'essay'):
    #pipeline.fit(essays_train[i].values, essays_train['is_exciting'].values)
    k_fold = KFold(n=len(essays_train), n_folds=6)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])
    for train_indices, test_indices in k_fold:
        train_text = essays_train.iloc[train_indices][i].values
        train_y = essays_train.iloc[train_indices]['is_exciting'].values

        test_text = essays_train.iloc[test_indices][i].values
        test_y = essays_train.iloc[test_indices]['is_exciting'].values

        pipeline.fit(train_text, train_y)

        predictions = pipeline.predict(test_text)
        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label='t')
        scores.append(score)
    print(i+': ')
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)
