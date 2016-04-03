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

# sort the data based on id
print('sorting data according to projectid..')
projects = projects.sort_values('projectid')
outcomes = outcomes.sort_values('projectid')
essays = essays.sort_values('projectid')


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

# split the training data and testing data
dates = np.array(projects.date_posted)
train_idx = (np.where(dates < '2014-01-01') and (np.where(dates >= '2010-04-14') ))[0]
#val_idx = np.where(dates >= '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

dataset = pd.merge(essays, outcomes, how='left', on='projectid')
print('sorting and merging to dataset complete..')

#separate train / test
essays_train = dataset.loc[train_idx]
essays_test = dataset.loc[test_idx]

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

#Drop Data with empty labels
essays_train= essays_train.dropna()

#delete empty labels from essays_test
del_list = ['is_exciting', 'at_least_1_teacher_referred_donor','fully_funded', 'at_least_1_green_donation', 'great_chat','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor', 'great_messages_proportion','teacher_referred_count', 'non_teacher_referred_count']
essays_test.drop(del_list, inplace=True, axis=1)

#Change label to binary
essays_train['y']=0
essays_train['y'][essays_train['is_exciting'] =='t'] =1
print('Processing done!')

#return essays_train, essays_test
#TODO balance exciting vs not_exciting

##TODO Run Bigram model

##TODO change to TF/IDF

#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

pipeline = Pipeline([ ('tfidf_vectorizer', TfidfVectorizer(max_df=0.99, min_df=0.01, ngram_range=(1, 2), sublinear_tf=True)), ('classifier', SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005))])

for i in ( 'short_description','need_statement', 'essay'):
    k_fold = KFold(n=len(essays_train), n_folds=3)
    for train_indices, test_indices in k_fold:
        train_text = essays_train.iloc[train_indices][i].values
        train_y = essays_train.iloc[train_indices]['y'].values

        test_text = essays_train.iloc[test_indices][i].values
        test_y = essays_train.iloc[test_indices]['y'].values

        pipeline.fit(train_text, train_y)

        predictions = pipeline.predict_proba(test_text)[:,1]
        print('AUC: '+ str(metrics.roc_auc_score(test_y,predictions)))
    pipeline.fit(essays_train[i].values, essays_train['y'].values)
    essays_train[i+'_proba'] = pipeline.predict_proba(essays_train[i])[:,1]


##TODO Try different classifier: from sklearn.naive_bayes import BernoulliNB
## Simple Perceptron, Logistic Regression, BernoulliNB
'''
from sklearn.naive_bayes import BernoulliNB

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
    ('classifier',         BernoulliNB(binarize=0.0)) ])
'''
