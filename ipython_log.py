# IPython log file

get_ipython().magic('load essaysPreprocess.py')
# %load essaysPreprocess.py
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
train_idx = (np.where(dates < '2014-01-01') and (np.where(dates >= '2010-04-14') ))[0]
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

print('done!')
get_ipython().magic('who')
essays_train.shape
essays_test.shape
outcomes.shape
outcomes.head()
outcomes = outcomes.reset_index(drop=True)
essays_train.head()
train_idx = (np.where(dates < '2014-01-01'))[0]
len(train_idx)
outcomes.shape
essays_train = essays.loc[train_idx]
essays_train.shape
essays_train.head()
essays_train.sort_values('projectid')
essays_train = essays_train.sort_values('projectid')
outcomes.head()
essays_train.head()
essays_train = essays_train.reset_index(drop=True)
outcomes.head()
labels.head()
labels = pd.DataFrame(outcomes.is_exciting)
essays_train = pd.concat([essays_train,labels], axis=1)
essays_train.shape
dates_new = np.array(essays_train.date_posted)
train_idx = (np.where(dates < '2014-01-01') and (np.where(dates >= '2010-04-14') ))[0]
train_idx.shape
essays_train =
essays_train = essays.loc[train_idx]
essays_train.shape
labels.shape
outcomes.shape
outcomes.head()
train_idx = (np.where(dates < '2014-01-01'))[0]
train_idx.shape
essays_train = essays.loc[train_idx]
essays_train = essays_train.reset_index(drop=True)
essays_train.head()
essays_train = essays_train.sort_values('projectid')
essays_train.head()
essays_train = essays_train.reset_index(drop=True)
outcomes.head()
outcomes.shape
labels = pd.DataFrame(outcomes.is_exciting)
essays_train = pd.merge(essays_train, outcomes,how='left',on='projectid')
essays_train.shape
essays_train.head()
essays.head(3)
outcomes.head(3)
dataset = pd.merge(essays, outcomes, how='left', on'projectid')
dataset = pd.merge(essays, outcomes, how='left', on='projectid')
dataset.head(3)
essays.columns.values
datasets.columns.values
dataset.columns.values
dataset.shape
dataset.shape
dataset.head()
dates = np.array(projects.date_posted)
train_idx = (np.where(dates < '2013-01-01') and (np.where(dates >= '2010-04-14') ))[0]
train_idx = (np.where(dates < '2014-01-01') and (np.where(dates >= '2010-04-14') ))[0]
totalCount = essays.shape[0]
for i in range(essays.shape[1]):
        nullcount = essays[essays[essays.columns[i]].isnull()].shape[0]
        percentage = float(nullcount) / float(totalCount)*100
        if(percentage>0):
                print(essays.columns[i],percentage,'%')
        
essays_train = dataset.loc[train_idx]
essays_test = dataset.loc[test_idx]
essays_train.shape
essays_test.shape
essays_train.head(3)
essays.head(1)
dataset.head(1)
dataset.iloc[1]
dataset.iloc[0]
projects.head(1)
essays_train.shape
essays_train.head(2)
essays_test.head(2)
essays_test.shape
essays_test = essays_test.sort_values('projectid')
essays_train = essays_train.sort_values('projectid')
essays_train.head(2)
essays_test.head(2)
essays_test = essays_test.reset_index(drop=True)
essays_test.dropna(how='all')
outcomes.columns.values
essays_test.drop(['projectid', 'is_exciting', 'at_least_1_teacher_referred_donor','fully_funded', 'at_least_1_green_donation', 'great_chat','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor', 'great_messages_proportion','teacher_referred_count', 'non_teacher_referred_count'])
essays_test.drop(('is_exciting', 'at_least_1_teacher_referred_donor','fully_funded', 'at_least_1_green_donation', 'great_chat','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor', 'great_messages_proportion','teacher_referred_count', 'non_teacher_referred_count'))
column_list = [essays_test.drop(('is_exciting', 'at_least_1_teacher_referred_donor','fully_funded', 'at_least_1_green_donation', 'great_chat','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor', 'great_messages_proportion','teacher_referred_count', 'non_teacher_referred_coun
del_list = ['is_exciting', 'at_least_1_teacher_referred_donor','fully_funded', 'at_least_1_green_donation', 'great_chat','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor', 'great_messages_proportion','teacher_referred_count', 'non_teacher_referred_count']
for i in del_list:
    essays_test.drop(i, inplace=True, axis=1)
    
essays_test.shape
essays_train.shape
for i in len(outcomes.shape[0]):
    if(outcomes['is_exciting'].iloc[i] ==t))))
count=0
for i in len(outcomes.shape[0]):
    if(outcomes['is_exciting'].iloc[i] =='t')
for i in len(outcomes.shape[0]):
    if(outcomes['is_exciting'].iloc[i] =='t'):
        count+=1
        
for i in range(outcomes.shape[0]):
    if(outcomes['is_exciting'].iloc[i] =='t'):
        count+=1
        
count
outcomes.shape[0]
for i in range(essays_train.shape[0]):
    if(outcomes['is_exciting'].iloc[i] =='t'):))
count1=0
for i in range(essays_train.shape[0]):
    if(essays_train['is_exciting'].iloc[i] =='t'):
        count1+=1
        
count1
619326-36710
465461-36710
outcomes.head(2)
labels = np.array(essays_train.is_exciting)
labels.shape
false_idx = np.where(labels =='f')[0]
false_idx.shape
true_idx = np.where(labels =='t')[0]
true_idx.shape
totalCount = essays.shape[0]
totalCount = essays_train.shape[0]
for i in range(essays_train.shape[1]):
        nullcount = essays_train[essays_train[essays_train.columns[i]].isnull()].shape[0]
        percentage = float(nullcount) / float(totalCount)*100
        if(percentage>0):
                print(essays_train.columns[i],percentage,'%')
        
projectid_validtrain = np.array(essays_train.projectid)
essays_train.columns.values
essays_train.columns.values[10:]
valid_outcome = essays_train[10:]
valid_outcome.columns.values
valid_outcome.shape
valid_outcome = essays_train[:]10:]
valid_outcome = essays_train[:][10:]
valid_outcome.shape
valid_outcome=essays_train[('is_exciting','fully_funded')] 
valid_outcome=essays_train['is_exciting','fully_funded'] 
valid_outcome=essays_train.getcolumn['is_exciting','fully_funded'] 
valid_outcome=essays_train.loc[:,10:]
valid_outcome=essays_train.iloc[:,10:]
valid_outcome.shape
valid_outcome.columns.values
valid_outcome.to_csv('valid_outcome.csv',encode='utf-8',index=False)
test1 = valid_outcome.dropna()
test1.shape
totalCount = test1.shape[0]
for i in range(test1.shape[1]):
        nullcount = test1[test1[test1.columns[i]].isnull()].shape[0]
        percentage = float(nullcount) / float(totalCount)*100
        if(percentage>0):
                print(test1.columns[i],percentage,'%')
        
valid_outcome = test1
valid_outcome.shape
essays_train.shape
test1 = essays_train.dropna()
test1.to_csv('valid_outcome.csv',encode='utf-8',index=False)
essays_train = test1
valid_outcome=essays_train.iloc[:,10:]
valid_outcome.shape
valid_outcome.to_csv('valid_outcome.csv',encode='utf-8',index=False)
t_count=0
for i in range(valid_outcome.shape[0]):
    if valid_outcome['is_exciting'].iloc[i] == 't':
        t_count+=1
        
t_count
f_count=0
for i in range(valid_outcome.shape[0]):
    if valid_outcome['is_exciting'].iloc[i] == 'f':
        f_count+=1
        
f_count
get_ipython().magic('logstart')
from sklearn.feature_extraction.text import TfidfTransformer
pipeline = Pipeline([ ('count_vectorizer', CountVectorizer(ngram_range=(1,2))), ('tfidf_transformer', TfidfTransformer()), ('classifier', MultinomialNB()) ])
for i in ( 'need_statement', 'essay'):
        k_fold = KFold(n=len(essays_train), n_folds=3)
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
for i in ( 'need_statement', 'essay'):
        k_fold = KFold(n=len(essays_train), n_folds=3)
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
        
get_ipython().magic('who')
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
pipeline = Pipeline([ ('tfidf_vectorizer', TfidfVectorizer(max_df=0.85, min_df=0.15, ngram_range=(1, 2), sublinear_tf=True)), ('classifier', SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005))])
for i in ( 'title', 'short_description','need_statement', 'essay'):
        k_fold = KFold(n=len(essays_train), n_folds=3)
        for train_indices, test_indices in k_fold:
                train_text = essays_train.iloc[train_indices][i].values
                train_y = essays_train.iloc[train_indices]['is_exciting'].values
                test_text = essays_train.iloc[test_indices][i].values
                test_y = essays_train.iloc[test_indices]['is_exciting'].values
                pipeline.fit(train_text, train_y)
                predictions = pipeline.predict_proba(test_text)
                print('AUC: '+ str(metrics.roc_auc_score(test_y,predictions)))
        
pipeline = Pipeline([ ('tfidf_vectorizer', TfidfVectorizer(max_df=0.99, min_df=0.01, ngram_range=(1, 2), sublinear_tf=True)), ('classifier', SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005))])
for i in ( 'title', 'short_description','need_statement', 'essay'):
    ))))    k_fold = KFold(n=len(essays_train), n_folds=3)
pipeline = Pipeline([ ('tfidf_vectorizer', TfidfVectorizer(max_df=0.99, min_df=0.01, ngram_range=(1, 2), sublinear_tf=True)), ('classifier', SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005))])
for i in ( 'short_description','need_statement', 'essay'):
        k_fold = KFold(n=len(essays_train), n_folds=3)
        for train_indices, test_indices in k_fold:
                train_text = essays_train.iloc[train_indices][i].values
                train_y = essays_train.iloc[train_indices]['is_exciting'].values
                test_text = essays_train.iloc[test_indices][i].values
                test_y = essays_train.iloc[test_indices]['is_exciting'].values
                pipeline.fit(train_text, train_y)
                predictions = pipeline.predict_proba(test_text)
                print(i+'::: AUC: '+ str(metrics.roc_auc_score(test_y,predictions)))
        
from sklearn import metrics
for i in ( 'short_description','need_statement', 'essay'):
        k_fold = KFold(n=len(essays_train), n_folds=3)
        for train_indices, test_indices in k_fold:
                train_text = essays_train.iloc[train_indices][i].values
                train_y = essays_train.iloc[train_indices]['is_exciting'].values
                test_text = essays_train.iloc[test_indices][i].values
                test_y = essays_train.iloc[test_indices]['is_exciting'].values
                pipeline.fit(train_text, train_y)
                predictions = pipeline.predict_proba(test_text)
                print(i+'::: AUC: '+ str(metrics.roc_auc_score(test_y,predictions)))
        
for i in ( 'short_description','need_statement', 'essay'):
        k_fold = KFold(n=len(essays_train), n_folds=3)
        for train_indices, test_indices in k_fold:
                train_text = essays_train.iloc[train_indices][i].values
                train_y = essays_train.iloc[train_indices]['is_exciting'].values
                test_text = essays_train.iloc[test_indices][i].values
                test_y = essays_train.iloc[test_indices]['is_exciting'].values
                pipeline.fit(train_text, train_y)
                predictions = pipeline.predict_proba(test_text)
                print('train_text,y: '+train_text.shape+' '+train_y.shape)
                print('test_text,y: '+test_text.shape+' '+test_y.shape)
                print('predictions: '+predictions.shape)
                print('AUC: '+ str(metrics.roc_auc_score(test_y,predictions)))
        
for i in ( 'short_description','need_statement', 'essay'):
        k_fold = KFold(n=len(essays_train), n_folds=3)
        for train_indices, test_indices in k_fold:
                train_text = essays_train.iloc[train_indices][i].values
                train_y = essays_train.iloc[train_indices]['is_exciting'].values
                test_text = essays_train.iloc[test_indices][i].values
                test_y = essays_train.iloc[test_indices]['is_exciting'].values
                pipeline.fit(train_text, train_y)
                predictions = pipeline.predict_proba(test_text)
                print('train_text,y: '+str(train_text.shape)+' '+str(train_y.shape))
                print('test_text,y: '+str(test_text.shape)+' '+str(test_y.shape))
                print('predictions: '+str(predictions.shape))
                print('AUC: '+ str(metrics.roc_auc_score(test_y,predictions)))
        
for i in ( 'short_description','need_statement', 'essay'):
        k_fold = KFold(n=len(essays_train), n_folds=3)
        for train_indices, test_indices in k_fold:
                train_text = essays_train.iloc[train_indices][i].values
                train_y = essays_train.iloc[train_indices]['is_exciting'].values
                test_text = essays_train.iloc[test_indices][i].values
                test_y = essays_train.iloc[test_indices]['is_exciting'].values
                pipeline.fit(train_text, train_y)
                predictions = pipeline.predict_proba(test_text)[:,1]
                print('AUC: '+ str(metrics.roc_auc_score(test_y,predictions)))
            pipeline.fit(essays_train[i].values, essays_train['is_exciting'].values)
for i in ( 'short_description','need_statement', 'essay'):
        k_fold = KFold(n=len(essays_train), n_folds=3)
        for train_indices, test_indices in k_fold:
                train_text = essays_train.iloc[train_indices][i].values
                train_y = essays_train.iloc[train_indices]['is_exciting'].values
                test_text = essays_train.iloc[test_indices][i].values
                test_y = essays_train.iloc[test_indices]['is_exciting'].values
                pipeline.fit(train_text, train_y)
                predictions = pipeline.predict_proba(test_text)[:,1]
                print('AUC: '+ str(metrics.roc_auc_score(test_y,predictions)))
        pipeline.fit(essays_train[i].values, essays_train['is_exciting'].values)
        essays_train[i+'_proba'] = pipeline.predict_proba(essays_train[i])[:,1]
        
essays_train['y']=0
essays_train['y'][essays_train['is_exciting'] =='t'] =1 
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
        
get_ipython().magic('load temp_load_saveTestPrediction.py')
# %load temp_load_saveTestPrediction.py
for i in ( 'short_description','need_statement', 'essay'):
        pipeline.fit(essays_train[i].values, essays_train['y'].values)
        test_result[i+'_proba'] = pipeline.predict_proba(essays_test[i])[:,1]

test_predictions = pd.concat([test_result['short_description_proba'],test_result['need_statement_proba'],test_result['essay_proba']], axis=1)
test_predictions.to_csv('test_predictions.csv',index=False)

get_ipython().magic('load temp_load_saveTestPrediction.py')
# %load temp_load_saveTestPrediction.py
for i in ( 'short_description','need_statement', 'essay'):
        pipeline.fit(essays_train[i].values, essays_train['y'].values)
        essays_test[i+'_proba'] = pipeline.predict_proba(essays_test[i])[:,1]

test_predictions = pd.concat([essays_test['short_description_proba'],essays_test['need_statement_proba'],essays_test['essay_proba']], axis=1)
test_predictions.to_csv('test_predictions.csv',index=False)

train_predictions = pd.concat([essays_train['short_description_proba'],essays_train['need_statement_proba'],essays_train['essay_proba']], axis=1) 
train_predictions.to_csv('train_predictions.csv',index=False)
train_predictions = pd.concat([essays_train['projectid'],train_predictions],axis=1)
train_predictions.to_csv('train_predictions.csv',index=False)
