# coding=utf-8
import csv
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def process_text(string):
    string = string.lower()
    string = re.sub(r'\\n', '', string)
    string = re.sub(r'\r','', string)
    string = re.sub(r'\t','', string)
    string = string.strip()
    return string

def get_length(string):
    return len(string)

if __name__ == "__main__":

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

    #check missing fields
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

    #reset index
    essays_train = essays_train.reset_index(drop=True)
    essays_test = essays_test.reset_index(drop=True)

    # set target labels
    labels = pd.DataFrame(outcomes.is_exciting)

    #concat label to data
    essays_train = pd.concat([essays_train,labels], axis=1)

    #feature extraction using word Counts
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(essays_train['short_description'].values)

    from sklearn.naive_bayes import MultinomialNB

    classifier = MultinomialNB()
    targets = essays_train['is_exciting'].values
    classifier.fit(counts, targets)

    '''##Test Example
    examples = ['Free Viagra call today!', "I'm going to attend the Linux users group tomorrow."]
    example_counts = count_vectorizer.transform(examples)

    predictions = classifier.predict_proba(example_counts)
    print (preditions)
        [[ 0.95173271  0.04826729]
        [ 0.75177793  0.24822207]]
    '''

    





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
