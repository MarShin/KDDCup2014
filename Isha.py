projects = pd.read_csv('../COMP4332_Dataset/projects.csv')
outcomes = pd.read_csv('../COMP4332_Dataset/outcomes.csv')

projects = projects.sort_values('projectid')
outcomes = outcomes.sort_values('projectid')

#check if ordered correctly: Check if ids are in ascending order
projects.head(3)
outcomes.head(3)

dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
projects_train = projects.loc[train_idx]
labels = pd.DataFrame(outcomes.is_exciting)

#check if ordered correctly: Check if ids are in ascending order
projects_train.head(3)
outcomes.head(3)

#reset index
projects_train = projects_train.reset_index(drop=True)

#check size if projects_train and outcomes are equal
projects_train.shape
outcomes.shape

train_dates = np.array(projects_train.date_posted)

dataset = pd.concat([train_dates,labels], axis=1)

dataset.to_csv('projects_train.csv',encode='utf-8',index=False)
