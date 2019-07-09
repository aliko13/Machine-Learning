
# import pandas library
import pandas as pd

# load feature names from featureNames.txt file, set to list
featureNames = pd.read_table("featureNames.txt", header=None)
featureNames = list(featureNames[0])

# read data
data = pd.read_csv('DataSet.txt', sep=',', header=None)
# set column names to featureNames list
data.columns=featureNames
# fill missing values with '?' mark
data.fillna('?')
# only print first 5 rows
data.head(5)


# generates descriptive statistics of the dataset, and transpose it
data.describe().T


# creating continuos features dataframe
continuous = pd.DataFrame(columns=['age','fnlwgt','education-num',
                                   'capital-gain','capital-loss',
                                   'hours-per-week'])

# setting empty list for continuous features
cont_features=[]

# append all column names to the list
cont_features.extend(continuous.columns)

# getting data from 'data' df to 'continuous' df
for i in cont_features:
    continuous[i]=data[i]

# the number of distinct values (cardinality)
cardinality=[]
for i in cont_features:
    cardinality.append(len(continuous[i].unique()))

# find median of each features
median=[]
for i in cont_features:
    median.append(continuous[i].median())

# find the missing values (percentage) of each features
missing=[]
for i in cont_features:
    missing.append(continuous[i].isin([' ?']).sum()/30940*100)

# create dataframe with new features
cont_describe = continuous.describe().T

# appen lists to new dataframe
cont_describe['card.']=cardinality
cont_describe['median']=median
cont_describe['missing']=missing
cont_describe.index.name='Features'

# testing
cont_describe



# ordering
df_cont = cont_describe[['count','missing','card.','min','25%',
                         'mean','median','75%','max','std']]
df_cont.columns = ['Count','%Miss','Card','Min','1st Quart',
                   'Mean', 'Median', '3rd Quart', 'Max', 'Std. Dev']

# send dataframe to .csv file
df_cont.to_csv('D14122782CONT.csv', sep='\t')

# final result of the continuous features
df_cont


# creating categorical features dataframe
categorical = pd.DataFrame(columns=['workclass','education','marital-status',
                                    'occupation','relationship','race',
                                   'sex','native-country','target'])

# setting empty list for categorical features
cat_features=[]

# append all column names to the list
cat_features.extend(categorical.columns)

# getting data from 'data' df to 'categorical' df
for i in cat_features:
    categorical[i]=data[i]
    
# describe df and transpose it  
cat_describe=categorical.describe(include='all').T

# delete count column because, this column results is not right
del cat_describe['count']

# give name to index
cat_describe.index.name="Features"
cat_describe


# required categorical features' lists
count=[]
missing_value=[]
second_mode=[]
second_count=[]
mode_percentage=[]
second_mode_percentage=[]

# fill lists
for i in cat_features:
    count.append(30940-(categorical[i].isin([' ?']).sum()))
    missing_value.append(categorical[i].isin([' ?']).sum()/30940*100)
    mode_percentage.append(categorical[i].value_counts().
                                    head(2)[0]/30940*100)
    second_mode.append(categorical[i].value_counts().head(2).index[1])
    second_count.append(categorical[i].value_counts().head(2)[1])
    second_mode_percentage.append(categorical[i].value_counts().
                                  head(2)[1]/30940*100)

# append lists to dataframe
cat_describe['Count']=count
cat_describe['Missing %']=missing_value
cat_describe['Mode %']=mode_percentage
cat_describe['2nd Mode']=second_mode
cat_describe['2nd Mode Count']=second_count
cat_describe['2nd Mode %']=second_mode_percentage

# testing
cat_describe


# ordering of the column names 
df_cat = cat_describe[['Count','Missing %','unique','top','freq','Mode %',
                        '2nd Mode','2nd Mode Count','2nd Mode %']]
df_cat.columns = ['Count','% Missing','Cardinality','Mode','Mode Count',
                  'Mode %', '2nd Mode', '2nd Mode Count', '2nd Mode %']


# send dataframe to .csv file
df_cat.to_csv('D14122782CAT.csv', sep='\t')

# final result
df_cat

