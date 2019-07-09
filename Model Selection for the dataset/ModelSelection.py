import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn import tree, neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import interp

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold

# columns
colnames = ['id','age','job','marital','education','default','balance','housing',
            'loan','contact','day','month','duration','campaign','pdays','previous','poutcome','target']

# Read trainingset.csv file to dataframe
training_set = pd.read_csv('/home/aliko/Desktop/Machine_Learning/Assignments/D14122782Assignment2/Datasets/trainingset.txt', 
                   sep=",", header=None)
training_set.columns = colnames

training_set.head(5)


# checking for null values
training_set.isnull().sum()

typeA,typeB = training_set['target'].value_counts()
print("TypeA count: "+str(typeA))
print("TypeB count: "+str(typeB))
print("Ratio TypeA: "+str(typeA / (typeA + typeB)))
print("Ratio TypeB: "+str(typeB / (typeA + typeB)))


labels = training_set['target']
# split dataset to traing and test set
trainX, testX, trainY, testY = cross_validation.train_test_split(training_set, labels, 
                                                                 test_size = 0.2, random_state = 0)

# Feature extraction for training set
# training set

# Numeric Features 
num_feat= ['age','balance','day','duration','campaign','pdays','previous'] 
numeric_features = trainX[num_feat]  

# Categorical Features 
cat_feat = ['job','marital','education','default','housing','loan','contact','month','poutcome'] 
categ_features = trainX[cat_feat]  

# Transpose cat features into array of dicts of features 
categ_features = categ_features.T.to_dict().values() 

# Convert to numeric encoding 
vectorizer = DictVectorizer(sparse = False) 
vec_categ_features = vectorizer.fit_transform(categ_features)

trainX = np.hstack((numeric_features.as_matrix(),vec_categ_features))
trainX


# Feature extraction for testing set

# Numeric Features 
num_feat= ['age','balance','day','duration','campaign','pdays','previous'] 
numerical_features = testX[num_feat]  

# Categorical Features 
cat_feat = ['job','marital','education','default','housing','loan','contact','month','poutcome'] 
categorical_features = testX[cat_feat]  

# Transpose cat features into array of dicts of features 
categ_features = categorical_features.T.to_dict().values() 

# Convert to numeric encoding 
vectorizer = DictVectorizer(sparse = False) 
vec_categ_features = vectorizer.fit_transform(categ_features)

testX = np.hstack((numerical_features.as_matrix(),vec_categ_features))
testX


# data preperation for Stratified K-folds 
stratcolnames = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','target']
X = training_set[stratcolnames]

# preprocess X
# Numeric Features
num_feat= ['age','balance','day','duration','campaign','pdays','previous']
numerical_features = X[num_feat]

# Categorical Features
cat_feat = ['job','marital','education','default','housing','loan','contact','month','poutcome']
categ_features = X[cat_feat]

# Transpose cat features into array of dicts of features
categ_features = categ_features.T.to_dict().values()
# Convert to numeric encoding
vectorizer = DictVectorizer(sparse = False)
vec_categ_features = vectorizer.fit_transform(categ_features)

X = np.hstack((numerical_features.as_matrix(),vec_categ_features))
X


# Binaryzing the labels for the model
labelbin = LabelBinarizer()
Y = training_set['target']
Y = labelbin.fit_transform(Y)
Y = pd.DataFrame(data=Y)
# first 5 rows
Y.head(5)


# Decision Tree Classifier

decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(trainX,trainY)


# test predictions accuracy
predictions = decision_tree.predict(testX) 

#Output the accuracy score of the model on the test set
print("Accuracy of Decision Tree Classifier = " + str(accuracy_score(testY, predictions, normalize=True)))

#Output the confusion matrix on the test set 
confusionMatrix = confusion_matrix(testY, predictions) 
print(confusionMatrix)


# K-Near Neighbors

knn = neighbors.KNeighborsClassifier()
knn.fit(trainX, trainY)


# test predictions accuracy
predictions = knn.predict(testX)
#Output the accuracy score of the model on the test set
print("Accuracy of KNN Classifier = " + str(accuracy_score(testY, predictions, normalize=True)))
#Output the confusion matrix on the test set
confusionMatrix = confusion_matrix(testY, predictions)
print(confusionMatrix)


# Naive Bayes Classifier

gnb = GaussianNB()
gnb.fit(trainX, trainY)


# test predictions accuracy
predictions = gnb.predict(testX)
#Output the accuracy score of the model on the test set
print("Accuracy= " + str(accuracy_score(testY, predictions, normalize=True)))
#Output the confusion matrix on the test set
confusionMatrix = confusion_matrix(testY, predictions)
print(confusionMatrix)


# The smaples are heavily biased, that is why we use stratified K-folds with ROC cureve to evaluate the models.
# Lets compare all the models to know which model fits bets to the dataset
# Stratified K fold for decision tree

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Reshaping Y to get around indices error
c, r = Y.shape
labels = Y.values.reshape(c,)
base_fpr = np.linspace(0, 1, 101)

plt.figure(figsize=(5, 5))

auc_avg = []
print("Accuracy for each fold")
print("------------------------------------------------")
for i,(train,test) in enumerate(kfold.split(X,labels)):
    model = decision_tree.fit(X[train],labels[train])
    scoreY = model.predict(X[test])
    print("Fold"+str(i+1)+" Accuracy = " + str(accuracy_score(labels[test], scoreY, normalize=True)))
    
    unique, counts = np.unique(scoreY,return_counts=True)
    res = dict(zip(unique,counts))
    fpr, tpr, _ = roc_curve(labels[test], scoreY[:])
    auc_avg.append(auc(fpr,tpr))

print("------------------------------------------------")
print("Area Under Curve Average: ")
print(str(np.mean(auc_avg)))


# Stratified K fold for KNN
knn = neighbors.KNeighborsClassifier()

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Reshaping Y to get around indices error
c, r = Y.shape
labels = Y.values.reshape(c,)
base_fpr = np.linspace(0, 1, 101)

plt.figure(figsize=(5, 5))

auc_avg = []
print("Accuracy for each fold")
print("------------------------------------------------")
for i,(train,test) in enumerate(kfold.split(X,labels)):
    model = knn.fit(X[train],labels[train])
    y_score = model.predict(X[test])
    print("Fold"+str(i+1)+" Accuracy = " + str(accuracy_score(labels[test], y_score, normalize=True)))
    
    unique, counts = np.unique(y_score,return_counts=True)
    res = dict(zip(unique,counts))
    fpr, tpr, _ = roc_curve(labels[test], y_score[:])
    auc_avg.append(auc(fpr,tpr))

print("------------------------------------------------")
print("Area Under Curve Average: ")
print(str(np.mean(auc_avg)))


# Stratified K fold for Naive Base

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
gnb = GaussianNB()
# Reshaping Y to get around indices error
c, r = Y.shape
labels = Y.values.reshape(c,)
base_fpr = np.linspace(0, 1, 101)

plt.figure(figsize=(5, 5))

auc_avg = []
print("Accuracy for each fold")
print("------------------------------------------------")
for i,(train,test) in enumerate(kfold.split(X,labels)):
    model = gnb.fit(X[train],labels[train])
    y_score = model.predict(X[test])
    print("Fold"+str(i+1)+" Accuracy = " + str(accuracy_score(labels[test], y_score, normalize=True)))
    
    unique, counts = np.unique(y_score,return_counts=True)
    res = dict(zip(unique,counts))
    fpr, tpr, _ = roc_curve(labels[test], y_score[:])
    auc_avg.append(auc(fpr,tpr))

print("------------------------------------------------")
print("Area Under Curve Average: ")
print(str(np.mean(auc_avg)))


# The ROC results show us that Naive Base Classifier produce better predictions.
# Train the model on the training set
gnb = GaussianNB()
gnb.fit(X,labels)


# Feature extraction on the queries

# Load in queries.txt
colnames = ['id','age','job','marital','education','default','balance','housing',
            'loan','contact','day','month','duration','campaign','pdays','previous','poutcome','target']

# Read queries.txt file to dataframe
df_queries = pd.read_csv('/home/aliko/Desktop/Machine_Learning/Assignments/D14122782Assignment2/Datasets/queries.txt', 
                   sep=",", header=None)
df_queries.columns = colnames

df_queries.head(5)


# Numeric Features 
num_feat= ['age','balance','day','duration','campaign','pdays','previous'] 
numerical_features = df_queries[num_feat]  

# Categorical Features 
cat_feat = ['job','marital','education','default','housing','loan','contact','month','poutcome'] 
categorical_features = df_queries[cat_feat]  

# Transpose cat features into array of dicts of features 
categ_features = categorical_features.T.to_dict().values() 

# Convert to numeric encoding 
vectorizer = DictVectorizer(sparse = False) 
vec_categ_features = vectorizer.fit_transform(categ_features)

queries = np.hstack((numerical_features.as_matrix(),vec_categ_features))

# Naive Base predictions
predictions = gnb.predict(queries)
# Decode predictions
pred =[]
for i,item in enumerate(predictions):
    if item == 0:
        pred.append('TypeA')
    else:
        pred.append('TypeB')

predictions = pd.DataFrame()
predictions['tstid'] = df_queries['id']
predictions['prediction'] = pred
file.head(5)


# Comparing the results with results file given by Svetlana
a,b = file['prediction'].value_counts()

# Ratio
print("Ratios")
print("Type A: "+str(a / (a + b)) +" Type B: "+str(b / (a + b)) )


# Read trainingset.csv file to dataframe
testing_set = pd.read_csv('/home/aliko/Downloads/testset_answers.txt', 
                   sep=",", header=None)
testing_set.columns = ['tstid','prediction']
c,d = testing_set['prediction'].value_counts()
print("Ratios")
print("Type A: "+str(c / (c + d)) +" Type B: "+str(d / (c + d)) )


# load the predictions to .txt file
predictions.to_csv('/home/aliko/Desktop/Machine_Learning/Assignments/D14122782Assignment2/Datasets/predictions.txt', 
          header=None, index=None, sep=',', mode='a')

