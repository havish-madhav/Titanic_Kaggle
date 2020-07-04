#importing the required libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


#reading the train and test data 
train_df=pd.read_csv("C:/Users/hp/Desktop/titanic/train.csv")
test_df=pd.read_csv("C:/Users/hp/Desktop/titanic/test.csv")

train_df.head()
test_df.head()


#checking for the null values
train_df.isnull()

#visualizing the null values 
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#to know the count of missing values
missing_values1=train_df.isnull().sum()
missing_values2=test_df.isnull().sum()
print(missing_values1,missing_values2)

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train_df)

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex',data=train_df,palette='RdBu_r')
#the below plot shows that large number of females survived and large number of males died 

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass',data=train_df,palette='rainbow')
#the below plot shows that more pclass=3 people died and most of the pclass 1 people survived

sns.distplot(train_df['Age'].dropna(),kde=False,color='darkblue',bins=40)
train_df['Fare'].hist(color='green',bins=40,figsize=(8,4)) 


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age',data=train_df,palette='winter')
#by observing the boxplot we can know the average ages of different class people 

plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age',data=test_df,palette='winter')
#by observing the boxplot we can know the average ages of different class people 


#now creating a function to fill the age based on the average age of different pclass
def input_age(input_data):
    Age=input_data[0]
    Pclass = input_data[1]
    if(pd.isnull(Age)):
        if(Pclass==1):
            return 37
        elif(Pclass==2):
            return 29
        else:
            return 24
    else:
        return Age

train_df['Age']=train_df[['Age','Pclass']].apply(input_age,axis=1)
#now check that there are no missing values in age 
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

def input_age(input_data):
    Age=input_data[0]
    Pclass = input_data[1]
    if(pd.isnull(Age)):
        if(Pclass==1):
            return 42
        elif(Pclass==2):
            return 27
        else:
            return 24
    else:
        return Age
test_df['Age']=test_df[['Age','Pclass']].apply(input_age,axis=1)
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


#since morethan 75% of cabin values are missing we will remove that coloumn 
drop_column = ['Cabin']
train_df.drop(drop_column, axis=1, inplace = True)
test_df.drop(drop_column,axis=1,inplace=True)


#now lets drop the columns and combine the datasets
train_df = train_df.drop(['Name', 'PassengerId','Ticket'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


#changing the sex values to numeric
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

#categorizing the sex values 
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

#changing the embarked values to numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()

#categorizing the fare values 
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
combine = [train_df, test_df]
    
train_df.head(10)


#splitting the data into train and test
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

#libraries for ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


#K Nearest Neighbour
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian