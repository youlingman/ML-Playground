import pandas as pd
import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold, cross_val_score

def cleanData(file):
	# data clean
	data = pd.read_csv(file, dtype={"Age": np.float64})
	
	# female = 0, Male = 1
	data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

	# Embarked from 'C', 'Q', 'S'
	# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

	# All missing Embarked -> just make them embark from most common place
	if len(data.Embarked[ data.Embarked.isnull() ]) > 0:
		data.Embarked[ data.Embarked.isnull() ] = data.Embarked.dropna().mode().values

	data.loc[data["Embarked"] == "S", "Embarked"] = 0
	data.loc[data["Embarked"] == "C", "Embarked"] = 1
	data.loc[data["Embarked"] == "Q", "Embarked"] = 2

	
	# Number of Siblings/Spouses Aboard maps to yes or no 
	data.SibSp = data.SibSp.map( lambda x: 0 if x == 0 else 1).astype(int)
	# Number of Parents/Children Aboard maps to yes or no
	data.Parch = data.Parch.map( lambda x: 0 if x == 0 else 1).astype(int)
	# Add family feature
	#data['Family'] =  data["Parch"] + data["SibSp"]
	#data['Family'].loc[data['Family'] > 0] = 1
	#data['Family'].loc[data['Family'] == 0] = 0
	
	# convert Fare from float to int
	data["Fare"].fillna(data["Fare"].median(), inplace=True)
	data['Fare'] = data['Fare'].astype(int)
	data['Fare'] = data['Fare'].map( lambda x: 0 if x < 50 else 1)
	
	# All the ages with no data -> make the median of all Ages
	data["Age"] = data["Age"].fillna(data["Age"].mean())
	#data["Age"] = data["Age"].map( lambda x: 0 if x < 14 else 1 if x < 50 else 2).astype(int)
	data["Age"] = data["Age"].map( lambda x: 0 if x < 14 else 1).astype(int)

	# Remove the Name column, Cabin, Ticket, Fare, SibSp and Parch
	#data = data.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Fare'], axis=1)
	#data = data.drop(['Name', 'Ticket', 'Cabin', 'Fare'], axis=1)
	
	return data

	
def trainerPicker(train_X, train_Y, trainers):
	print train_X.columns
	best_score = 0
	best_trainer = None
	for name, trainer in trainers.items():
		# 10-fold validate to pick best trainer
		print name
		score = 0
		for i in range(0, 10):
			score += cross_val_score(trainer, train_X, train_Y, cv=StratifiedKFold(train_Y, n_folds=10, shuffle=True)).sum()
		print "sum score for 10-10 fold: " + str(score)
		if best_score == 0 or best_score < score:
			print "now the best trainer is: " + name
			best_score = score
			best_trainer = trainer
		trainer.fit(train_X, train_Y)
		if hasattr(trainer, 'feature_importances_'):
			print "feature importances: " + str(trainer.feature_importances_)
		print
	best_trainer.fit(train_X, train_Y)
	return best_trainer
	
# train data
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Fare"]
train_df = cleanData('train.csv')
trainers = { "LogisticRegression" : LogisticRegression(),
             "LinearSVC" : LinearSVC(),
			 "KNeighborsClassifier" : KNeighborsClassifier(n_neighbors = 4),
			 "DecisionTreeClassifier" : DecisionTreeClassifier(),
			 "BaggingClassifier" : BaggingClassifier(n_estimators=100),
			 "RandomForestClassifier" : RandomForestClassifier(n_estimators=100),
			 "AdaBoostClassifier" : AdaBoostClassifier(n_estimators=100),
			 #"GradientBoostingClassifier" : GradientBoostingClassifier(n_estimators=100)
			 }
clf = trainerPicker(train_df[predictors], train_df['Survived'], trainers)

# predict data
test_df = cleanData('test.csv')
res = clf.predict(test_df[predictors])

# write answer
predictions_file = open("titanic.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(test_df['PassengerId'], res))
predictions_file.close()
print 'Done.'