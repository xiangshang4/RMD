import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

train_data=pd.read_pickle('Customer_train.pkl')
credit_data=pd.read_pickle('Customer_Credits.pkl')
test_data=pd.read_pickle('Customer_test.pkl')
combine=pd.merge(train_data, credit_data, on='ID',how='left')
#combine.to_csv('train_credit_combine.csv')
train_id=train_data.ID
test_id=test_data.ID
    #train_data.to_csv('train_data.csv')
    #credit_data.to_csv('credit_data.csv')
    #test_data.to_csv('test_data.csv')
'''
    with open('Customer_train.pkl', 'rb') as train:
        train_data = pickle.load(train)
    with open('Customer_Credits.pkl', 'rb') as credit:
        credit_data = pickle.load(credit)
    with open('Customer_test.pkl', 'rb') as test:
        test_data = pickle.load(test)

    #print(train_data)
    #print(credit_data)
    print(test_data)
    '''
def logreg():
    dropnan=train_data.fillna(value={'Car_Age':0}).iloc[:,1:15].dropna(axis=0,how='any')
    dropnan['Gender'] = np.where(dropnan['Gender'] == 'F', 1, 0)
    dropnan['Own_Car'] = np.where(dropnan['Own_Car'] == 'Y', 1, 0)
    dropnan['Own_Residence'] = np.where(dropnan['Own_Residence'] == 'Y', 1, 0)
    x=dropnan.iloc[:,1:14]
    y=dropnan.Default
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(x,y)
    score=cross_val_score(clf,x,y,cv=5)
    print(score)
    fillnan=test_data.fillna(value=0)
    fillnan['Gender'] = np.where(fillnan['Gender'] == 'F', 1, 0)
    fillnan['Own_Car'] = np.where(fillnan['Own_Car'] == 'Y', 1, 0)
    fillnan['Own_Residence'] = np.where(fillnan['Own_Residence'] == 'Y', 1, 0)
    testx=fillnan.iloc[:,1:14]
    predictions = clf.predict_proba(testx)
    file=pd.DataFrame(predictions)
    file.to_csv('predictionlog.csv')

def linear_test():
    dropnan=train_data.fillna(value={'Car_Age':0}).iloc[:,1:15].dropna(axis=0,how='any')
    dropnan['Gender'] = np.where(dropnan['Gender'] == 'F', 1, 0)
    dropnan['Own_Car'] = np.where(dropnan['Own_Car'] == 'Y', 1, 0)
    dropnan['Own_Residence'] = np.where(dropnan['Own_Residence'] == 'Y', 1, 0)
    x=dropnan.iloc[:,1:14]
    y=dropnan.Default
    for i in x.columns:
        dropnan.plot.scatter(x=i,y='Default')
        plt.show()
    
    lm = linear_model.LinearRegression()
    model = lm.fit(x,y)

    score=cross_val_score(model,x,y,cv=5)
    print(score)

def test_lm():
    fillnan=test_data.fillna(value=0)
    fillnan['Gender'] = np.where(fillnan['Gender'] == 'F', 1, 0)
    fillnan['Own_Car'] = np.where(fillnan['Own_Car'] == 'Y', 1, 0)
    fillnan['Own_Residence'] = np.where(fillnan['Own_Residence'] == 'Y', 1, 0)
    testx=fillnan.iloc[:,1:14]
    predictions = lm.predict(testx)
    file=pd.DataFrame(predictions)
    file.to_csv('prediction.csv')

def random_forest():
    dropnan=train_data.fillna(value={'Car_Age':0}).iloc[:,1:15].dropna(axis=0,how='any')
    dropnan['Gender'] = np.where(dropnan['Gender'] == 'F', 1, 0)
    dropnan['Own_Car'] = np.where(dropnan['Own_Car'] == 'Y', 1, 0)
    dropnan['Own_Residence'] = np.where(dropnan['Own_Residence'] == 'Y', 1, 0)
    x=dropnan.iloc[:,1:14]
    y=dropnan.Default
    clf = RandomForestClassifier(n_estimators=110, max_features='auto',min_samples_split=2,bootstrap=True, class_weight='balanced_subsample',random_state=0,n_jobs=-1)
    clf.fit(x, y)
    score=cross_val_score(clf,x,y,cv=5)
    print(score)
    print(clf.feature_importances_)
    fillnan=test_data.fillna(value=0)
    fillnan['Gender'] = np.where(fillnan['Gender'] == 'F', 1, 0)
    fillnan['Own_Car'] = np.where(fillnan['Own_Car'] == 'Y', 1, 0)
    fillnan['Own_Residence'] = np.where(fillnan['Own_Residence'] == 'Y', 1, 0)
    testx=fillnan.iloc[:,1:14]
    predictions = clf.predict_proba(testx)
    file=pd.DataFrame(predictions)
    file.to_csv('prediction.csv')
