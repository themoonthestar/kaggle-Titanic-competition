## USERNAME ON KAGGLE: richardjw

import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
#sns
sns.set(style='white',context='notebook',palette='muted')
import matplotlib.pyplot as plt

train=pd.read_csv(r'C:\Users\richard\Desktop\train.csv')
test=pd.read_csv(r'C:\Users\richard\Desktop\test.csv')
every=train.append(test,ignore_index=True)
every.describe()
every.info()

ageFacet=sns.FacetGrid(train,hue='Survived',aspect=3)
ageFacet.map(sns.kdeplot,'Fare',shade=True)
ageFacet.set(xlim=(0,150))
ageFacet.add_legend()
plt.show()
farePlot=sns.distplot(every['Fare'][every['Fare'].notnull()],label='skewness:%.2f'%(every['Fare'].skew()))
farePlot.legend(loc='best')
plt.show()

every['Fare']=every['Fare'].map(lambda x: np.log(x) if x>0 else 0)
every['Cabin']=every['Cabin'].fillna('U')
print(every['Cabin'].head())
every[every['Embarked'].isnull()]
print(every['Embarked'].value_counts())

age_by_pclass_sex = every.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {} '.format(pclass, sex, age_by_pclass_sex[sex][pclass].astype(int)))

age_by_pclass_sex = every.groupby(['Sex', 'Pclass']).mean()['Age']
for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Mean age of Pclass {} {}s: {} '.format(pclass, sex, age_by_pclass_sex[sex][pclass].astype(int)))
age_by_pclass_sex = every.groupby(['Sex', 'Pclass']).std()['Age']
for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('standard deviation age of Pclass {} {}s: {} '.format(pclass, sex, age_by_pclass_sex[sex][pclass].astype(int)))



every['Embarked']=every['Embarked'].fillna('S')
every[every['Fare'].isnull()]
every['Fare']=every['Fare'].fillna(every[(every['Pclass']==3)&(every['Embarked']=='S')&(every['Cabin']=='U')]['Fare'].mean())

#Title
every['Title']=every['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
print(every['Title'].value_counts())
TitleDict={}
TitleDict['Mr']='Mr'
TitleDict['Mlle']='Miss'
TitleDict['Miss']='Miss'
TitleDict['Master']='Master'
TitleDict['Jonkheer']='Master'
TitleDict['Mme']='Mrs'
TitleDict['Ms']='Mrs'
TitleDict['Mrs']='Mrs'
TitleDict['Don']='Royalty'
TitleDict['Sir']='Royalty'
TitleDict['the Countess']='Royalty'
TitleDict['Dona']='Royalty'
TitleDict['Lady']='Royalty'
TitleDict['Capt']='Officer'
TitleDict['Col']='Officer'
TitleDict['Major']='Officer'
TitleDict['Dr']='Officer'
TitleDict['Rev']='Officer'

every['Title']=every['Title'].map(TitleDict)
print(every['Title'].value_counts())
sns.barplot(data=every,x='Title',y='Survived')
plt.show()
sns.barplot(data=train,x='Pclass',y='Survived')
plt.show()
every['familyNum']=every['Parch']+every['SibSp']+1
sns.barplot(data=every,x='familyNum',y='Survived')
plt.show()
#deck
every['Deck']=every['Cabin'].map(lambda x:x[0])
sns.barplot(data=every,x='Deck',y='Survived')
plt.show()
#familysize
def familysize(familyNum):
    if familyNum==1:
        return 0
    if familyNum>=2 and familyNum<=4:
        return 1
    if familyNum>4:
        return 2

every['familySize']=every['familyNum'].map(familysize)
print(every['familySize'].value_counts())
sns.barplot(data=every,x='familySize',y='Survived')
plt.show()

every[every['Age'].isnull()].head()

#filter
AgePre=every[['Age','Parch','Pclass','SibSp','Title','familyNum']]
AgePre=pd.get_dummies(AgePre)
ParAge=pd.get_dummies(AgePre['Parch'],prefix='Parch')
SibAge=pd.get_dummies(AgePre['SibSp'],prefix='SibSp')
PclAge=pd.get_dummies(AgePre['Pclass'],prefix='Pclass')
AgeCorrDf=pd.DataFrame()
AgeCorrDf=AgePre.corr()
print(AgeCorrDf['Age'].sort_values())

#
AgeKnown=AgePre[AgePre['Age'].notnull()]
AgeUnKnown=AgePre[AgePre['Age'].isnull()]
AgeKnown_X=AgeKnown.drop(['Age'],axis=1)
AgeKnown_y=AgeKnown['Age']
AgeUnKnown_X=AgeUnKnown.drop(['Age'],axis=1)
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(random_state=None,n_estimators=500,n_jobs=-1)
rfr.fit(AgeKnown_X,AgeKnown_y)
rfr.score(AgeKnown_X,AgeKnown_y)
AgeUnKnown_y=rfr.predict(AgeUnKnown_X)
every.loc[every['Age'].isnull(),['Age']]=AgeUnKnown_y
every.info()  

#filter surname
every['Surname']=every['Name'].map(lambda x:x.split(',')[0].strip())
SurNameDict={}
SurNameDict=every['Surname'].value_counts()
every['SurnameNum']=every['Surname'].map(SurNameDict)
MaleDf=every[(every['Sex']=='male')&(every['Age']>12)&(every['familyNum']>=2)]
FemChildDf=every[((every['Sex']=='female')|(every['Age']<=12))&(every['familyNum']>=2)]
#male
MSurNamDf=MaleDf['Survived'].groupby(MaleDf['Surname']).mean()
MSurNamDf.head()
print(MSurNamDf.value_counts())
MSurNamDict={}
MSurNamDict=MSurNamDf[MSurNamDf.values==1].index
MSurNamDict
#female
FCSurNamDf=FemChildDf['Survived'].groupby(FemChildDf['Surname']).mean()
FCSurNamDf.head()
print(FCSurNamDf.value_counts())
FCSurNamDict={}
FCSurNamDict=FCSurNamDf[FCSurNamDf.values==0].index
FCSurNamDict
#modification
every.loc[(every['Survived'].isnull())&(every['Surname'].isin(MSurNamDict))&(every['Sex']=='male'),'Age']=5
every.loc[(every['Survived'].isnull())&(every['Surname'].isin(MSurNamDict))&(every['Sex']=='male'),'Sex']='female'

every.loc[(every['Survived'].isnull())&(every['Surname'].isin(FCSurNamDict))&((every['Sex']=='female')|(every['Age']<=12)),'Age']=60
every.loc[(every['Survived'].isnull())&(every['Surname'].isin(FCSurNamDict))&((every['Sex']=='female')|(every['Age']<=12)),'Sex']='male'
#filter
fullSel=every.drop(['Cabin','Name','Ticket','PassengerId','Surname','SurnameNum'],axis=1)
corrDf=pd.DataFrame()
corrDf=fullSel.corr()
print(corrDf['Survived'].sort_values(ascending=True))
#heatmap
plt.figure(figsize=(8,8))
sns.heatmap(fullSel[['Survived','Age','Embarked','Fare','Parch','Pclass',
                    'Sex','SibSp','Title','familyNum','familySize','Deck',
                     ]].corr(),cmap='BrBG',annot=True,
           linewidths=.5)
plt.xticks(rotation=45)
plt.show()
fullSel=fullSel.drop(['familyNum','SibSp','Parch'],axis=1)
#one-hot
fullSel=pd.get_dummies(fullSel)
PclassDf=pd.get_dummies(every['Pclass'],prefix='Pclass')
familySizeDf=pd.get_dummies(every['familySize'],prefix='familySize')
fullSel=pd.concat([fullSel,PclassDf,familySizeDf],axis=1)

#choose model
experData=fullSel[fullSel['Survived'].notnull()]
preData=fullSel[fullSel['Survived'].isnull()]

experData_X=experData.drop('Survived',axis=1)
experData_y=experData['Survived']
preData_X=preData.drop('Survived',axis=1)

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
kfold=StratifiedKFold(n_splits=10)
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,experData_X,experData_y,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))
            
cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
#summary
cvResDf=pd.DataFrame({'cv_mean':cv_means,
                     'cv_std':cv_std,
                     'algorithm':['SVC','DecisionTreeCla','RandomForestCla','ExtraTreesCla',
                                  'GradientBoostingCla','KNN','LR','LinearDiscrimiAna']})

print(cvResDf)

#gradient boosting
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, 
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsGBC.fit(experData_X,experData_y)

#LogisticRegression
modelLR=LogisticRegression()
LR_param_grid = {'C' : [1,2,3],
                'penalty':['l1','l2']}
modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold, 
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsLR.fit(experData_X,experData_y)
print('GBC accuracy score ：%.3f'%modelgsGBC.best_score_)
#modelgsLR
print('LR accuracy score：%.3f'%modelgsLR.best_score_)


#AUCs score
modelgsGBCtestpre_y=modelgsGBC.predict(experData_X).astype(int)
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(experData_y, modelgsGBCtestpre_y)
print("ROC-AUC-Score:", r_a_score)
#f1
from sklearn.metrics import f1_score
print("f1 score is:",f1_score(experData_y, modelgsGBCtestpre_y))

GBCpreData_y=modelgsGBC.predict(preData_X)
GBCpreData_y=GBCpreData_y.astype(int)

GBCpreResultDf=pd.DataFrame()
GBCpreResultDf['PassengerId']=every['PassengerId'][every['Survived'].isnull()]
GBCpreResultDf['Survived']=GBCpreData_y
GBCpreResultDf

GBCpreResultDf.to_csv(r'C:\Users\richard\submission.csv',index=False)

