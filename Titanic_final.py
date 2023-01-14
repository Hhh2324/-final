# imports
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")
### 数据导入
test = pd.read_excel(r"C:\Users\哈铂\Desktop\test.xlsx")
train = pd.read_excel(r"C:\Users\哈铂\Desktop\train.xlsx")
print('---Train data---')
print(train.shape)
print('---Test data---')
print(test.shape)

# 合并数据集，方便同时对两个数据集进行清洗
full = train.append(test, ignore_index=True)
print('---Full data---')
print(full.shape)

### 查看数据缺失情况
print('---Missing data in the train set---')
print(train.isnull().sum())
print('---Missing data in the test set---')
print(test.isnull().sum())

# 缺失值处理：Cabin
full['Cabin'] = full['Cabin'].fillna('U')

# 缺失值处理：Age
n_bins = 60
fig, ax = plt.subplots(1, 1)
all_n, all_bins, all_patches = ax.hist(full['Age'], bins=n_bins, color="steelblue")
plt.xlabel('Age')
plt.show()
print('---description of Age---')
print(full['Age'].describe())
'''
采用中位数填补Age缺失字段
'''
med_age = full['Age'].median()
full['Age'] =full['Age'].fillna(med_age)

# 缺失值处理：Fare
n_bins = 60
fig, ax = plt.subplots(1, 1)
all_n, all_bins, all_patches = ax.hist(full['Fare'], bins=n_bins, color="steelblue")
plt.xlabel('Fare')
plt.show()
print('---description of Fare---')
print(full['Fare'].describe())
'''
采用众数填补Fare缺失字段
'''
def fill_fare(fare):
    if np.isnan(fare):
        return round(full['Fare']).value_counts().max()
    else:
        return fare

full['Fare'] = full.apply(lambda x: fill_fare(x['Fare']), axis=1)

# 缺失值处理：Embarked
sns.countplot(x="Embarked",  data=full)
plt.show()
'''
采用众数填补Embarked缺失字段
'''
common_value = train['Embarked'].value_counts().idxmax()
full['Embarked'] = full['Embarked'].fillna(common_value)

# check
print('---check---')
class CheckNull(Exception):
    def __init__(self,dat):
        self.dat = dat

    def __str__(self):
        print("存在缺失数据")

def check(dat):
    s = dat.isnull().any()
    if True in list(s):
        raise CheckNull(dat)
    else:
        print(dat.head(5))

dat = full.drop('Survived', axis=1)
check(dat)


### 各特征变量与survive的相关性初步探索性分析
train['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
plt.legend(['0=Not Survived','1=Survived'], loc='upper right', bbox_to_anchor=(1.2,1))
plt.show()

plt.figure()
# 1. Pclass
plt.subplot(3,3,1)
sns.countplot(x="Pclass", hue="Survived", data=train)

# 2. Name
def getTitle(name):
    str1 = name.split(',')[1]   # Mr. Owen Harris
    str2 = str1.split('.')[0]   # Mr
    # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3 = str2.strip()
    return str3

# map函数：对Series每个数据应用自定义的函数计算
full['Title'] = full['Name'].map(getTitle)
print(full['Title'].unique())

title_mapDict = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
# map函数：对Series每个数据应用自定义的函数计算
full['Title'] = full['Title'].map(title_mapDict)

plt.subplot(3,3,2)
sns.countplot(x="Title", hue="Survived", data=full)

# 3. Sex性别
plt.subplot(3,3,3)
sns.countplot(x="Sex", hue="Survived", data=full)

# 4. Age
plt.subplot(3,3,4)
sns.violinplot(x='Survived',y='Age',data=full)

# 5. Sibsp 兄弟姐妹和配偶的数量
plt.subplot(3,3,5)
sns.countplot(x="SibSp", hue="Survived", data=full)

# 6. Parch 父母以及小孩的数量
plt.subplot(3,3,6)
sns.countplot(x="Parch", hue="Survived", data=full)

# 7. Ticket 船票编号

# 8. Fare 船票的花费
plt.subplot(3,3,7)
sns.violinplot(x='Survived', y='Fare', data=full, ylab='Fare')

# 9. Cabin 船舱的编号
plt.subplot(3,3,8)
full['Cabin'] = full['Cabin'].map(lambda c: c[0])   #客舱号的首字母代表处于哪个，U代表不知道属于哪个船舱
sns.countplot(x="Cabin", hue="Survived", data=full)

# 10. Embarked 乘客上船的港口
plt.subplot(3,3,9)
sns.countplot(x="Embarked", hue="Survived", data=full)
plt.show()

# 5 & 6. 家庭人数
full['FamilySize'] = full["Parch"] + full["SibSp"] + 1
sns.violinplot(x='Survived',y='FamilySize',data=full)
plt.show()

# Dropping columns that are not useful
full.drop('Ticket',axis=1,inplace=True)
full.drop('SibSp',axis=1,inplace=True)
full.drop('Parch',axis=1,inplace=True)
full.drop('PassengerId',axis=1,inplace=True)

### one-hot 编码
# Pclass
pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
full = pd.concat([full,pclassDf],axis=1)
full.drop('Pclass',axis=1,inplace=True)
# Name(Title)
titleDf = pd.get_dummies(full['Title'], prefix='Title')
full = pd.concat([full, titleDf], axis=1)
full.drop('Title',axis=1,inplace=True)
full.drop('Name',axis=1,inplace=True)
# Sex
sex_mapDict = {'male':1, 'female':0}
full['Sex'] = full['Sex'].map(sex_mapDict)
# FamilySize
familyDf = pd.DataFrame()
familyDf['Family_Single'] = full['FamilySize'].map(lambda s: 1 if s == 1 else 0)
familyDf['Family_Small'] = full['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
familyDf['Family_Large'] = full['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
full = pd.concat([full, familyDf], axis=1)
full.drop('FamilySize',axis=1,inplace=True)
# Cabin
full['Cabin'] = full['Cabin'].map(lambda s: 1 if s == 'U' else 0)
# Embarked
embarkedDf = pd.get_dummies(full['Embarked'], prefix = 'Embarked')
full = pd.concat([full, embarkedDf], axis=1)
full.drop('Embarked',axis=1,inplace=True)

# 防止多重共线性
full.drop('Pclass_1',axis=1,inplace=True)
full.drop('Title_Master',axis=1,inplace=True)
full.drop('Family_Single',axis=1,inplace=True)
full.drop('Embarked_C',axis=1,inplace=True)


### 特征选择
corrDf = full.corr()

fig = plt.figure(figsize=(12,12))
ax=sns.heatmap(corrDf,vmin=-1,annot=False,cmap="rainbow")
plt.show()

print(corrDf['Survived'].sort_values(ascending =False))

# clear Age
full.drop('Age',axis=1,inplace=True)


### 模型建立
# 数据划分
sourceRow=891
source_X = full.loc[0:sourceRow-1,:]
source_X = source_X.drop('Survived', axis=1)
source_y = full.loc[0:sourceRow-1,'Survived']

pred_X = full.loc[sourceRow:,:]
pred_X = pred_X.drop('Survived', axis=1)

# 模型训练
size=np.arange(0.6,1,0.1)
scorelist=[[],[]]
for i in range(0,4):
    train_X, test_X, train_y, test_y = train_test_split(source_X ,
                                                        source_y,
                                                        train_size=size[i],
                                                        random_state=5)
    # 逻辑回归
    model = LogisticRegression()
    model.fit(train_X, train_y)
    scorelist[0].append(model.score(test_X, test_y))

    # 随机森林Random Forests Model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_X, train_y)
    scorelist[1].append(model.score(test_X, test_y))

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
color_list = ('red', 'blue')
for i in range(0,2):
    plt.plot(size, scorelist[i], color=color_list[i])
plt.legend(['逻辑回归', '随机森林'])

plt.xlabel('训练集占比')
plt.ylabel('准确率')
plt.title('不同的模型随着训练集占比变化曲线')
plt.show()


### 逻辑回归
# 参数设置
params = {'C':[0.0001, 1, 100, 1000],
          'max_iter':[1, 10, 100, 500],
          'class_weight':['balanced', None],
          'solver':['liblinear','sag','lbfgs','newton-cg']
          }
lr = LogisticRegression()
clf = GridSearchCV(lr, param_grid=params, cv=10)
clf.fit(source_X, source_y)
print('---最优参数---')
print(clf.best_params_)

logistic_model = LogisticRegression(**clf.best_params_)
logistic_model.fit(source_X, source_y)
source_pred = logistic_model.predict(source_X)

# 模型评估
class Performance:
    def __init__(self, source, pred):
        self.source = source
        self.pred = pred

    def cm(self):
        self.confusion = confusion_matrix(self.source, self.pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion)
        disp.plot()
        plt.show()

    def classify(self):
        self.accuracy = accuracy_score(self.source, self.pred)
        self.precision = precision_score(self.source, self.pred, average='macro')
        self.recall = recall_score(self.source, self.pred, average='macro')
        self.f1 = f1_score(self.source, self.pred, average='macro')
        classify = pd.DataFrame(data=[self.accuracy, self.precision, self.recall, self.f1],
                                index=['Accuracy','Precision','Recall','F1-score'],
                                columns = ['Logistic Regression'])
        print(classify)

performance = Performance(source_y, source_pred)
performance.cm()
performance.classify()

### 测试集预测
pred_Y = model.predict(pred_X)
pred_Y = pred_Y.astype(int)

passenger_id = test.loc[:,'PassengerId']
predDf = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': pred_Y
})

predDf.to_csv('titanic_pred.csv', index=False)





