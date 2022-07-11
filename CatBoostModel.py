import joblib
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import imblearn


class Model:
    cat_features = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    catBoost = None
    n_estimators = None
    criterion = 'gini'
    max_depth = None
    max_samples = None
    n_jobs = -1
    mappings = {}
    objectType = None
    boolType = None
    data = None
    train_dataset = None
    eval_dataset = None

    def __init__(self):
        self.catBoost = CatBoostClassifier(iterations=3000,
                                           depth=2,
                                           # learning_rate=1,
                                           loss_function='MultiClass',
                                           # custom_loss = ["AUC","Accuracy"],
                                           train_dir="MultiClass",
                                           verbose=True)

    def train(self,train_dataset):
        self.catBoost.fit(train_dataset, plot=True)

    def predict(self, test_data):
        y_pred = self.catBoost.predict(test_data)
        return y_pred

    def load(self):
        self.catBoost = joblib.load("./random_forest.joblib")

    def save(self, path):
        joblib.dump(self.catBoost, path)

    def loadData(self):
        import pandas as pd
        df = pd.read_csv('ActualDatasets.csv', skipinitialspace=True, sep=';')
        return df

    def clearData(self,data):
        data = data.drop("WordsCount", axis=1)
        data = data.drop("LeftIndentation", axis=1)
        data = data.drop("SpecialIndentation", axis=1)
        data = data.drop("SpaceBefore", axis=1)
        data = data.drop("SpaceAfter", axis=1)
        data = data.drop("RightIndentation", axis=1)
        data = data.drop("LineSpacing", axis=1)
        data = data.drop("Alignment", axis=1)
        data = data.drop("Content", axis=1)
        self.target_names = list(data['CurElementMark'].unique())
        self.objectType = list(data.select_dtypes(['object']).columns)
        self.boolType = list(data.select_dtypes(['bool']).columns)
        return data

    def mapping(self,data, boolType, objectType):
        from sklearn import preprocessing
        from sklearn.preprocessing import LabelEncoder
        le = preprocessing.LabelEncoder()
        for i in self.boolType:
            data[i] = data[i].astype(int)
        for i in self.objectType:
            data[i] = data[i].fillna('0')
            le.fit(data[i])
            np.save(i + '.npy', le.classes_)
            data[i] = le.transform(data[i])
            self.mappings[i] = dict(zip(le.classes_, le.transform(le.classes_)))

    def samplingData(datax, datay):
        oversample = RandomOverSampler(sampling_strategy='minority')
        X_over, y_over = oversample.fit_resample(datax, datay)
        X_over, y_over = oversample.fit_resample(X_over, y_over)
        X_over, y_over = oversample.fit_resample(X_over, y_over)
        undersample = RandomUnderSampler(sampling_strategy='majority')
        X_over, y_over = undersample.fit_resample(X_over, y_over)
        return train_test_split(X_over, y_over, test_size=0.35, random_state=1)

    def prepare_data(self,data_train_x, data_train_y, data_test_x, data_test_y):
        train_dataset = Pool(data=data_train_x,
                             label=data_train_y,
                             cat_features=self.cat_features)
        eval_dataset = Pool(data=data_test_x,
                             label=data_test_y,
                             cat_features=self.cat_features)
        return train_dataset, eval_dataset

model = Model()
data = model.loadData()
data = model.clearData(data)
mappings = model.mapping(data, model.boolType, model.objectType);
y = data["CurElementMark"].to_numpy()
x = data.drop("CurElementMark", axis = 1).to_numpy()
X_train, X_test, y_train, y_test = Model.samplingData(datax=x, datay=y)
train_dataset, eval_dataset = model.prepare_data(X_train,y_train,X_test,y_test)
model.train(train_dataset)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
