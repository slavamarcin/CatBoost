import joblib
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import imblearn


class Model:
    catBoost = None
    n_estimators = None
    criterion = 'gini'
    max_depth = None
    max_samples = None
    n_jobs = -1
    def __init__(self):
        self.catBoost = CatBoostClassifier(iterations=3000,
                           depth=2,
                          # learning_rate=1,
                           loss_function='MultiClass',
                           # custom_loss = ["AUC","Accuracy"],
                            train_dir="MultiClass",
                            verbose=True)

    def train(self, train_data, eval_data):
        self.catBoost.fit(train_data,eval_set = eval_data,plot=True)

    def predict(self, test_data):
        y_pred = self.catBoost.predict(test_data)
        return y_pred
    def load(self):
        self.catBoost = joblib.load("./random_forest.joblib")

    def save(self,path):
        joblib.dump(self.catBoost, path)

    def loadData(self):
        import pandas
      #  excel_data_df = pandas.read_csv('mainDataset.csv', on_bad_lines='skip', delimiter=";")
        data = np.genfromtxt("NEWDATA3.csv",delimiter=';',encoding = 'UTF-8', dtype=None)
        data = data[1:,::]
        datax = np.delete(data, 8, 1)[::, ::]
        datay = data[::, 8]
        oversample = RandomOverSampler(sampling_strategy='minority')
        X_over, y_over = oversample.fit_resample(datax, datay)
     #   X_over, y_over = oversample.fit_resample(X_over, y_over)
     #   X_over, y_over = oversample.fit_resample(X_over, y_over)
     #   X_over, y_over = oversample.fit_resample(X_over, y_over)
        return train_test_split(X_over,y_over,test_size=0.3)
    def prepare_data(data_train_x,data_train_y,data_test_x,data_test_y):
        '''
        train_y = train[::, 8]
        train_x = np.delete(train, 8, 1)[::, ::]
        train_y = train_y.reshape(len(train_y))
        test_y = test[::, 8]
        test_y = test_y.reshape(len(test_y))
        test_x = np.delete(test, 8, 1)[::, ::]
        '''
        cat_features = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14, 15]

        train_dataset = Pool(data=data_train_x,
                             label=data_train_y,
                             cat_features=cat_features)

        eval_dataset = Pool(data=data_test_x,
                            label=data_test_y,
                            cat_features=cat_features)
        return train_dataset,eval_dataset,train_y

model = Model()
train_x,test_x,train_y,test_y = model.loadData()
train_dataset,eval_dataset, train_y = Model.prepare_data(train_x,train_y,test_x,test_y)
model.train(train_dataset,eval_dataset)
y_pred = model.predict(eval_dataset)

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_pred, test_y)
print(CM)
