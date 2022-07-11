import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

mappings = {}
objectType = None
boolType = None
data = None
target_names = None

def loadData():
    import pandas as pd
    df = pd.read_csv('ActualDatasets.csv', skipinitialspace=True, sep=';')
    return df

def clearData(data,mappings, objectType, boolType):
    data = data.drop("WordsCount", axis=1)
    data = data.drop("LeftIndentation", axis=1)
    data = data.drop("SpecialIndentation", axis=1)
    data = data.drop("SpaceBefore", axis=1)
    data = data.drop("SpaceAfter", axis=1)
    data = data.drop("RightIndentation", axis=1)
    data = data.drop("LineSpacing", axis=1)
    data = data.drop("Alignment", axis=1)
    data = data.drop("Content", axis=1)
    target_names = list(data['CurElementMark'].unique())
    objectType = list(data.select_dtypes(['object']).columns)
    boolType = list(data.select_dtypes(['bool']).columns)
    return data,target_names,objectType,boolType


def mapping(data, boolType, objectType):
    from sklearn import preprocessing
    from sklearn.preprocessing import LabelEncoder
    le = preprocessing.LabelEncoder()
    for i in boolType:
        data[i] = data[i].astype(int)
    for i in objectType:
        data[i] = data[i].fillna('0')
        le.fit(data[i])
        np.save(i + '.npy', le.classes_)
        data[i] = le.transform(data[i])
        mappings[i] = dict(zip(le.classes_, le.transform(le.classes_)))


def samplingData(datax, datay):
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(datax, datay)
    X_over, y_over = oversample.fit_resample(X_over, y_over)
    X_over, y_over = oversample.fit_resample(X_over, y_over)
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_over, y_over = undersample.fit_resample(X_over, y_over)
    return train_test_split(X_over, y_over, test_size=0.35, random_state=1)

data = loadData()
data,target_names,objectType,boolType = clearData(data,target_names,objectType,boolType)
mappings = mapping(data, boolType, objectType);
y = data["CurElementMark"].to_numpy()
x = data.drop("CurElementMark", axis = 1).to_numpy()
X_train, X_test, y_train, y_test = samplingData(datax=x, datay=y)
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
dummy_y = np_utils.to_categorical(encoded_Y)

encoder1 = LabelEncoder()
encoder1.fit(y_test)
encoded_Y1 = encoder1.transform(y_test)
dummy_y1 = np_utils.to_categorical(encoded_Y1)


from keras.models import Sequential
from keras.layers import Dense, Activation
modelsq = Sequential()
modelsq.add(Dense(1024, input_shape=(None,1,15),activation='sigmoid'))
modelsq.add(Dense(512,activation='sigmoid'))
modelsq.add(Dense(11,activation='softmax'))
modelsq.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
modelsq.fit(X_train,dummy_y,validation_data=(X_test,dummy_y1), epochs=100, verbose=2)