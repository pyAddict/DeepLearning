from keras_all_code import Multiclass_Without_Grid_Search,Binary_Without_Grid_Search,Binary_With_Grid_Search,Regression_Without_Grid_Search,Regression_With_Grid_Search
import numpy as np
import pandas as pd
from keras.utils import np_utils
# Multiclass Testing
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('./iris.data.csv', delimiter=',', header=None)
data = df.values
x = data[
    :, 0:4].astype(float)
y = data[:, 4]

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(
    y)
dummy_y = np_utils.to_categorical(
    encoded_y)
obj = Multiclass(
    x, dummy_y)
obj._model_build()

# Binary without grid search Testing
data = np.loadtxt('./pima-indians-diabetes.data.csv', delimiter=',')
x = data[:, 0:8]
y = data[:, 8]
obj = Binary_Without_Grid_Search(x,y)
obj._model_build()

# Binary with grid search Testing
data = np.loadtxt('./pima-indians-diabetes.data.csv', delimiter=',')
x = data[:, 0:8]
y = data[:, 8]
obj = Binary_With_Grid_Search(x,y)
obj._model_build()

# Regression_without_Grid_Search
load dataset
df = pd.read_csv("./housing.data.csv", delim_whitespace=True, header=None)
data = df.values
# split into input (X) and output (Y) variables
x = data[:,0:13]
y = data[:,13]

obj = Regression_Without_Grid_Search(x,y)
obj._model_build()

# Regression_with_Grid_Search
load dataset
df = pd.read_csv("./housing.data.csv", delim_whitespace=True, header=None)
data = df.values
# split into input (X) and output (Y) variables
x = data[:,0:13]
y = data[:,13]

obj = Regression_With_Grid_Search(x,y)
obj._model_build()