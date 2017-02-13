import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import load_model
from keras_all_code import Multiclass_Without_Grid_Search
train = pd.read_csv('./train_modified.csv')
test = pd.read_csv('./test_modified.csv')
y = np.array(train['Surge_Pricing_Type'])
x = np.array(train.drop(['Surge_Pricing_Type'],axis=1))
x_test = np.array(test.drop(['Surge_Pricing_Type'],axis=1))

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(
    y)
dummy_y = np_utils.to_categorical(
    encoded_y)
test2 = pd.read_csv('/home/delhivery/Downloads/test_XaoFywY.csv')
model = load_model('./model.h5')
yy_pred = model.predict(x_test)
y_pred = np.argmax(yy_pred, axis=1)
print(len(y_pred))
print(len(test2.Trip_ID))
sub = pd.DataFrame({'Trip_ID':np.array(test2.Trip_ID),'Surge_Pricing_Type':np.array(y_pred)})
sub.to_csv('./my_sub2.scv',index=None)
# obj = Multiclass_Without_Grid_Search(x,dummy_y)
# obj._model_build()