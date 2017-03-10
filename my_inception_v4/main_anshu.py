import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  
from keras.callbacks import ModelCheckpoint, History
from keras.utils import np_utils
from my_inception_v4 import create_model_Inception_v4
from data_pre_new import x_train, y_train,x_val, y_val
model = create_model_Inception_v4()
weightsOutputFile = 'inception-v4.{epoch:02d}-{val_precision:.3f}.hdf5'

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

batch_size = 30
nb_epoch = 30

model.compile(optimizer='adagrad',loss='categorical_crossentropy',metrics=['precision'])
checkpointer = ModelCheckpoint(weightsOutputFile, monitor='val_precision', save_best_only=False, mode='auto')
history = History()
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,validation_data=(x_val, y_val), callbacks=[checkpointer, history])

logging.basicConfig(filename='model.log', level=logging.INFO)
logging.info(history.history)
fig=plt.figure()
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Inception-V4 Loss Trend')
plt.plot(loss, 'blue',label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch))
plt.legend()
fig = plt.gcf()
fig.savefig("Inception-V4-Loss-Trend.png")
fig.clf()

fig=plt.figure()
precision = history.history['precision']
val_precision = history.history['val_precision']

plt.xlabel('Epochs')
plt.ylabel('precision')
plt.title('Inception-V4 precision Trend')
plt.plot(precision, 'blue',label='Training precision')
plt.plot(val_precision, 'green', label='Validation precision')
plt.xticks(range(0,nb_epoch))
plt.legend()
fig = plt.gcf()
fig.savefig("Inception-V4-precision-Trend.png")
fig=plt.figure()
