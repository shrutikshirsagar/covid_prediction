import warnings
warnings.filterwarnings('ignore')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adam

from keras.datasets import mnist
import numpy as np
import scipy.io as spio
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

### data loading and normalizing


X_train = pd.read_csv('//home/amrgaballah/Desktop/Yi_zu_ICASSP/fused_train.csv').values
print(X_train.shape)
X_test = pd.read_csv('//home/amrgaballah/Desktop/Yi_zu_ICASSP/fused_test.csv').values
print(X_test.shape)
X_val = pd.read_csv('//home/amrgaballah/Desktop/Yi_zu_ICASSP/fused_val.csv').values
print(X_val.shape)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
Y_train = pd.read_csv('//home/amrgaballah/Desktop/Yi_zu_ICASSP/label_train.csv').values
print(Y_train.shape)
Y_test = pd.read_csv('//home/amrgaballah/Desktop/Yi_zu_ICASSP/label_test.csv').values
print(Y_test.shape)
Y_val = pd.read_csv('//home/amrgaballah/Desktop/Yi_zu_ICASSP/label_val.csv').values
print(Y_val.shape)

### hyperparameters
batch_size = 64
num_classes = 2
epochs = 500


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(Y_train, num_classes)
y_test = keras.utils.to_categorical(Y_test, num_classes)
y_val = keras.utils.to_categorical(Y_val, num_classes)


from keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(300, activation='relu', input_shape=(35,)))
model.add(Dropout(0.5))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))
adam_1 = Adam(lr =0.01)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=adam_1,
              metrics=['accuracy'])
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=1, nb_epoch=epochs, batch_size=batch_size, shuffle=False, callbacks=[early_stopping])


# history = model.fit(X_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(X_val, y_val))
score = model.evaluate(X_test, y_test, verbose=0)
#np.argmax(y_test, axis=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
y_val_cat_prob=model.predict(X_test)
from sklearn.metrics import roc_curve,roc_auc_score
auc_score=roc_auc_score(y_test,y_val_cat_prob)
 
print('auc_score', auc_score)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
class_names = ['no covid', 'covid']


y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)
accuracy_score(y_test, y_pred, normalize=False)
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
