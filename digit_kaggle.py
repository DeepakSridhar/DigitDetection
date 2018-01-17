# Load in our libraries
import pandas as pd
import time
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from misc import *
from inception_network import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

from model_keras import DigitModel
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import scipy.io as spio
from kt_utils import *
from one_hot import one_hot_matrix
from export_model import export_model
import time
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.callbacks import TensorBoard

import keras.backend as K

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

start_time=time.clock()

# Load in the train and test datasets
train = pd.read_csv('data/digit_train.csv')
test = pd.read_csv('data/digit_test.csv')


# Create Numpy arrays of train and test dataframes to feed into our models
y_train = train['label'].ravel()
train = train.drop(['label'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

x_train = x_train.reshape(-1,1,28,28)
x_test = x_test.reshape(-1,1,28,28)


train_x=x_train/255
test_x=x_test/255
train_y=one_hot_matrix(y_train,10)
train_y=train_y.T

train_x, cross_val_x,train_y, cross_val_y = train_test_split(train_x, train_y, test_size = 0.02, random_state=2)

print ("number of training examples = " + str(train_x.shape[0]))
print ("number of test examples = " + str(test_x.shape[0]))
print ("X_train shape: " + str(train_x.shape))
print ("Y_train shape: " + str(train_y.shape))
print ("X_test shape: " + str(test_x.shape))
print ("X_val shape: " + str(cross_val_x.shape))
print ("Y_val shape: " + str(cross_val_y.shape))

np.set_printoptions(threshold=np.nan)

# Dmodel = faceRecoModel(input_shape=(1, 28, 28))
#
# print("Total Params:", Dmodel.count_params())

# model = DigitModel((train_x.shape[1],train_x.shape[2],train_x.shape[3]))
#
# model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])

tensorboard = TensorBoard(log_dir="logs/digit_3")

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(train_x)


model = load_model('out/digit_model_3.h5')
print("Model loaded!")

model.fit_generator(datagen.flow(train_x, train_y, batch_size=32),callbacks=[tensorboard], samples_per_epoch=len(train_x),epochs=3)

# model.fit(x=train_x,y=train_y,callbacks=[tensorboard],epochs=20,batch_size=32)

path = 'out/digit_model_3.h5'
model.save(path)
print("Model saved!")

val_preds = model.evaluate(x=cross_val_x,y=cross_val_y)
print()
print ("Loss = " + str(val_preds[0]))
print ("Validation Accuracy = " + str(val_preds[1]))

preds = model.predict(x=test_x)
preds = preds.reshape(preds.shape[0],10)
preds = np.argmax(preds, axis=1) #convert back from one-hot encoding
print(preds[:10])
# model = load_model('out/digit_model_3.h5')
# print("Model loaded!")

# model.fit(x=train_x,y=train_y,callbacks=[tensorboard],epochs=20,batch_size=128)

# path = 'out/digit_model_3.h5'
# model.save(path)
# print("Model saved!")

ImageId = np.arange(1,test_x.shape[0]+1)

print()

titanic_deep = pd.DataFrame({ 'ImageId': ImageId,
                            'Label': preds })
titanic_deep.to_csv("digit_deep_3.csv", index=False)

end_time =time.clock()
print("Elapsed time: " + str(end_time-start_time))

# Y_pred = model.predict(X_val)
# # Convert predictions classes to one hot vectors
# Y_pred_classes = np.argmax(Y_pred,axis = 1)
# # Convert validation observations to one hot vectors
# Y_true = np.argmax(Y_val,axis = 1)
# # compute the confusion matrix
# confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# # plot the confusion matrix
# plot_confusion_matrix(confusion_mtx, classes = range(10))

# # Errors are difference between predicted labels and true labels
# errors = (Y_pred_classes - Y_true != 0)
#
# Y_pred_classes_errors = Y_pred_classes[errors]
# Y_pred_errors = Y_pred[errors]
# Y_true_errors = Y_true[errors]
# X_val_errors = X_val[errors]
#
# def display_errors(errors_index,img_errors,pred_errors, obs_errors):
#     """ This function shows 6 images with their predicted and real labels"""
#     n = 0
#     nrows = 2
#     ncols = 3
#     fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
#     for row in range(nrows):
#         for col in range(ncols):
#             error = errors_index[n]
#             ax[row,col].imshow((img_errors[error]).reshape((28,28)))
#             ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
#             n += 1
#
# # Probabilities of the wrong predicted numbers
# Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
#
# # Predicted probabilities of the true values in the error set
# true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
#
# # Difference between the probability of the predicted label and the true label
# delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
#
# # Sorted list of the delta prob errors
# sorted_dela_errors = np.argsort(delta_pred_true_errors)
#
# # Top 6 errors
# most_important_errors = sorted_dela_errors[-6:]
#
# # Show the top 6 errors
# display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

# frozen_graph = freeze_session(K.get_session(), output_names=["fc3/Softmax"])

# tf.train.write_graph(frozen_graph, "out/", "digit_model.pb", as_text=False)



# # model.summary()
# # for n in tf.get_default_graph().as_graph_def().node:
# #     print(n.name)
# # export_model(tf.train.Saver(),model, ["dense_1_input"], "dense_2/Softmax")
# # plot_model(happyModel, to_file='HappyModel.png')
# # SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))