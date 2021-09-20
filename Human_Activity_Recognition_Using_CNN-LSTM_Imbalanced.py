#!/usr/bin/env python
# coding: utf-8

# ## Human Activity Recognition Using CNN-LSTM

# Ever wondered how our smartphone, smartwatch or wristband knows when we are walking, running or sitting? Well, our device probably has multiple sensors that give various information. GPS, audio (i.e. microphones), image (i.e. cameras), direction (i.e. compasses) and acceleration sensors are very common nowadays.
# 
# ![](simulinkandroidsupportpackage_galaxys4_accelerometer.png)
# 
# We will use data collected from accelerometer sensors. Virtually every modern smartphone has a tri-axial accelerometer that measures acceleration in all three spatial dimensions. Additionally, accelerometers can detect device orientation.
# 
# In this project, I will train an CNN-LSTM Neural Network (implemented in TensorFlow) for Human Activity Recognition (HAR) from accelerometer data. The trained model will be exported/saved and added to an Android app.
# 

# In[1]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 10, 8


# # Importing the data
# 
# We will use data provided by the [Wireless Sensor Data Mining (WISDM) Lab](http://www.cis.fordham.edu/wisdm/). It can be download from [here](http://www.cis.fordham.edu/wisdm/dataset.php). The dataset was collected in controlled, laboratory setting. The lab provides another dataset collected from real-world usage of a smartphone app. You're free to use/explore it as well.
# 
# Our dataset contains 1,098,207 rows and 6 columns. There are no missing values. There are 6 activities that we'll try to recognize: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing.

# In[2]:


# Loading dataset
columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
data = pd.read_csv(r'data/data_csv.csv', header = None, names = columns)
data = data.dropna()


# In[3]:


data.head()


# In[4]:


data.info()


# # Exploration
# 
# The columns we will be most interested in are activity, x-axis, y-axis and z-axis. Let's dive into the data:

# In[5]:


# Comparing the number of datas for each of class in a bar-graph
data['activity'].value_counts().plot(kind='bar', title='Training examples by activity type');


# In[6]:


# Displayng number of datas shared by 30 different users in a bar-graph
data['user'].value_counts().plot(kind='bar', title='Training examples by user');


# In[7]:


# Plotting the graph for each of activities in 200 timesteps
def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']][:200]
    axis = data.plot(subplots=True, figsize=(16, 10), 
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))


# In[8]:


activities = data['activity'].value_counts().index
activities


# In[9]:


for activity in activities:
    plot_activity(activity,data)


# # DATA PREPROCESSING: Standardizing and Frame Preparation
# 

# BALANCING DATA

# In[13]:


df = data.drop(['user', 'timestamp'], axis = 1).copy()
df.head()


# In[14]:


df.tail()


# In[15]:


df['activity'].value_counts()


# STANDARDIZING DATA

# In[17]:


X = df[['x-axis', 'y-axis', 'z-axis']]
y = df['activity']


# In[18]:


X.shape


# In[19]:


y.shape


# In[20]:


# Standardizing the values of features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data = X, columns = ['x-axis', 'y-axis', 'z-axis'])
scaled_X['activity'] = y.values

scaled_X


# FRAME PREPARATION

# In[21]:


# Frame preparation of the data
N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
segments = []
labels = []
for i in range(0, len(scaled_X) - N_TIME_STEPS, step):
    xs = scaled_X['x-axis'].values[i: i + N_TIME_STEPS]
    ys = scaled_X['y-axis'].values[i: i + N_TIME_STEPS]
    zs = scaled_X['z-axis'].values[i: i + N_TIME_STEPS]
    label = stats.mode(scaled_X['activity'][i: i + N_TIME_STEPS])[0][0]
    segments.append([xs, ys, zs])
    labels.append(label)


# In[22]:


np.array(segments).shape


# In[23]:


labels


# In[24]:


reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)


# In[25]:


# Converting the label into a one-hot encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


# In[26]:


label_encoder.classes_


# In[27]:


reshaped_segments.shape


# Let's split the data into training and test (20%) set:

# In[28]:


# Splitting the training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, onehot_encoded, test_size=0.2)


# In[29]:


len(X_train)


# In[30]:


len(X_test)


# In[31]:


X_train.shape


# In[32]:


X_train[0],y_train[0]


# ## BUILDING THE CNN-LSTM MODEL

# In[48]:


#Importing Required Libraries for the model
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers import Conv1D, MaxPooling1D, LSTM


# In[49]:


# Creating model
model = Sequential([Conv1D(filters=128, kernel_size=3, activation='relu',padding = 'same',input_shape=(200,3), name = 'cnn_1'),
    MaxPooling1D(pool_size=2, name = 'maxpooling_1'),
    Conv1D(filters=128, kernel_size=3, activation='relu',padding = 'same', name = 'cnn_2'),
    BatchNormalization(name='batchnorm_layer'),
    MaxPooling1D(pool_size=2, name = 'maxpooling_2'),
         
    LSTM(64, return_sequences = True, name = 'lstm_1'),
    Dropout(0.3, name = 'Dropout_1'),
    LSTM(32, return_sequences = False, name = 'lstm_2'),
    Dropout(0.3, name = 'Dropout_2'),

    Dense(32, activation= 'relu', name = 'dense_1'),
    Dropout(0.3, name = 'Dropout_3'),
    Dense(6, activation = 'softmax', name = 'output')])


# In[50]:


# Compiling the model
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.summary()


# ## TRAINING THE MODEL

# In[51]:


# prepare callbacks
from keras.callbacks import ModelCheckpoint

callbacks= [ModelCheckpoint('my_model_imbalanced.h5', save_weights_only=False, save_best_only=True, verbose=1)]


# In[52]:


# training the model
from timeit import default_timer as timer
start = timer()

history = model.fit(X_train, y_train, epochs = 15, validation_data = (X_test,y_test), verbose=1, callbacks = [callbacks])

end = timer()
print("\n")
print("Time: ",(end - start),"secs = ",(end - start)/3600,"hours")


# ## EVALUATION AND THE PERFORMANCE METRICS

# In[62]:


# Plotting loss and accuracy graph
plt.figure(figsize=(12, 8))

plt.plot(np.array(history.history['loss']), "r--", label="Train loss")
plt.plot(np.array(history.history['accuracy']), "g--", label="Train accuracy")

plt.plot(np.array(history.history['val_loss']), "r-", label="Test loss")
plt.plot(np.array(history.history['val_accuracy']), "g-", label="Test accuracy")

plt.title("Training session's progress over iterations(imbalanced data)")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training Epoch')
plt.ylim(0)

plt.show()


# In[54]:


# Evaluating model
model.evaluate(X_test, y_test)


# In[55]:


# Prediction of testing data
test_pred = np.argmax(model.predict(X_test), axis=1)


# In[56]:


test_pred.shape


# In[57]:


test_pred[100]


# CONFUSION MATRIX

# In[58]:


# Confusion Matrix for this classificatin
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[59]:


max_test = np.argmax(y_test, axis=1)
mat = confusion_matrix(max_test, test_pred)
plot_confusion_matrix(conf_mat=mat, class_names=label_encoder.classes_, show_normed=True, figsize=(7,7))


# In[60]:


# Displaying classification report
from sklearn.metrics import classification_report
print(classification_report(label_encoder.classes_[max_test], label_encoder.classes_[test_pred]))


# In[ ]:





# In[ ]:




