#!/usr/bin/env python
# coding: utf-8

# **Face detection**
# Task is to predict the boundaries(mask) around the face in a given image.
# 
# Dataset
# Faces in images marked with bounding boxes. Have around 500 images with around 1100 faces manually tagged via bounding box.

# **Mount Google drive if you are using google colab**

# In[ ]:


import tensorflow as tf 
tf.test.gpu_device_name()


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# **Load the "images.npy" file (4 marks)**

# In[ ]:


import numpy as np
filepath = '/content/gdrive/My Drive/Colab Notebooks/Face Detection Project/'
file = filepath + 'images.npy'
data = np.load(file,allow_pickle=True)


# In[ ]:


print (data.shape)
print (data.size)
print (data.itemsize)
print(type(data))
print (data.dtype)


# **Check one sample from the loaded "images.npy" file (4 marks)**

# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
arr = data[220][0]
img = Image.fromarray(arr, mode="RGB")
plt.imshow(img)
plt.show()


# **Set image dimensions (2 marks)**
# 
# Initialize image height, image width with value: 224

# In[ ]:


width, height = img.size
print (width, height)

new_size= 224, 224
print (new_size)
img = img.resize(new_size)
print(img.size)
plt.imshow(img)


# In[ ]:


''' Resize and Print all images
print(data.size)
for i in range(data.size):
  arr = data[i][0]
  img = Image.fromarray(arr, mode="RGB")
  width, height = img.size
  new_size= 224, 224
  img = img.resize(new_size)
  plt.imshow(img)
  plt.show()

'''


# In[ ]:


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


# Create features and labels
# 
# *   The label is the mask
# *   Here feature is the image
# *   Images will be stored in "X_train" array
# *   Masks will be stored in "masks" array

# In[ ]:


import cv2
from tensorflow.keras.applications.mobilenet import preprocess_input

masks = np.zeros((int(data.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH))
X_train = np.zeros((int(data.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH, 3))
for index in range(data.shape[0]):
    img = data[index][0]
    img = cv2.resize(img, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
    try:
      img = img[:, :, :3]
    except:
      continue
    X_train[index] = preprocess_input(np.array(img, dtype=np.float32))
    for i in data[index][1]:
        x1 = int(i["points"][0]['x'] * IMAGE_WIDTH)
        x2 = int(i["points"][1]['x'] * IMAGE_WIDTH)
        y1 = int(i["points"][0]['y'] * IMAGE_HEIGHT)
        y2 = int(i["points"][1]['y'] * IMAGE_HEIGHT)
        masks[index][y1:y2, x1:x2] = 1


# **Print the shape of X_train and mask array (1 mark)**

# In[ ]:


print (X_train.shape)
print (masks.shape)


# **Print a sample image and image array**

# In[ ]:


from matplotlib import pyplot
n = 6
print(X_train[n])

sample_image = X_train[n]
plt.imshow(sample_image)


# In[ ]:


pyplot.imshow(masks[n])


# Create the model (10 marks)
# 
# 1.   Add MobileNet as model with below parameter values
# 
# *   input_shape: IMAGE_HEIGHT, IMAGE_WIDTH, 3
# *   include_top: False
# *   alpha: 1.0
# *   weights: "imagenet"
# 
# 2.   Add UNET architecture layers
# 
# *   This is the trickiest part of the project, you need to research and implement it correctly

# In[ ]:


from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Concatenate, UpSampling2D, Conv2D, Reshape
from tensorflow.keras.models import Model
ALPHA = 1 

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def create_model(trainable=True):
    model = MobileNet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH,3), include_top=False, alpha=ALPHA, weights="imagenet")
    
    for layer in model.layers:
        layer.trainable = trainable

    block = model.get_layer("conv_pw_1_relu").output
    block1 = model.get_layer("conv_pw_3_relu").output
    block2 = model.get_layer("conv_pw_5_relu").output
    block3 = model.get_layer("conv_pw_11_relu").output
    block4 = model.get_layer("conv_pw_13_relu").output

    x = Concatenate()([UpSampling2D()(block4), block3])
    x = Concatenate()([UpSampling2D()(x), block2])
    x = Concatenate()([UpSampling2D()(x), block1])
    x = Concatenate()([UpSampling2D()(x), block])

    x = UpSampling2D()(x)
    x = Conv2D(1, kernel_size=1, activation="sigmoid")(x)

    x = Reshape((IMAGE_HEIGHT, IMAGE_WIDTH))(x)

    return Model(inputs=model.input, outputs=x)


# **Call the create_model function**

# In[ ]:


# Give trainable=False as argument, if you want to freeze lower layers for fast training (but low accuracy)
model = create_model()

# Print summary
model.summary()


# **Define dice coefficient function (5 marks)**
# Create a function to calculate dice coefficient

# In[ ]:


def dice_coefficient(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator + tf.keras.backend.epsilon())


# **Define loss**

# In[ ]:


from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import log, epsilon
def loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - log(dice_coefficient(y_true, y_pred) + epsilon())


# **Compile the model (5 marks)**
# Complie the model using below parameters
# loss: use the loss function defined above
# optimizers: use Adam optimizer
# metrics: use dice_coefficient function defined above

# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile('Adam', loss=loss, metrics=[dice_coefficient])


# **Define checkpoint and earlystopping**

# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
checkpoint = ModelCheckpoint("model-{loss:.2f}.h5", monitor="loss", verbose=1, save_best_only=True,
                             save_weights_only=True, mode="min", period=1)
stop = EarlyStopping(monitor="loss", patience=5, mode="min")
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="min")


# In[ ]:


print (X_train.shape)
print (masks.shape)


# **Fit the model (5 marks)**
# Fit the model using below parameters
# epochs: you can decide
# batch_size: 1
# callbacks: checkpoint, reduce_lr, stop

# In[21]:


model.fit(X_train, masks, batch_size=1, epochs=10, verbose=1, shuffle=True, callbacks=[checkpoint, reduce_lr, stop])
# model.fit(X_train, masks, batch_size=1, epochs=10)


# In[22]:


# Save Model to further use it

from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# In[ ]:


model.load_weights('model.h5')


# In[24]:


X_train[10].shape


# In[25]:


img1=np.expand_dims(X_train[10], axis=0)
print(img1.shape)


# **Get the predicted mask for a sample image (5 marks)**

# In[26]:


preds_train = model.predict(img1, verbose=1)
print(preds_train.shape)


# In[27]:


print(X_train.shape)
print(preds_train[0].shape)
mask1=np.stack([preds_train[0]]*3, axis=-1)
print(mask1.shape)


# In[28]:


plt.imshow(mask1)


# **Impose the mask on the image (5 marks)**

# In[29]:


plt.imshow(X_train[10]-mask1)

