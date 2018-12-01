import pandas as pd
import os
import keras
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import cv2
import numpy as np
from model import model
from PIL import Image
import tifffile as tiff
from keras import callbacks
from keras.models import load_model

#base_image_dir = "./train_preprocessed/"
#retina_df = pd.read_csv('trainLabels.csv')

base_image_dir = "./mixed-dataset/preprocessed_3channel/"
retina_df = pd.read_csv("./mixed-dataset/trainLabels.csv")

retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(base_image_dir,'{}.jpeg'.format(x)))
retina_df['exists'] = retina_df['path'].map(os.path.exists)
print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')

#retina_df.dropna(inplace = True)
#retina_df = retina_df[retina_df['exists']]
retina_df['level_binary']=retina_df['level'].map(lambda x: 0 if x == 0 else 1)

#retina_df = retina_df[retina_df.exists == True]

class_0 = retina_df[retina_df.level_binary == 0].index
random_indices = np.random.choice(class_0, 5000, replace=False)
healthy = retina_df.loc[random_indices]
healthy_val = healthy.iloc[0:1000]
healthy_train = healthy.iloc[1000:]

class_4 = retina_df[retina_df.level == 4].index
#random_indices = np.random.choice(class_4, 5000, replace=False)
sick_4 = retina_df.loc[class_4]
sick4_val = sick_4.iloc[0:150]
sick4_train = sick_4.iloc[150:]

class_3 = retina_df[retina_df.level == 3].index
#random_indices = np.random.choice(class_4, 5000, replace=False)
sick_3 = retina_df.loc[class_3]
sick3_val = sick_3.iloc[0:150]
sick3_train = sick_3.iloc[150:]

class_2 = retina_df[retina_df.level == 2].index
random_indices = np.random.choice(class_2, len(class_2), replace=False)
sick_2 = retina_df.loc[class_2]
sick2_train = sick_2.iloc[0:1500]
sick2_val = sick_2.iloc[1500:1850]

class_1 = retina_df[retina_df.level == 1].index
random_indices = np.random.choice(class_1, len(class_1), replace=False)
sick_1 = retina_df.loc[class_1]
sick1_train = sick_1.iloc[0:1219]
sick1_val = sick_1.iloc[1219:1569]

undersampled_train = pd.concat([healthy_train, sick1_train, sick2_train, sick3_train, sick4_train])
undersampled_train = undersampled_train.sample(frac=1).reset_index(drop=True)

undersampled_val = pd.concat([healthy_val, sick1_val, sick2_val, sick3_val, sick4_val])
undersampled_val = undersampled_val.sample(frac=1).reset_index(drop=True)


def prepare_data(image_path):
    """
    Preparing image data.
    :param image_path: list of image to be processed
    
    Return: 
            x: array of resized images
            y: array of labels
    """
    count = len(image_path)
    x = np.ndarray((count, 512, 512, 3), dtype=np.float32)

    #Generate input image
    for i, image in enumerate(image_path):
        if i%500 == 0: print('Processed {} of {}'.format(i, count))
        #img = tiff.imread(image)
        img = cv2.imread(image)
        x[i] = img
    print("Preprocessing done!")
    print("proceed to normalization [-1, 1]")
    x *= (2/255)
    x -= 1
    print("Normalization done!")
    return x

X_train = prepare_data(undersampled_train.path)
Y_train = np.array(undersampled_train.level_binary)

X_val = prepare_data(undersampled_val.path)
Y_val = np.array(undersampled_val.level_binary)

#X = np.concatenate((X_train,X_val), axis=0)
#Y = np.concatenate((Y_train,Y_val), axis=0)

model = model(X_train.shape[1], X_train.shape[2], 2, 2e-4)
#model = load_model("./saved_model/xception_dr.hdf5")
model.summary()
batch_size = 8

model_name = "xception_dr_v5.hdf5"
model_dir = "saved_model"
if os.path.exists("saved_model") == False:
    os.makedirs("saved_model")
    
filepath = model_dir + '/' + model_name

#os.mkdir("modell")
#filepath = "CityScapes.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor = 'val_acc', save_best_only=True, save_weights_only=False,
                                  verbose = 1)
tensorboard = callbacks.TensorBoard(log_dir='./logdir_'+model_name.split(".")[0], batch_size=batch_size, write_images=True)

callbacks_list = [checkpoint, tensorboard]

model.fit(X_train, Y_train, epochs=10, batch_size = batch_size, callbacks=callbacks_list, 
          validation_data = (X_val, Y_val))


Y_pred = model.predict(X_val, batch_size = 8)
Y_pred = np.around(Y_pred).astype(int)
from sklearn.metrics import confusion_matrix
result = confusion_matrix(Y_val, Y_pred)
print(result)