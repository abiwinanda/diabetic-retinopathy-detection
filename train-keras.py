import os
import cv2
import keras
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from PIL import Image
from keras import callbacks
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from utility.model_keras import model
from utility.dataset_keras import prepare_data, create_confusion_matrix

parser = argparse.ArgumentParser(description='keras training script')
parser.add_argument('--src', '-s', help='path to dataset')
parser.add_argument('--csv', '-c', help='path to the csv file that contain the labels')
parser.add_argument('--model', '-m', help='path to the pretrained model')
parser.add_argument('--epoch', '-e', type=int, default=10, help='number of epoch to train the model')
parser.add_argument('--batch', '-b', type=int, default=8, help='batch size to train the model')
parser.add_argument('--norm', '-n', type=int, default=0, help='normalization range (0: [-1,1], else: [0,1])')
parser.add_argument('--output', '-o', default='eye-model-keras.hdf5', help='name of model output (with .hdf5 extension)')
parser.add_argument('--excel', '-x', default='confusion-matrix', help='name of excel file containing the confusion matrix (without .xlsx extension)')
args = parser.parse_args()

if __name__ == '__main__':
    if args.src == None:
        print('Please specify the path to dataset with -s flag')
    elif args.csv == None:
        print('Please specify the path to the csv file that contain the labels with -c flag')
    else:
        base_image_dir = args.src
        retina_df = pd.read_csv(args.csv)

        retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
        retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(base_image_dir,'{}.jpeg'.format(x)))
        retina_df['exists'] = retina_df['path'].map(os.path.exists)
        print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')

        # class labeling (0 => Healthy, else => DR)
        retina_df['level_binary']=retina_df['level'].map(lambda x: 0 if x == 0 else 1)

        # steps:
        # 1: get the index of label that has label or level_binary equal to 0 (healthy)
        # 2: out of all the healthy indexes, choose 5000 data randomly
        # 3: get the healthy eyes
        # 4: allocate for val
        # 5: allocate for train
        class_0 = retina_df[retina_df.level_binary == 0].index
        random_indices = np.random.choice(class_0, 5000, replace=False)
        healthy = retina_df.loc[random_indices]
        healthy_val = healthy.iloc[0:1000]
        healthy_train = healthy.iloc[1000:]

        class_4 = retina_df[retina_df.level == 4].index
        sick_4 = retina_df.loc[class_4]
        sick4_val = sick_4.iloc[0:150]
        sick4_train = sick_4.iloc[150:]

        class_3 = retina_df[retina_df.level == 3].index
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

        X_train = prepare_data(undersampled_train.path, args.norm)
        Y_train = np.array(undersampled_train.level_binary)

        X_val = prepare_data(undersampled_val.path, args.norm)
        Y_val = np.array(undersampled_val.level_binary)

        model = model(X_train.shape[1], X_train.shape[2], 2, 2e-4)
        if (args.model != None):
            model = load_model(args.model)
        model.summary()

        if os.path.exists('saved_model') == False:
            os.makedirs('saved_model')
        filepath = 'saved_model/' + args.output

        checkpoint = callbacks.ModelCheckpoint(filepath, monitor = 'val_acc', save_best_only=True, save_weights_only=False, verbose = 1)
        tensorboard = callbacks.TensorBoard(log_dir='./logdir_'+args.output.split(".")[0], batch_size=args.batch, write_images=True)
        callbacks_list = [checkpoint, tensorboard]

        model.fit(X_train, Y_train, epochs=args.epoch, batch_size = args.batch, callbacks=callbacks_list, validation_data = (X_val, Y_val))

        create_confusion_matrix(model, args.batch, X_val, Y_val, args.excel)
