import os
import shutil
import pandas as pd
import torch

from os import listdir
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def seperate_dataset_to_labels_folder(src, dst, labels_csv, label_rule, eff=1, train_test_split=0.8):
    # check if src folder exist
    if (not os.path.exists(src)):
        print('source path does not exist')
        return False

    # check if dst folder exist
    if (not os.path.exists(dst)):
        print('destination path does not exist')
        return False

    # delete everything inside dst folder
    for file in listdir(dst):
        # get the path to individual file or folder
        file_path = os.path.join(dst, file)

        # delete the file or folder
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            return False

    # create the train and val folder
    train_dir = dst + '/train'
    os.mkdir(train_dir)
    val_dir = dst + '/val'
    os.mkdir(val_dir)

    # all dataset labels
    labels = [x for x in label_rule.keys()]

    # create the label folders in each train and val folders
    for label in labels:
        os.mkdir(train_dir + '/' + label)
        os.mkdir(val_dir + '/' + label)

    # get images path in src folder as specified by the efficiency
    num_of_imgs_in_src = len(listdir(src))
    src_imgs = [src + '/' + x for x in listdir(src)[0:int(eff * num_of_imgs_in_src)]]

    # split to train and val images
    split_index = int(train_test_split * len(src_imgs))
    train_imgs = src_imgs[0:split_index]

    if train_test_split != 1:
        val_imgs = src_imgs[split_index:]
    else:
        val_imgs = []

    # read the label csv
    label_dataset = pd.read_csv(labels_csv)

    # train routing
    print('--- Train routing start ---')
    i = 0
    for train_img in train_imgs:
        print('routing train image {0} ({0}/{1})'.format(i+1, len(train_imgs)))

        # get the img name path
        img_name = os.path.basename(os.path.splitext(train_img)[0])
        img_extension = os.path.splitext(train_img)[1]

        # find the label for the current image
        current_img_label = label_dataset[label_dataset.image == img_name].level.values[0]

        # route the img to the corresponding folder
        for key in label_rule.keys():
            if current_img_label in label_rule[key]:
                # route the image
                shutil.copyfile(train_img, dst + '/train/' + key + '/' + img_name + img_extension)

        i += 1
    print('--- Train routing done ---')
    print()

    # val routing
    print('--- Val routing start ---')
    i = 0
    for val_img in val_imgs:
        print('routing val {0} ({0}/{1})'.format(i+1, len(val_imgs)))

        # get the img name path
        img_name = os.path.basename(os.path.splitext(val_img)[0])
        img_extension = os.path.splitext(val_img)[1]

        # find the label for the current image
        current_img_label = label_dataset[label_dataset.image == img_name].level.values[0]

        # route the img to the corresponding folder
        for key in label_rule.keys():
            if current_img_label in label_rule[key]:
                # route the image
                shutil.copyfile(val_img, dst + '/val/' + key + '/' + img_name + img_extension)

        i += 1
    print('--- Val routing done ---')

    # success
    return True


def create_dataloader(data_dir, batch_size, n_gpu=1, shuffle=True):
    # if use cpu then set n_gpu = 1
    if n_gpu == 0:
        n_gpu = 1

    # create ToTensor tranform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # create the dataloader
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform) for x in ['train', 'val']}
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size = batch_size, shuffle = shuffle, num_workers = 4 * n_gpu) for x in ['train', 'val']}

    return dataloaders_dict
