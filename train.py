import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from utility.model import initialize_model, train_model, get_trainable_params
from utility.dataset import seperate_dataset_to_labels_folder, create_dataloader

parser = argparse.ArgumentParser(description='Dr research preprocessing script')
parser.add_argument('--model', '-m', help='name of the model you want to train')
parser.add_argument('--classes', '-n', type=int, help='number of output class')
parser.add_argument('--feature', '-f', type=bool, default=True, help='if set to true then train only the new initialized layers')
parser.add_argument('--pretrained', '-p', type=bool, default=True, help='if set to true then use torch pretrained model')
parser.add_argument('--dataset', '-d', help='path to dataset folder that contain the train and val folder')
parser.add_argument('--batch', '-b', type=int, default=32, help='batch size')
parser.add_argument('--gpu', '-g', type=int, default=1, help='number of gpu to train')
parser.add_argument('--epoch', '-e', type=int, default=10, help='number of epoch')
parser.add_argument('--output', '-o', default='eye-model.pth', help='file name of the model with .pth extension')
args = parser.parse_args()

if __name__ == '__main__':
    if args.model == None:
        print('Please specify the model name with -m flag')
    elif args.classes == None:
        print('Please specify the number of output classes with -n flag')
    elif args.dataset == None:
        print('Please specify the path to dataset folder with -d flag')
    else:
        # initialize the model
        net, _ = initialize_model(args.model, args.classes, args.feature, args.pretrained)

        # send net to gpu if exist
        if args.gpu != 0:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            net = nn.DataParallel(net)
        else:
            device = torch.device('cpu')
        net = net.to(device)

        # set the loss function
        loss_func = nn.CrossEntropyLoss()

        # set the learning algorithm
        params_to_update = get_trainable_params(net) # get the trainable parameter
        lr_algo = optim.Adam(params_to_update, lr=0.001)

        # create the data loader
        dataloaders = create_dataloader(args.dataset, args.batch, args.gpu, True)

        # train the model
        net, val_acc_hist = train_model(net, dataloaders, loss_func, lr_algo, device, args.epoch)

        torch.save(net.state_dict(), args.output)
