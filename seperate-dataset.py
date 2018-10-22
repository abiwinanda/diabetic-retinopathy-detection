import argparse
from utility.dataset import seperate_dataset_to_labels_folder

parser = argparse.ArgumentParser(description='script to seperate dataset to their label folder')
parser.add_argument('--src', '-s', help='path to images that want to be seperated')
parser.add_argument('--dst', '-d', help='location to create the train and val folder')
parser.add_argument('--csv', '-c', help='path to the csv file that contain the labels')
parser.add_argument('--eff', '-e', type=float, default=1.0, help='if set to 1 then all images in src will be used or seperated')
parser.add_argument('--split', type=float, default=0.8, help='train test split ratio')
args = parser.parse_args()

if __name__ == '__main__':
    if args.src == None:
        print('Please specify the path to the images you want to be seperated with -s flag')
    elif args.dst == None:
        print('Please specify the path to put the train and val folder with -d flag')
    elif args.csv == None:
        print('Please specify the csv file that contain the labels with -d flag')
    else:
        label_rule = {
            'Healthy': [0],
            'DR': [1,2,3,4]
        }

        if seperate_dataset_to_labels_folder(args.src, args.dst, args.csv, label_rule, args.eff, args.split):
            print('seperating dataset successed')
        else:
            print('seperating dataset failed')
