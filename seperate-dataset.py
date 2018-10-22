import argparse
import json
from utility.dataset import seperate_dataset_to_labels_folder

parser = argparse.ArgumentParser(description='script to seperate dataset to their label folder')
parser.add_argument('--src', '-s', help='path to images that want to be seperated')
parser.add_argument('--dst', '-d', help='location to create the train and val folder')
parser.add_argument('--csv', '-c', help='path to the csv file that contain the labels')
parser.add_argument('--rule', '-r', help='path to the rule json file')
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
        # create the label rule from the rule json file
        label_rule_file = open(args.rule)
        label_rule_string = label_rule_file.read()
        label_rule = json.loads(label_rule_string)

        if seperate_dataset_to_labels_folder(args.src, args.dst, args.csv, label_rule, args.eff, args.split):
            print('seperating dataset successed')
        else:
            print('seperating dataset failed')
