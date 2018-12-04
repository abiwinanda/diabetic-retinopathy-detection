import argparse
import json
from utility.preprocessing import preprocess_images

parser = argparse.ArgumentParser(description='preprocessing script')
parser.add_argument('--src', '-s', help='path to images that want to be preprocessed')
parser.add_argument('--partial', '-p', help='path to json file that contain list of images to be preprocessed')
parser.add_argument('--dst', '-d', help='location to put the preprocess images')
parser.add_argument('--red', '-r', action='store_true', help='if set then red channel will be clahe (default false)')
parser.add_argument('--green', '-g', action='store_true', help='if set then green channel will be clahe (default false)')
parser.add_argument('--blue', '-b', action='store_true', help='if set then blue channel will be clahe (default false)')
parser.add_argument('--centre_crop', '-c', action='store_true', help='if set then preprocessing will perform centre crop (default false)')
parser.add_argument('--format', '-f', type=int, default=0, help='format of output image (0: tiff, else: jpeg)')
parser.add_argument('--output', '-o', type=int, default=512, help='output size of the preprocess images')
args = parser.parse_args()

if __name__ == '__main__':
    if args.src == None:
        print('Please specify the path to input images using -s flag')
    elif args.dst == None:
        print('Please specify the path to put the preprocessed images using -d flag')
    else:
        # parse the partial list from the partial json file
        if (args.partial == None):
            partial_list = None
        else:
            selected_preprocess_images = open(args.partial)
            selected_preprocess_images = selected_preprocess_images.read()
            selected_preprocess_images = json.loads(selected_preprocess_images)
            partial_list = selected_preprocess_images['partial']

        # generate the clahe_channels list
        clahe_channels = []
        if args.red:
            clahe_channels.append('R')
        if args.green:
            clahe_channels.append('G')
        if args.blue:
            clahe_channels.append('B')

        print(clahe_channels)

        if preprocess_images(args.src, partial_list, args.dst, clahe_channels, args.centre_crop, args.format, args.output):
            print('preprocessing successed')
        else:
            print('preprocessing failed')
