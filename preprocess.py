import argparse
from utility.preprocessing import preprocess_images

parser = argparse.ArgumentParser(description='Dr research preprocessing script')
parser.add_argument('--src', '-s', help='path to images that want to be preprocessed')
parser.add_argument('--dst', '-d', help='location to put the preprocess images')
parser.add_argument('--output', '-o', type=int, default=512, help='output size of the preprocess images')
args = parser.parse_args()

if __name__ == '__main__':
    if args.src == None:
        print('Please specify the path to input images using -s flag')
    elif args.dst == None:
        print('Please specify the path to put the preprocessed images using -d flag')
    else:
        if preprocess_images(args.src, args.dst, args.output):
            print('preprocessing successed')
        else:
            print('preprocessing failed')
