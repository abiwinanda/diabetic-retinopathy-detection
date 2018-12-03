import cv2
import argparse
import numpy as np
from keras.models import load_model

parser = argparse.ArgumentParser(description='keras inference script')
parser.add_argument('--model', '-m', help='path to the trained model')
parser.add_argument('--image', '-i', help='path to the image that wants to be inferenced')
args = parser.parse_args()

if __name__ == '__main__':
    if args.model == None:
        print("Please specify your trained model path with -m flag")
    elif args.image == None:
        print("Please specify the image path that wants to be inferenced with -m flag")
    else:
        model = load_model(args.model)
        image = np.expand_dims(cv2.imread(args.image), axis=0)
        prediction = model.predict(image)
        if prediction >= 0.5:
            print("This is a healthy eye!")
        else:
            print("Diabetic Retinopathy is detected")