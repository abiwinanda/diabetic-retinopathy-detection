from os import listdir
import cv2


def preprocess_images(src, dst, output_size = 800, ):
    # get all files in src
    img_paths = [src + x for x in listdir(src)]
    im_pth = "C:/Users/Wida/test4.jpeg"

    for img_path in img_paths:
        # get the image name
        img_name = None
        img_extension = None

        # read the image
        imgOri = cv2.imread(img_path)

        # extract individual channel of the image (b, g, r)
        b_channel, g_channel, r_channel = cv2.split(imgOri)

        # CLAHE-ing green channel
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        g_channel = clahe.apply(g_channel)

        # resizing the image
        old_size = g_channel.shape[:2] # old_size is in (height, width) format
        ratio = float(output_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        g_channel = cv2.resize(g_channel, (new_size[1], new_size[0])) # new_size should be in (width, height) format

        # zero padding the image
        delta_w = output_size - new_size[1]
        delta_h = output_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        g_channel = cv2.copyMakeBorder(g_channel, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)

        # save the final preprocessed image
        cv2.imwrite(dst + img_name + "-preprocessed" + img_extension, g_channel)
