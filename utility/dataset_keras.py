import cv2
import xlsxwriter
import numpy as np
from sklearn.metrics import confusion_matrix

def prepare_data(image_path, norm):
    """
    Preparing image data.
    :param image_path: list of image to be processed

    Return:
            x: array of resized images
            y: array of labels
    """
    count = len(image_path)
    x = np.ndarray((count, 512, 512, 3), dtype=np.float32)

    # generate input image
    for i, image in enumerate(image_path):
        if i%500 == 0: print('Processed {} of {}'.format(i, count))
        #img = tiff.imread(image)
        img = cv2.imread(image)
        x[i] = img

    # normalize all images
    if (norm == 0):
        print("Normalize all images into [-1, 1] range")
        x *= (2/255)
        x -= 1
    else:
        print("Normalize all images into [-1, 1] range")
        x *= (1/255)
    print("Normalization done!")
    return x

def create_confusion_matrix(model, batch_size, X_val, Y_val, excel_file_name):
    Y_pred = model.predict(X_val, batch_size = batch_size)
    Y_pred = np.around(Y_pred).astype(int)
    result = confusion_matrix(Y_val, Y_pred)

    # create new workbook
    workbook = xlsxwriter.Workbook(excel_file_name + '.xlsx')
    worksheet = workbook.add_worksheet()

    # confusion matrix header
    worksheet.write(0, 1, '0')
    worksheet.write(0, 2, '1')
    worksheet.write(1, 0, '0')
    worksheet.write(2, 0, '1')

    # confusion matrix data
    worksheet.write(1, 1, result[0][0])
    worksheet.write(1, 2, result[0][1])
    worksheet.write(2, 1, result[1][0])
    worksheet.write(2, 2, result[1][1])

    workbook.close()
