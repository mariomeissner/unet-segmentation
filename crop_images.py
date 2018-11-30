import os
from imageio import imread, imwrite
from skimage import transform

USE_DRIVE = True
if (USE_DRIVE):
    from google.colab import drive
    folder = '/content/gdrive/My Drive/Projects/datasets/steven2358-larynx_data/'
    drive.mount('/content/gdrive')
else:
    folder = 'your/local/folder/here'

# How much to crop from each side
top, bottom, left, right = 15, 9, 51, 52


for filename in os.listdir(folder + 'images/images/'):
    image = imread(folder + 'images/images/' + filename, pilmode='RGB')
    image = image[top:-bottom, left:-right]
    image = transform.resize(image, (160, 240))
    imwrite(folder + 'images_cropped/images/' + filename, image)

for filename in os.listdir(folder + 'labels/labels/'):
    label = imread(folder + 'labels/labels/' + filename, pilmode='RGB')
    label = label[top:-bottom, left:-right]
    label = transform.resize(label, (160, 240))
    # Change black to blue to make category distance uniform
    for row in label:
        for pixel in row:
            if pixel[0] == 0 and pixel[1] == 0:
                pixel[2] = 1
    imwrite(folder + 'labels_cropped/labels/' + filename, label)
