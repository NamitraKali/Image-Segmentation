from PIL import Image
import os

def crop(path, im, size=(256, 256), out=('val_x/', 'val_y/')):
    image = Image.open(path+im)
    imgwidth, imgheight = image.size
    width, height = size

    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = image.crop(box)

            if j == 0:
                a.save(out[0]+im)
            else:
                a.save(out[1]+im)

if __name__ == '__main__':
    path = 'cityscapes_data_raw/val/'
    for im in os.listdir(path):
        if ".jpg" in im:
            crop(path, im)