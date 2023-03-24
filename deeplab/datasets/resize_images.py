# Resize the images

import os
from PIL import Image
f = "/home/dfdavii/Downloads/models/research/deeplab/datasets/Custom_Dataset/JPEGImages"
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((256,256))
    img.save(f_img)