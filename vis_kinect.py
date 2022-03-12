import cv2
import os.path
import glob
import numpy as np
from PIL import Image

depth_max = 1500
depth_min = 200

def convertPNG(image_name):
    # READ THE DEPTH
    img = Image.open('./KinectV2/'+image_name)  # open image
    depth = np.asarray(img, np.float32)
    depth[depth > depth_max] = depth_max
    depth[depth < depth_min] = depth_min
    depth = (depth - depth_min)/(depth - depth_min).max() * 255
    # apply colormap on deoth image(image must be converted to 8-bit per pixel first)
    # depth = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=1), cv2.COLORMAP_BONE)
    # convert to mat png
    cv2.imwrite('./KinectV2/h-'+image_name, depth)

if __name__ == '__main__':
    for file in os.listdir('./KinectV2/'):
        convertPNG(file)