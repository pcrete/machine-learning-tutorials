from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import sys
import os
import glob
import cv2
import caffe
import pandas as pd
import json
import numpy as np
from caffe.proto import caffe_pb2
from time import sleep

LOGS_PATH = '../../webapp/data/logs/'
SCORES_PATH = '../../webapp/data/results/scores.json'
IMG_DIR = '../../webapp/data/photos/'

AVA_ROOT = 'AVA/'
IMAGE_MEAN= AVA_ROOT + 'mean_AADB_regression_warp256.binaryproto'
DEPLOY = AVA_ROOT + 'initModel.prototxt'
MODEL_FILE = AVA_ROOT + 'initModel.caffemodel'
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
input_layer = 'imgLow'

# Image processing helper function
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

caffe.set_mode_gpu()
# caffe.set_mode_cpu()

# Reading mean image, caffe model and its weights
mean_blob = caffe_pb2.BlobProto()
with open(IMAGE_MEAN) as f:
    mean_blob.ParseFromString(f.read())

mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape((
                 mean_blob.channels,
                 mean_blob.height,
                 mean_blob.width
                )
               )

net = caffe.Net(DEPLOY, MODEL_FILE, caffe.TEST)

# Define image transformers
print "Shape mean_array : ", mean_array.shape
print "Shape net : ", net.blobs[input_layer].data.shape
net.blobs[input_layer].reshape(1,       # batch size
                               3,       # channel
                               IMAGE_WIDTH, IMAGE_HEIGHT)  # image size
transformer = caffe.io.Transformer({input_layer: net.blobs[input_layer].data.shape})
transformer.set_mean(input_layer, mean_array)
transformer.set_transpose(input_layer, (2,0,1))

# =================================================================================
print 'Let\'s do Aesthetic Assessment\n'
print 'Deep AADB Stand by!!'
while True:
    if len(os.listdir(IMG_DIR)) == 1:
        break
    sleep(3)
print 'Yeah!! I got it.'

IMG_PATH = glob.glob(os.path.join(IMG_DIR,'*.jpg'))
# =================================================================================
print 'Processing..'
img = cv2.imread(IMG_PATH[0], cv2.IMREAD_COLOR)
img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

net.blobs[input_layer].data[...] = transformer.preprocess(input_layer, img)
out = net.forward()

entry = {
    'AADB_mode':{}
}
for feature in out:
    entry['AADB_mode'][feature] = round(out[feature][0][0],4)

with open(SCORES_PATH, 'w') as outfile:
     json.dump(entry, outfile, sort_keys = True, indent = 4)

# logs file
with open(os.path.join(LOGS_PATH, 'AADB'), 'w') as logs_file:
    logs_file.write('')

print 'Done!!'
print '=================================='
