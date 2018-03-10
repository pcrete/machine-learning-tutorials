

#coding=utf-8
import numpy as np
import sys,os
import caffe
import glob
import json

LOGS_PATH = '../webapp/data/logs'
SCORES_PATH = '../webapp/data/results/scores.json'
IMG_DIR = '../webapp/data/photos/'
IMG_PATH = glob.glob(os.path.join(IMG_DIR,'*.jpg'))

with open(SCORES_PATH, 'r') as fp:
    entry = json.load(fp)
# ===================================

net_file='deploy.prototxt'
caffe_model='ILGnet-AVA2.caffemodel'
mean_file='mean/AVA2_mean.npy'

#if you have no GPUs,set mode cpu
caffe.set_mode_cpu()
net = caffe.Net(net_file, caffe_model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

# ===================================
print '\nNow It\'s my turn!!, \"ILGNet said\"'

img = caffe.io.load_image(IMG_PATH[0])
net.blobs['data'].data[...] = transformer.preprocess('data',img)
out = net.forward()

entry['ILGnet'] = {
    'score': round(out["loss1/loss"][0][1], 4)
}
with open(SCORES_PATH, 'w') as outfile:
     json.dump(entry, outfile, sort_keys = True, indent = 4)

# logs file
with open(os.path.join(LOGS_PATH, 'ILGnet'), 'w') as logs_file:
    logs_file.write('')

print 'Done!!'
print '=================================='
