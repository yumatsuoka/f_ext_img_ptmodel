#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import argparse
import numpy as np
from PIL import Image
import six.moves.cPickle as pickle
from chainer.links import VGG16Layers


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='lena.jpg', type=str)
parser.add_argument('--e_layer', default='fc7', type=str)
args = parser.parse_args()

# param
# 'beer_imgset_pc2.pkl'
input_dir = "../input_data/" + args.input_dir 
# 特徴抽出に使うことができる層はchainer.links.model.vision.vggを見れ.
# output of the fc7 layer of VGG with input image. the output size is 4096.
e_layer = args.e_layer 

caffemodel = "../models/VGG_ILSVRC_16_layers.caffemodel"
caffe_url = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel"
chainermodel = "../models/VGG_ILSVRC_16_layers.npz"

# load image or pickled dict data
if os.path.splitext(input_dir)[1] == '.jpg':
    input_data = Image.open(input_dir)
    print("input_data.shape", np.asarray(input_data).shape)

elif os.path.splitext(input_dir)[1] == '.pkl':
    dataset = pickle.load(open(input_dir))
    input_data = dataset['data']
    print("input_data.shape", input_data.shape)

# load model. 
# コンストラクタの引数はchainerモデルのパス．chainerモデルがない場合はクラスメソッドを使って事前に作成.
if os.path.exists(caffemodel) == False:
    os.system("wget " + caffe_url)
if os.path.exists(chainermodel) == False:
    VGG16Layers.convert_caffemodel_to_npz(caffemodel, chainermodel)

model = VGG16Layers(chainermodel)
feature = model.extract([input_data], layers=[e_layer])[e_layer].data\
        if os.path.splitext(input_dir)[1] == '.jpg'\
        else np.asarray([model.extract([ipd], layers=[e_layer])[e_layer].data\
        for ipd in input_data])

pickle.dump(feature, open('../dump/{}_fv.pkl'.format(args.input_dir), 'wb'), protocol=2)
print("feature vector shape:{}".format(feature.shape))
