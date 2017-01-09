#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import numpy as np
from PIL import Image
import six.moves.cPickle as pickle
from chainer.links import VGG16Layers


# param
input_flag = 'dataset'
data_dir = "../input_data/lena.jpg" if input_flag == 'img'\
        else '../input_data/beer_imgset_pc2.pkl'
# 特徴抽出に使うことができる層はchainer.links.model.vision.vggを見れ.
# output of the fc7 layer of VGG with input image. the output size is 4096.
e_layer = "fc7" 

# params
caffemodel = "../models/VGG_ILSVRC_16_layers.caffemodel"
caffe_url = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel"
chainermodel = "../models/VGG_ILSVRC_16_layers.npz"

# load image data
if input_flag == 'img':
    input_data = Image.open(data_dir)
    print("input_data.shape", np.asarray(input_data).shape)
else:
    # load sake dataset
    print(data_dir)
    dataset = pickle.load(open(data_dir))
    input_data = dataset['data']
    print("input_data.shape", input_data.shape)

# load model. chainer.links.*Layersのコンストラクタの引数はchainerモデルのパスとする．chainerモデルがない場合はクラスメソッドを使って事前に作成.
if os.path.exists(caffemodel) == False:
    os.system("wget " + caffe_url)
if os.path.exists(chainermodel) == False:
    VGG16Layers.convert_caffemodel_to_npz(caffemodel, chainermodel)

model = VGG16Layers(chainermodel)
feature = np.asarray([model.extract([ipd], layers=[e_layer])[e_layer].data for ipd in input_data])

pickle.dump(feature, open('../dump/{}_fv.pkl'.format(data_dir), 'wb'), -1)
print("feature vector shape:{}".format(feature.shape))
