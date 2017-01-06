#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from PIL import Image
import chainer
from chainer import cuda
from chainer.links import VGG16Layers, ResNet50Layers


# param
input_flag = 'img'
model_flag = 'VGG'
e_layer = "fc7"
data_dir = "lena.jpg" if input_flag == 'img' else 'all_imgs_dic.pkl'

# load image data
if input_flag == 'img':
    input_data = Image.open(data_dir)
else:
    # load sake dataset
    dataset = pickle.load(open(data_dir))
    input_data = dataset['data']

# load model 
model = VGG16Layers()
#model = ResNet50Layers() if model_flag == 'ResNet' else VGG16Layers()

#if args.gpu >= 0:
#    cuda.get_device(args.gpu).use()
#    model.to_gpu()

# $BFCD'Cj=P$K;H$&$3$H$,$G$-$kAX$O(Bchainer.links.model.vision.vgg$B$r8+$l(B.
# output of the fc7 layer of VGG with input image. the output size is 4096.
feature = model.extract([input_data], layers=[e_layer])[e_layer]

print("feature vector shape".format(feature.data.shape))

#pickle.dump(feature, open('{}_fv.pkl'.format(data_dir), 'wb'), -1)