#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
from PIL import Image
import six.moves.cPickle as pickle
#import chainer
from chainer import cuda
from chainer.links import VGG16Layers, ResNet50Layers


# param
input_flag = 'img'
data_dir = "lena.jpg" if input_flag == 'img' else 'beer_imgset.pkl'
# $BFCD'Cj=P$K;H$&$3$H$,$G$-$kAX$O(Bchainer.links.model.vision.vgg$B$r8+$l(B.
# output of the fc7 layer of VGG with input image. the output size is 4096.
e_layer = "fc7" 

# load image data
if input_flag == 'img':
    input_data = Image.open(data_dir)
    print("input_data.shape", np.asarray(input_data).shape)
else:
    # load sake dataset
    dataset = pickle.load(open(data_dir))
    input_data = dataset['data']
    print("input_data.shape", input_data.shape)


# load model. chainer.links.*Layers$B$N%3%s%9%H%i%/%?$N0z?t$O(Bchainer$B%b%G%k$N%Q%9$H$9$k!%(Bchainer$B%b%G%k$,$J$$>l9g$O%/%i%9%a%=%C%I$r;H$C$F;vA0$K:n@.(B.
model = VGG16Layers("./VGG_ILSVRC_16_layers.npz")
#model = ResNet50Layers("./ResNet-50-model.npz")
feature = model.extract([input_data], layers=[e_layer])[e_layer]
#pickle.dump(feature, open('{}_fv.pkl'.format(data_dir), 'wb'), -1)

print("feature vector shape:{}".format(feature.data.shape))
