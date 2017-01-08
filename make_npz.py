#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from chainer.links import VGG16Layers, ResNet50Layers

ResNet50Layers.convert_caffemodel_to_npz("./ResNet-50-model.caffemodel", "./ResNet-50-model.npz")
# VGG16Layers.convert_caffemodel_to_npz("./VGG_ILSVRC_16_layers_wget.caffemodel", "./VGG_ILSVRC_16_layers.npz")
print("make npz from caffe model")
