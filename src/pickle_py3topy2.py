#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle

data_dir = '../input_data/beer_imgset.pkl'
dataset = pickle.load(open(data_dir, 'rb'))
# pickleの最後の引数が-1のときは有効なプロトコルのうち最も高いバージョンのものが使われる．
# 今回はpython2との互換性を持ちたいために2を指定する． 
pickle.dump(dataset, open('../input_data/beer_imgset_pc2.pkl', 'wb'),\
        protocol=2) 
