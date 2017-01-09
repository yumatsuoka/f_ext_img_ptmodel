# feature extract with pre-train model
## やりたいこと

caffe modelを読み込んで画像から特徴抽出を行う．

カフェモデルのダウンロード
- VGG16 ->  http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
- ResNet -> Heさんのgistからリンクをたどっていく．

## 参考URL
- メインで使う特徴抽出の関数について http://docs.chainer.org/en/v1.19.0/reference/links.html#pre-trained-models
- カフェモデルを読み込む関数について http://docs.chainer.org/en/stable/_modules/chainer/links/caffe/caffe_function.html
