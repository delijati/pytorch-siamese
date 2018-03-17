# pytorch-siamese

This is a port of [chainer-siamese](https://github.com/mitmul/chainer-siamese)

## Install

This installation requires `cuda` to be installed. 

```
$ virtualenv /usr/bin/python3.5 env
$ env/bin/pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 
$ env/bin/pip install torchvision
```

## Run

```
$ env/bin/python train_mnist.py --epoch 10
```

This dumps for every epoch ther current `state` and creates a `result.png`.

### Run specific model

```
$ env/bin/python train_mnist.py -m model-epoch-7.pth
```

## Result

![](https://raw.githubusercontent.com/delijati/pytorch-siamese/master/result.png)

