# Tensorflow tutorial

This is a code adapted from this [awesome example](https://github.com/Slaski/Tensorflow-Getting-Started/blob/master/house_price_prediction.py). I used this code during my learning process of Tensorflow. Also, I used this code as an explanation on this [tutorial post on Medium]().

## Setting up the environment
1. Install Python 3, pip and Virtualenv
```shell
sudo apt-get install python3-pip python3-dev python-virtualenv
```

2. Create and activate the *VirtualEnv*:
```shell
virtualenv --system-site-packages -p python3 .env3
source .env3/bin/activate
```

3. Install Tensorflow
```shell
pip install tensorflow
```

4. Verify if it is all working good
```
python -c "import tensorflow as tf; print(tf.__version__)"
```

## How to run

There are two examples here:
- Simple tutorial: The same code used in the tutorial. It just predicts the house price.
```shell
python simple_tutorial.py
```

- Complete code: Besides predicting house price, this code also plot a graph that help visualizing the Gradient Decent evolution. Also, this code is compatible with Tensorboard.

```shell
python house_pred.py
```
## How to run Tensorboard

After run the `house_pred.py`. Just run the command above:
```shell
tensorboard --logdir=/tmp/tf/train
```
