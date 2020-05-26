#!/bin/bash

chmod 1777 /tmp
apt-add-repository ppa:gviz-adm/graphviz-dev -y
apt-get update
apt-get install -y unzip
apt-get autoremove graphviz
apt-get install graphviz -y

mkdir data
unzip -qq /opt/ml/input/data/training/covers64.zip -d data

pip install -r requirements.txt
python train/train_progan.py

mkdir /opt/ml/model/learning_progress
mv  -v learning_progress/* /opt/ml/model/learning_progress