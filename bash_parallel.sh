#!/bin/bash

python3.11 launch.py -c resnet.10fold.combined.bs_256.lr_1e-3.json &
python3.11 launch.py -c resnet.10fold.combined.bs_256.lr_1e-4.json &
python3.11 launch.py -c resnet.10fold.combined.bs_400.lr_1e-4.json &
python3.11 launch.py -c resnet.10fold.combined.bs_512.lr_1e-4.json &
python3.11 launch.py -c mlp.10fold.combined.bs_128.lr_1e-4.json &
python3.11 launch.py -c mlp.10fold.combined.bs_256.lr_1e-3.json &
python3.11 launch.py -c mlp.10fold.combined.bs_256.lr_1e-4.json &
python3.11 launch.py -c mlp.10fold.combined.bs_256.lr_1e-5.json &
python3.11 launch.py -c mlp.10fold.combined.bs_512.lr_1e-4.json &
python3.11 launch.py -c cnn2d.10fold.combined.bs_128.lr_1e-4.json &
python3.11 launch.py -c cnn2d.10fold.combined.bs_256.lr_1e-4.json &
python3.11 launch.py -c cnn2d.10fold.combined.bs_256.lr_10-5.json &
python3.11 launch.py -c cnn2d.10fold.combined.bs_400.lr_1e-4.json &
python3.11 launch.py -c cnn2d.10fold.combined.bs_512.lr_1e-4.json &