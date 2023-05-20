# Pytorch based Mitotic Cell Classification

Specially developed for Mitotic Cell Classification

## TODO: 
 - ?

### Train
``
python3 train.py --config configs/base.yaml
``

### Test

performs tests for the given config .yaml

``
python3 test.py --config configs/base.yaml --task test
``

`--task` can be `test`, `val` or `train`

### Demo

run a demo for the given config .yaml and test image folder

``
python3 demo.py --config configs/base.yaml --images-root tests/TestDataset/a-class
``

`--images-root` is test image folder