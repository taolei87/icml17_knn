# Weisfeiler-Lehman Kernel Network
This directory contains implementations of 
* [Neural Fingerprint](https://hips.seas.harvard.edu/files/duvenaud-graphs-nips-2015.pdf)
* [Embedded Loopy BP](https://arxiv.org/abs/1603.05629)
* Our method: [Weisfeiler-Lehman Kernel Network](https://arxiv.org/abs/1705.09037)

These models are tested on Harvard Clean Energy Project, a molecule property prediction task.

## Data
Data can be downloaded from `https://drive.google.com/drive/folders/0B0GLTTNiVPEkdmlac2tDSzBFVzg?usp=sharing`

We thank [Hanjun Dai](http://www.cc.gatech.edu/~hdai8/) and Prof. [Le Song](http://www.cc.gatech.edu/~lsong/) for sharing this dataset.

## Usage
All codes are tested on tensorflow 0.12.0, with CUDA 8.0
Suppose you downloaded our code at $BASEDIR/graph_knn, please run the following command to add all modules in the pythonpath.
```
export PYTHONPATH=$BASEDIR/graph_knn
```

Training: 
```
python nntrain --train $TRAIN_FILE --valid $VALIDATION_FILE --save_dir $MODEL_DIR
```
Testing: 
```
python nntrain --test $TEST_FILE --model $MODEL_DIR
```
