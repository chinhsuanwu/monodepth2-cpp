# MonoDepth2-cpp

## Overview
This is a revised version of [MonoDepth2](https://github.com/nianticlabs/monodepth2) for C++ inference usage.

<p align="center">
  <img src="monodepth2/assets/teaser.gif" width="600" />
</p>

## Dependencies
Python
- torch >= 1.4
- torchvision >= 0.5.0
- numpy

C++
- cmake >= 3.0
- libtorch
- opencv >= 3.0

## Run example
You can download the pre-trained models [here](https://drive.google.com/drive/folders/1WDOIVET_O6kHMPH8SugtIADEcqT9_TBf?usp=sharing), with the file structure shown below:
```
├── pth
│   ├── depth.jit.pth
│   ├── depth.pth
│   ├── encoder.jit.pth
│   ├── encoder.pth
│   └── ...
```
Afterwards,
```
mkdir build && cd build
cmake ..
make
./demo --pth={ABSOLUTE_PATH_TO_PTH_FOLDER}
```

## Training
Please refer to the official [README](monodepth2/README.md#training).
