# Exposure Completing for Temporally Consistent Neural High Dynamic Range Video Rendering

[Paper](https://arxiv.org/pdf/2407.13309) <br>

Jiahao Cui, Wei Jiang, Zhan Peng, Zhiyu Pan, Zhiguo Cao <br>

ACMMM 2024 <br>

In this paper, we propose a novel paradigm to render HDR frames via completing the absent exposure information, hence the exposure information is complete and consistent. Our approach involves interpolating neighbor LDR frames in the time dimension to reconstruct LDR
frames for the absent exposures. Combining the interpolated and given LDR frames, the complete set of exposure information is
available at each time stamp. This benefits the fusing process for HDR results, reducing noise and ghosting artifacts therefore improving temporal consistency. Extensive experimental evaluations
on standard benchmarks demonstrate that our method achieves state-of-the-art performance, highlighting the importance of absent exposure completing in HDR video rendering.

## Installation

### Set up the python environment

```
conda create -n NECHDR python=3.9
conda activate NECHDR
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Set up datasets

#### 1. Set up training datasets
The Vimeo-90K dataset is used as training dataset, which can be downloaded at [DeepHDRVideo-Dataset](https://github.com/guanyingc/DeepHDRVideo-Dataset).. The training dataset can be organized as follows:
```
├── NECHDR_huawei/data
            ├── vimeo_septuplet
                ├── sequences
```

#### 2. Set up testing datasets
Following [HDRFlow](https://github.com/OpenImagingLab/HDRFlow), We evaluate our method on HDR_Synthetic_Test_Dataset (Cinematic Video dataset), DeepHDRVideo, and TOG13_Dynamic_Dataset (HDRVideo dataset). These datasets can be downloaded at [DeepHDRVideo-Dataset](https://github.com/guanyingc/DeepHDRVideo-Dataset). The HDR_Synthetic_Test_Dataset contains two synthetic videos (POKER FULLSHOT and CAROUSEL FIREWORKS), DeepHDRVideo consists of both real-world dynamic scenes and static scenes that have been augmented with random global motion. The TOG13_Dynamic_Dataset does not have ground truth, so we use it for qualitative evaluation. The test datasets are organized as follows:

```
├── NECHDR_huawei/data
            ├── HDR_Synthetic_Test_Dataset
            ├── dynamic_RGB_data_2exp_release
            ├── static_RGB_data_2exp_rand_motion_release
            ├── dynamic_RGB_data_3exp_release
            ├── static_RGB_data_3exp_rand_motion_release
            ├── TOG13_Dynamic_Dataset
```

## Evaluation and Training on 2-exposure Setting

### Evaluation

```
python test_2E.py --dataset DeepHDRVideo --dataset_dir data/dynamic_RGB_data_2exp_release
python test_2E.py --dataset DeepHDRVideo --dataset_dir data/static_RGB_data_2exp_rand_motion_release
python test_2E.py --dataset CinematicVideo --dataset_dir data/HDR_Synthetic_Test_Dataset
python test_tog13_2E.py
```

### Training

```
python train_2E.py
```

## Evaluation and Training on 3-exposure Setting

```
python test_3E.py --dataset DeepHDRVideo --dataset_dir data/dynamic_RGB_data_3exp_release
python test_3E.py --dataset DeepHDRVideo --dataset_dir data/static_RGB_data_3exp_rand_motion_release
python test_3E.py --dataset CinematicVideo --dataset_dir data/HDR_Synthetic_Test_Dataset
python test_tog13_3E.py
```

### Training

```
python train_3E.py
```
## Acknowledgement

This project is based on [HDRFlow](https://github.com/OpenImagingLab/HDRFlow) and [DeepHDRVideo](https://github.com/guanyingc/DeepHDRVideo), we thank all authors for their excellent work.
