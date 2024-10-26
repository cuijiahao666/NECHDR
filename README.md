# Exposure Completing for Temporally Consistent Neural High Dynamic Range Video Rendering
作者列表：崔佳豪，江炜，彭展，潘治宇，曹治国


## 安装

### Python环境配置

```
conda create -n NECHDR python=3.9
conda activate NECHDR
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# pytorch版本根据服务器算力对应选择，这里也可以采用pip下载安装
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### 配置数据集

#### 1. 配置训练数据集
我们使用视频领域的学术数据集 Vimeo-90K 作为我们的训练数据集。Vimeo-90K数据集下载地址为：[DeepHDRVideo-Dataset](https://github.com/guanyingc/DeepHDRVideo-Dataset). 训练集可以如下图所示放置：
```
├── NECHDR_huawei/data
            ├── vimeo_septuplet
                ├── sequences
```

#### 2. 配置测试数据集
本技术方案可以在 HDR_Synthetic_Test_Dataset (Cinematic Video dataset), DeepHDRVideo和TOG13_Dynamic_Dataset三个数据集上进行测试. 这些数据集都可以在链接 [DeepHDRVideo-Dataset](https://github.com/guanyingc/DeepHDRVideo-Dataset) 中下载。 HDR_Synthetic_Test_Dataset数据集包含了两个合成视频场景 (POKER FULLSHOT and CAROUSEL FIREWORKS), DeepHDRVideo数据集包含了真实动态场景和带有全局随机运动的静态场景。TOG13_Dynamic_Dataset数据集由于不包含输入对应的真实HDR标签，因此仅被用于进行定性测试。测试集可以按照如下结构放置在文件夹中：

```
├── NECHDR_huawei/data
            ├── HDR_Synthetic_Test_Dataset
            ├── dynamic_RGB_data_2exp_release
            ├── static_RGB_data_2exp_rand_motion_release
            ├── dynamic_RGB_data_3exp_release
            ├── static_RGB_data_3exp_rand_motion_release
            ├── TOG13_Dynamic_Dataset
```

注：HDR_Synthetic_Test_Dataset作为常用测试数据集，在本次交付的工程中，已经放在指定文件夹下，因此可以直接用来实现代码的测试验证。

## 测试和训练的指令

### 指定显卡

训练和测试之前都需要，修改以下代码中指定的显卡号

```
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

### 测试
```
python test_2E.py --dataset DeepHDRVideo --dataset_dir data/dynamic_RGB_data_2exp_release
python test_2E.py --dataset DeepHDRVideo --dataset_dir data/static_RGB_data_2exp_rand_motion_release
python test_2E.py --dataset CinematicVideo --dataset_dir data/HDR_Synthetic_Test_Dataset
python test_tog13_2E.py
```

注：若是测试时希望保存预测图像结果，则需要在上述指令之后添加一下指令：

```
--save_results --save_dir ./output_results/方法名
```

### 训练

```
python train_2E.py
```

