# A Dual-Branch Self-Boosting Framework for Self-Supervised 3D Hand Pose Estimation (TIP2022) 

This is the official PyTorch implementation code. For technical details, please refer to:

**A Dual-Branch Self-Boosting Framework for Self-Supervised 3D Hand Pose Estimation** <br />
Pengfei Ren, Haifeng Sun, Jiachang Hao, Qi Qi, Jingyu Wang, Jianxin Liao <br />
[[Paper]](https://ieeexplore.ieee.org/document/9841448)

<img src="pic/S1-introduction.jpg" width = 900 align=middle>


## 3D Hand Pose Estimation and 3D Hand Mesh Reconstruction
Compared with semi-automatic annotation methods in ICVL and MSRA datasets, our self-supervised method can generate more accurate and robust 3D hand pose and hand mesh.

### ICVL Dataset
![demo1](pic/ICVL.gif)

### MSRA Dataset
![demo2](pic/MSRA.gif)

## Skeleton-based Action Recognition
Using the 3D skeleton generated by DSF can greatly improve the accuracy of the skeleton-based action recognition.

 |  Method      |   Modality   | SHREC 14| SHREC 28 | DHG 14 | DHG 28 |
 | ---          | :---:        | :---: | :---: |:---: | :---:|
 | PointLSTM    | Point clouds | 95.9  | 94.7  | -    | -    |
 | Res-TCN      | Skeleton     | 91.1  | 87.3  | 86.9 | 83.6 |
 | ST-GCN       | Skeleton     | 92.7  | 87.7  | 91.2 | 87.1 |
 | STA-Res-TCN  | Skeleton     | 93.6  | 90.7  | 89.2 | 85.0 |
 | ST-TS-HGR-NET| Skeleton     | 94.3  | 89.4  | 87.3 | 83.4 |
 | HPEV         | Skeleton     | 94.9  | 92.3  | 92.5 | 88.9 |
 | DG-STA       | Skeleton     | 94.4  | 90.7  | 91.9 | 88.0 |
 | DG-STA (AWR) | Skeleton     | 96.3<sub>**↑1.9** | 93.3<sub>**↑2.6**  | 94.5<sub>**↑2.6** | 92.1 <sub>**↑4.1** |
 | DG-STA (DSF) | Skeleton     | 96.8<sub>**↑2.4** | 95.0<sub>**↑4.3**  | 96.3<sub>**↑4.4** | 95.9<sub>**↑7.9** |

## Installation
### Prerequisites

- Python >= 3.8
- PyTorch >= 1.10
- pytorch3d == 0.4.0
- CUDA (tested with cuda11.3)
- Other dependencies described in requirements.txt

### MANO

- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Download Models and copy the `models/MANO_RIGHT.pkl` into the `MANO` folder
- Your folder structure should look like this:
```
DSF/
  MANO/
    MANO_RIGHT.pkl
```
### NYU Dataset
- Download and decompress [NYU](https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm) and modify the `root_dir` in `config.py` according to your setting.
- Download the center files [[Google Drive](https://drive.google.com/drive/folders/1POQ5g3LnzAtXCvtzVF_WJoZuxLoseKuX?usp=sharing)] and put them into the `train` and `test` directories of NYU respectively.
- Download the MANO parameter file of the NYU dataset we generated [MANO Files](https://github.com/PengfeiRen96/DSF/releases/tag/Dataset)
- Your folder structure should look like this:
```
.../
  nyu/
    train/
      center_train_0_refined.txt
      center_train_1_refined.txt
      center_train_2_refined.txt
      ...
    test/
      center_test_0_refined.txt
      center_test_1_refined.txt
      center_test_2_refined.txt
      ...
    posePara_lm_collosion/
      nyu-train-0-pose.txt
      ...
```

### Pretrained Model
- Download our pre-trained model with self-supervised training [[Google Drive](https://drive.google.com/drive/folders/1XCU3ZifvaF47Fih9y-i47kTshwvcNzii?usp=sharing)]
- Download our pre-trained model with only synthetic data [[Google Drive](https://drive.google.com/drive/folders/1VQDbboU8dVSMi2ZA26mkkDJ3jOPxDTWy?usp=sharing)]
- Download the Consis-CycleGAN model [[Google Drive](https://drive.google.com/drive/folders/1tyiLc8isxyfg7vi8cS9F4gmCzrSmceBc?usp=sharing)]

## Running DSF
### Evaluation
Set `load_model` as the path to the pretrained model and change the `phase` to "test" in config.py, run
```bash
python train_render.py
```

### Self-supervised Training
To perform self-supervised training, set `finetune_dir` as the path to the pretrained model with only synthetic data and `tansferNet_pth` as the path to the Consis-CycleGAN model in config.py.
Then, change the `phase` to "train", run
```bash
python train_render.py
```

### Pre-training with Synthetic Data
To perform pre-training, set `train_stage` to "pretrain" in config.py, run
```bash
python train_render.py
```
### Citation

If you find our work useful in your research, please citing:

```
@ARTICLE{9841448,
  author={Ren, Pengfei and Sun, Haifeng and Hao, Jiachang and Qi, Qi and Wang, Jingyu and Liao, Jianxin},
  journal={IEEE Transactions on Image Processing}, 
  title={A Dual-Branch Self-Boosting Framework for Self-Supervised 3D Hand Pose Estimation}, 
  year={2022},
  volume={31},
  number={},
  pages={5052-5066},
  doi={10.1109/TIP.2022.3192708}}
```

