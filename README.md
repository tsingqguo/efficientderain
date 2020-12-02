# EfficientDerain
we propose EfficientDerain for high-efficiency single-image deraining

<img align="center" src="./results/structure.png" swidth="750">

## Requirements

- python 3.6
- pytorch 1.6.0
- opencv-python 4.4.0.44
- scokit-image 0.17.2

## Datasets
- Rain100L http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html
- Rain100H http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html
- Rain1400 https://xueyangfu.github.io/projects/cvpr2017.html
- SPA https://stevewongv.github.io/derain-project.html

## Pretrained models
Here is the url of pretrained models for SPA-data: 
https://drive.google.com/file/d/1dpchUh3rP2gCHeUDm5W8d48xkZ2P4-FC/view?usp=sharing

## Train

- The code shown corresponds to version **v3**, for **v4** change the value of argument "**rainaug**" in file "**./train.sh**" to the "**true**" (You need to unzip the "Streaks_Garg06.zip" in the "./rainmix")
- Change the value of argument "**baseroot**" in file "**./train.sh**" to **the path of training data**
- Edit the function "**get_files**" in file "**./utils**" according to the format of the training data
- Execute

```
sh train.sh
```

## Test

- The code shown corresponds to version **v3**
- Change the value of argument "**load_name**" in file "**./test.sh**" to **the path of pretained model**
- Change the value of argument "**baseroot**" in file "**./test.sh**" to **the path of testing data**
- Edit the function "**get_files**" in file "**./utils**" according to the format of the testing data
- Execute

```
sh test.sh
```

## Results

<img align="center" src="./results/psnr_ssim-time.png" swidth="750">

## Bibtex

```
@inproceedings{guo2020efficientderain,
      title={EfficientDeRain: Learning Pixel-wise Dilation Filtering for High-Efficiency Single-Image Deraining}, 
      author={Qing Guo and Jingyang Sun and Felix Juefei-Xu and Lei Ma and Xiaofei Xie and Wei Feng and Yang Liu},
      year={2021},
      booktitle={accepted to AAAI}
}
```

