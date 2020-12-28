# EfficientDerain
we propose EfficientDerain for high-efficiency single-image deraining

<img align="center" src="./results/structure.png" swidth="750">

## Requirements

- python 3.6
- pytorch 1.6.0
- opencv-python 4.4.0.44
- scikit-image 0.17.2

## Datasets
- Rain100L-old_version https://drive.google.com/file/d/1YcL74X90M4z_9O7wr2miWgZC1GfJuNOR/view?usp=sharing
- Rain100H-old_version https://drive.google.com/file/d/1ZczoGWvXS0Liz1_B96SRTU6fmMePWodM/view?usp=sharing
- Rain1400 https://xueyangfu.github.io/projects/cvpr2017.html
- SPA https://stevewongv.github.io/derain-project.html

## Pretrained models
Here is the url of pretrained models (includes v3_rain100H, v3_rain1400, v3_SPA, v4_rain100H, v4_rain1400, v4_SPA) : 
https://drive.google.com/file/d/1OBAIG4su6vIPEimTX7PNuQTxZDjtCUD8/view?usp=sharing


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


<center><font size="2">ssim/psnr of v3 and v4 in four datasets</font></center>
|    | SPA-data       | rain1400       | rain100H       | rain100L       |
|----|----------------|----------------|----------------|----------------|
| v3 | 0.9825 / 41.09 | 0.9272 / 32.30 | 0.8970 / 30.35 | 0.9486 / 33.18 |
| v4 | 0.9810 / 40.66 | 0.9226 / 31.81 | 0.9099 / 31.12 | 0.9549 / 34.06 |

## Bibtex

```
@inproceedings{guo2020efficientderain,
      title={EfficientDeRain: Learning Pixel-wise Dilation Filtering for High-Efficiency Single-Image Deraining}, 
      author={Qing Guo and Jingyang Sun and Felix Juefei-Xu and Lei Ma and Xiaofei Xie and Wei Feng and Yang Liu},
      year={2021},
      booktitle={accepted to AAAI}
}
```

