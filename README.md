# EfficientDerain
we propose EfficientDerain for high-efficiency single-image deraining



## Requirements

- python 3.6
- pytorch 1.6.0
- opencv-python 4.4.0.44
- scokit-image 0.17.2





## Train

- The code shown corresponds to version **v3**, For **v4** change the value of argument "**rainaug**" in file "**./train.sh**" to the "**true**"
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
