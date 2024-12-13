# Introduction
这个文档用于描述DeepSpeed多卡训练的安装和使用过程

# installation
官方库的地址为:
[DeepSpeed Github](https://github.com/microsoft/DeepSpeed) <br>
页面有安装方法,pip和编译库文件方法自行选择。<br>
推荐使用docker安装，避免环境不一致问题：
```shell
docker run -it ubuntu:22.04 /bin/bash
apt-get update
apt-get upgrade -y
# 安装miniforge
# 安装缺失库
apt-get update && apt-get install -y libaio-dev
apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
apt-get install wget -y
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

#安装完成后
source ~/.bashrc
#创建deepspeed环境
conda create --name dsenv python=3.11
# 启动环境
conda activate dsenv
# 安装deep_speed
pip install deepspeed
```
pycharm如何连接docker环境：
[Link ](https://www.cnblogs.com/lantingg/p/14927981.html)<br>



# Refs
[1] [DeepSpeed Github](https://github.com/microsoft/DeepSpeed) <br>
[2] [other tutorial](https://github.com/OvJat/DeepSpeedTutorial)<br>