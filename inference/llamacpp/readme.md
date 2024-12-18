# Introduction
机器只有建议通过这个这个库进行推理，性能不错，而且长期维护.

## Installation
如果没有cmake,安装下cmake工具,[CMake Link](https://cmake.org/download/).
安装可以查看是否安装完成
```shell
cmake --version
# 如果显示右边为成功===> cmake version 3.31.2
```
下载llama.cpp进入目录
```shell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```
编译
```shell
cmake -B build
cmake --build build --config Release
```



# Refs
[1] [llama.cpp Github]()<br>
[2] [llama.cpp build docs](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)<br>

