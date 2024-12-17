# Introduction
这个文档用于展示使用DeepSpeed如何配置多机多卡进行训练，代码依旧是使用deepspeed_pretrain.py文件.

# Process
## 安装SSH
```shell
apt install openssh-server

# 安装ss命令工具
apt install iproute2
# 验证是否安装成功
ssh -V 

# 启动ssh服务
/usr/sbin/sshd -D &

# 查看启动情况
ss -tulnp | grep ssh

# 可以看到22端口已经被监听
>> (base) root@99b1129a7b41:~/sspaas-tmp/LLMScratchs# ss -tulnp | grep ssh
>> tcp   LISTEN 0      128          0.0.0.0:22         0.0.0.0:*    users:(("sshd",pid=382,fd=3))       
>> tcp   LISTEN 0      128             [::]:22            [::]:*    users:(("sshd",pid=382,fd=4))  
```

```shell
# 生成秘钥
ssh-keygen -t rsa -b 4096
```

