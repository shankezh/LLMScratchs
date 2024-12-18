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
获得主从机ip
```shell
# 可以直接在主机和从机上运行
ip a
# 结果中找到inet 172.17.0.5/16 brd 172.17.255.255 scope global eth0 例如当前从机机器的地址：172.17.0.5

# 也可以安装nmap工具，扫描全局网络，看看有哪些地址
apt-get install nmap -y

nmap -sP 172.17.0.1/16
# 结合ip a 命令，扫描网络
```

```shell
# 安装ping工具
apt install inetutils-ping
# 验证
ping 172.17.0.5
```
主机从机均设置访问密码
```shell
passwd root
```

```shell
# 生成秘钥
ssh-keygen -t rsa -b 4096

# 拷贝公钥到子节点
ssh-copy-id root@172.17.0.5
```



