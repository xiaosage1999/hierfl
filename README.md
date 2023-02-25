# Hierarchical_FL
Implementation of HierFAVG algorithm in [Client-Edge-Cloud Hierarchical Federated Learning](https://arxiv.org/abs/1905.06641) with Pytorch.

For running HierFAVG with mnist and lenet:
```python
python3 hierfavg.py --dataset mnist --model lenet --num_clients 50 --num_edges 5 --frac 1 --num_local_update 60 --num_edge_aggregation 1 --num_communication 100 --batch_size 20 --iid 0 --edgeiid 1 --show_dis 1 --lr 0.01 --lr_decay 0.995 --lr_decay_epoch 1 --momentum 0 --weight_decay 0
```





# autodl环境问题解决

```python
# 依赖包
conda install scikit-learn
conda install tensorboardX
conda install tqdm
```



该项目在autodl上跑报错：ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /root/miniconda3/lib/python3.8/site-packages/google/protobuf/pyext/_message.cpython-38-x86_64-linux-gnu.so)

[解决方案](https://huaweicloud.csdn.net/63803d86dacf622b8df86b14.html)

```shell
find /home/xxx/tools/miniconda3 -name libstdc++.so.6

strings path/libstdc++.so.6 | grep GLIBCXX

# cd到报错位置，将libstdc++.so.6删除

# 拷贝过来
```





项目地址：

- https://github.com/xiaosage1999/hierfl.git
- git@github.com:xiaosage1999/hierfl.git

# 代码疑问：



1. print哪些数据？每轮中心聚合后的模型准确率

2. 梯度，用的什么类型，具体的值？print(dict)

   ```python
   import numpy as np
   args={'x1':'y1','x2':'y2','x3':'y3'}
   print(np.array(list(map(lambda x: [{x: args[x]}], args))))
   # [[{'x1': 'y1'}]
   #  [{'x2': 'y2'}]
   #  [{'x3': 'y3'}]]
   ```

   

3. 加高斯噪声后的模型准确率？

4. 加随机响应后的模型准确率？

5. 基于cifar10、mnist数据集的更多模型和更多数据集？



# 记录

1. ```python
   python3 hierfavg.py --dataset mnist --model lenet --num_clients 50 --num_edges 5 --frac 1 --num_local_update 60 --num_edge_aggregation 1 --num_communication 100 --batch_size 20 --iid 0 --edgeiid 1 --show_dis 1 --lr 0.01 --lr_decay 0.995 --lr_decay_epoch 1 --momentum 0 --weight_decay 0
   
   全局准确率 92.23%
   ```

2. cifar10 + cnn_complex

   - k0: 20
   - k1: 2
   - k2: 3

   ```python
   python3 hierfavg.py --dataset cifar10 --model cnn_complex --num_clients 50 --num_edges 5 --frac 1 --num_local_update 3 --num_edge_aggregation 2 --num_communication 20 --batch_size 20 --iid 0 --edgeiid 1 --show_dis 0 --lr 0.01 --lr_decay 0.995 --lr_decay_epoch 1 --momentum 0 --weight_decay 0
   ```

3. mnist + logistic，

   - k0: 20
   - k1: 2
   - k2: 3

   ```python
   python3 hierfavg.py --dataset mnist --model logistic --num_clients 50 --num_edges 5 --frac 1 --num_local_update 3 --num_edge_aggregation 2 --num_communication 20 --batch_size 20 --iid 0 --edgeiid 1 --show_dis 0 --lr 0.01 --lr_decay 0.995 --lr_decay_epoch 1 --momentum 0 --weight_decay 0
   ```

   0.4721 0.6057 0.6954 0.7396 0.7676 0.7858 0.8004 0.8105 0.8192 0.8247 0.8292 0.8329 0.8387 0.8417 0.8463 0.8486 0.8503 0.8534 0.856 0.8579

4. mnist + lenet，效果差

   - k0: 20
   - k1: 2
   - k2: 3

   ```python
   python3 hierfavg.py --dataset mnist --model lenet --num_clients 50 --num_edges 5 --frac 1 --num_local_update 3 --num_edge_aggregation 2 --num_communication 20 --batch_size 20 --iid 0 --edgeiid 1 --show_dis 0 --lr 0.01 --lr_decay 0.995 --lr_decay_epoch 1 --momentum 0 --weight_decay 0
   ```

5. 









```
```











