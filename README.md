# Hierarchical_FL
Implementation of HierFAVG algorithm in [Client-Edge-Cloud Hierarchical Federated Learning](https://arxiv.org/abs/1905.06641) with Pytorch.

For running HierFAVG with mnist and lenet:
```python
python3 hierfavg.py --dataset mnist --model lenet --num_clients 50 --num_edges 5 --frac 1 --num_local_update 60 --num_edge_aggregation 1 --num_communication 100 --batch_size 20 --iid 0 --edgeiid 1 --show_dis 1 --lr 0.01 --lr_decay 0.995 --lr_decay_epoch 1 --momentum 0 --weight_decay 0
```





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



# 代码疑问：

1. 梯度，用的什么类型，具体的值？



## 记录

1. ```python
   python3 hierfavg.py --dataset mnist --model lenet --num_clients 50 --num_edges 5 --frac 1 --num_local_update 60 --num_edge_aggregation 1 --num_communication 100 --batch_size 20 --iid 0 --edgeiid 1 --show_dis 1 --lr 0.01 --lr_decay 0.995 --lr_decay_epoch 1 --momentum 0 --weight_decay 0
   
   全局准确率 92.23%
   ```

2. 