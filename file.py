# # 打开文件，如果不存在则创建, 存在则从文件开头开始写
# file=open('res.txt', 'w')
# # 打印字符串str
# file.write('test\n1234\n')
# # 打印变量
# a=0.72
# file.write(str(a) + '\n')
# # 打印列表
# b=[1,2,3,4,5]
# file.write(' '.join(str(i) for i in b))
# # 关闭文件
# file.close()


# 打印字典
import numpy as np
args={'x1':'y1','x2':'y2','x3':'y3'}
print(np.array(list(map(lambda x: [{x: args[x]}], args))))
# [[{'x1': 'y1'}]
#  [{'x2': 'y2'}]
#  [{'x3': 'y3'}]]
