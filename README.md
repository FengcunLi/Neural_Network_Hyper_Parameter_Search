## 学习率、网络结构、参数初始化之间的相互影响关系
### Hyper parameter search 
#### Overview
![accuracy_test](./assets/accuracy_test.png)
![accuracy_train](./assets/accuracy_train.png)
#### 不同学习率
![不同学习率_test](./assets/不同学习率_test.png)
![不同学习率_legend](./assets/不同学习率_legend.png)
![不同学习率_train](./assets/不同学习率_train.png)
结论：
合适的学习率十分重要，影响模型性能的上限。

#### 不同网络结构不同参数初始化 
![不同网络结构不同参数初始化_test](./assets/不同网络结构不同参数初始化_test.png)
![不同网络结构不同参数初始化_legend](./assets/不同网络结构不同参数初始化_legend.png)
![不同网络结构不同参数初始化_train](./assets/不同网络结构不同参数初始化_train.png)
结论：
1. 合适的参数初始化十分重要，影响模型性能的上限。
2. 在合适的参数初始化下，网络结构也并非越复杂越好，而是要求“合理”（即卷积层和全连接层的数量的合理搭配）
3. 网络结构复杂会使学习过程变慢。

#### Top
![top_test](./assets/top_test.png)
![top_legend](./assets/top_legend.png)
![top_train](./assets/top_train.png)
结论：
1. 相同参数初始化和相同网络结构下，较大的学习率会加速学习过程。
2. 不同的参数初始化，合适的学习率是不同的。

#### 结论
学习率、网络结构、参数初始化三者之间需要合理搭配，才能使学习过程顺畅地进行，并最终提升性能。