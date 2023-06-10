# F-SOUL表情识别云平台后端

基于Python的表情识别云平台后端。模型复现了[Nan, Y., Ju, J., Hua, Q., Zhang, H., & Wang, B. A-MobileNet: An approach of facial expression recognition[J]. Alexandria Engineering Journal, 2022, 61(6): 4435-4444.](https://doi.org/10.1016/j.aej.2021.09.066)中所述的A-MobileNet模型，中心损失函数实现参考[pytorch-center-loss](https://github.com/KaiyangZhou/pytorch-center-loss)，数据集采用[FERPlus](https://github.com/microsoft/FERPlus)开源数据集。



## 开发

首次运行需要额外进行：

1. 依赖安装：

   ```bash
   $ pip install -r requirements.txt
   ```
   如果torch系包下载过慢或运行时出现问题，请前往[PyTorch](https://pytorch.org/get-started/locally/)官网获取下载链接。

2. 在Mysql服务器上运行该目录下的`create_table.sql`文件以初始化数据库。数据库中自带用户名为“EileenQueen”，密码为“SweetCounter”的超管用户。

3. 配置`./server.py`文件中第32行的数据库链接

   ```python
   connection = pymysql.connect(host='localhost',# 数据库所在地址，默认3306端口
                                user='root',# 数据库用户名
                                password='',# 数据库密码
                                database='project',# 后续无需修改
                                charset='utf8mb4',
                                cursorclass=pymysql.cursors.DictCursor)
   ```

   

作为服务器运行时：

```bash
python ./server.py
```

使用摄像头进行测试时：

```bash
python ./real_time_video.py
```

训练模型时：

```bash
python ./main.py
```

## 构建

```bash
pyinstaller -D server.py
```