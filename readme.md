#  项目架构
```
project/
├── config/
├── datasets/
├── model/
│   ├── classifier/
│   └── gan/
├── saved_model/
├── tools/
├── utils/
├── main.py
├── gan_main.py
├── readme.md
└── requirements.txt
```
## 目录结构
- `config/`: 存放配置文件。
- `datasets/`: 存放数据集。
- `model/`: 存放模型代码，分为`classifier`和`gan`两个子目录，分别存放分类器和GAN模型。
- `saved_models/`:存放训练好的数据
- `tools/`: 工具类脚本。
- `utils/`: 实用函数和工具类代码。
- `main.py`: 主要的分类器训练和测试代码。
- `gan_main.py`: 主要的GAN训练和生成代码。
- `requirements.txt`: 项目所需的Python包。

## 使用方法
1. 安装依赖：`pip install -r requirements.txt`
2. 训练GAN：`python gan_main.py`
3. 生成图像：`python utils/generate_images.py`
4. 训练分类器：`python main.py`
5. 分类图像：`python model/classifier/classify.py <image_path> <model_path>`
