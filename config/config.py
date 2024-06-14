# config/config.py
class Config:
    def __init__(self):
        self.image_size = (64, 64)
        self.batch_size = 64
        self.latent_dim = 100
        self.num_epochs = 20
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.num_classes = 5
        self.max_images_per_class = 1000  # 训练分类器每个类别最多使用的照片个数
        self.train_dataset = 'datasets/flowers'