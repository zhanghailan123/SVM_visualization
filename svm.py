import copy

import cv2
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd                    # for data handling
import numpy as np                     # for random selections, mainly
import matplotlib.pyplot as plt        # for plotting
plt.rcParams['figure.figsize'] = 10,7   # graph dimensions
plt.rcParams['font.size'] = 14         # graph font size
from dataset_adni import get_adni
from torchvision.models import resnet18
from torchvision import models
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, \
    accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import make_moons, fetch_openml
import matplotlib.pyplot as plt

(xtrain,ytrain),(xtest,ytest)=get_adni()
X=xtrain
Xtest=xtest
# ytrain=numpy.reshape(ytrain,newshape=(ytrain.shape[0],224*224))
y=ytrain
y = pd.Series(y, name='class')

# Unlabel a certain number of data points
hidden_size = 1565
y.loc[
    np.random.choice(
        y[y == 1].index,
        replace = False,
        size = hidden_size
    )
] = 0

# 转换数据为 PyTorch Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

X_tensor_test = torch.tensor(Xtest, dtype=torch.float32)
y_tensor_test = torch.tensor(ytest, dtype=torch.long)

# X_tensor=torch.reshape(X_tensor,shape=(X_tensor.shape[0],224*224))
# X_tensor_test=torch.reshape(X_tensor_test,shape=(X_tensor_test.shape[0],224*224))

# 创建数据集和数据加载器
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

dataset_test = TensorDataset(X_tensor_test, y_tensor_test)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

# 定义 6 层 MLP 网络
class Res18(nn.Module):
    def __init__(self, num_classes=2):
        super(Res18, self).__init__()

        # 加载预定义的ResNet18模型
        self.backbone = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


        # 替换最后的全连接层，以适应自己的任务
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        h=self.backbone(x)
        return h

# 初始化模型
model_ = Res18()
model_=model_.cuda()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model_.parameters(), lr=0.001)  # Adam 优化器

num_epochs = 20
best_acc=0
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
    # 将标签转换为 numpy 数组
        y_train = np.array(batch_X)
        y_test = np.array(batch_y)
        X_train = xtrain.reshape(xtrain.shape[0], -1)
        X_test = xtest.reshape(xtest.shape[0], -1)
        # 初始化 SVM 模型
        svm_model = SVC(kernel='linear', probability=True)  # 使用线性核，并启用概率估计
        # 训练 SVM 模型
        svm_model.fit(X_train, y_train)

        # 预测
        y_pred = svm_model.predict(X_test)
        y_prob = svm_model.predict_proba(X_test)[:, 1]  # 获取正类的概率

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        # 打印评估结果
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
        print(f'AUC: {auc:.4f}, AP: {ap:.4f}')






