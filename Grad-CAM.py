import cv2
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from dataset_adni import get_adni

# 设置 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（SimHei）显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
(xtrain,ytrain),(xtest,ytest)=get_adni()

# 加载测试数据
Xtest = xtest  # 假设 xtest 是测试集数据
ytest = ytest  # 假设 ytest 是测试集标签

# 转换数据为 PyTorch Tensor
X_tensor_test = torch.tensor(Xtest, dtype=torch.float32)
y_tensor_test = torch.tensor(ytest, dtype=torch.long)

# 创建测试集 DataLoader
dataset_test = TensorDataset(X_tensor_test, y_tensor_test)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

# 定义模型（与训练时相同）
class Res18(nn.Module):
    def __init__(self, num_classes=2):
        super(Res18, self).__init__()
        self.backbone = resnet18(weights=None)  # 不加载预训练权重
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


from torch.nn import functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # 注册 forward 和 backward hook
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output  # 存储激活图

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # 存储梯度

    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)  # 前向传播
        if class_idx is None:
            class_idx = output.argmax().item()

        self.model.zero_grad()  # 清空梯度
        output[0, class_idx].backward(retain_graph=True)  # 计算目标类别梯度

        gradients = self.gradients.cpu().data.numpy()  # 获取梯度
        activations = self.activations.cpu().data.numpy()  # 获取激活图

        # 计算权重
        weights = np.mean(gradients, axis=(2, 3))
        cam = np.sum(weights[:, :, np.newaxis, np.newaxis] * activations, axis=1)

        # 归一化
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam[0]
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        ##强行扩大
        cam = cv2.resize(cam, (224, 224))

        return cam

# 初始化模型
model_ = Res18()
model_ = model_.cuda()
path='your pth'
# 加载保存的模型权重
model_.load_state_dict(torch.load(path))
model_.eval()  # 设置模型为评估模式
target_layer = model_.backbone.layer4[-1]  # 选取 ResNet18 的最后一个卷积层
# 生成 Grad-CAM 可视化
grad_cam = GradCAM(model_, target_layer)
for batch_X, batch_y in dataloader_test:
    input_tensor=batch_X[0]
    input_tensor=torch.reshape(input_tensor,shape=(1,3,224,224))
    input_tensor=input_tensor.cuda()
    break

cam = grad_cam.generate_cam(input_tensor)

def show_cam_on_image(img_path, cam):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img.resize((224, 224))) / 255.0

    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # 伪彩色处理
    cam = cam.astype(np.float32) / 255
    superimposed = cam * 0.2 + img  # 叠加热力图
    superimposed = (superimposed * 255).astype(np.uint8)

    fig, ax = plt.subplots()
    im = ax.imshow(superimposed)

    # 添加颜色条（图例）
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax)
    cbar.set_label("激活强度", fontsize=12)
    cbar.set_ticks([0, 0.5, 1])  # 设定颜色条的刻度
    cbar.set_ticklabels(["低", "中", "高"])  # 设定刻度标签

    plt.axis('off')
    plt.show()

image_path = r"D:\python_new\Robust-PU-master\Robust-PU-master\figures\archive (4)\Alzheimer_MRI_4_classes_dataset\MildDemented\1 (2).jpg"
show_cam_on_image(image_path, cam)
a=1
assert a!=1
# 测试模型
with torch.no_grad():
    acc_test = 0
    len_ = len(dataloader_test)
    all_preds = []
    all_labels = []
    all_probs = []

    for batch_X, batch_y in dataloader_test:

        batch_X = batch_X.cuda()
        batch_y = batch_y.cuda()
        outputs = model_(batch_X)
        y_pred = torch.argmax(outputs, dim=1)
        y_prob = torch.softmax(outputs, dim=1)[:, 1]  # 假设二分类问题，获取正类的概率

        # 计算准确率
        accuracy = (y_pred == batch_y).sum().item() / len(batch_y)
        acc_test += accuracy

        # 收集预测结果和真实标签
        all_preds.extend(y_pred.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        all_probs.extend(y_prob.cpu().numpy())

    accuracy_all = acc_test / len_
    print(f'Model Accuracy: {accuracy_all:.4f}')

    # 计算精确率、召回率、F1-score
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # 计算ROC曲线面积（AUC）和平均精确率（AP）
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    print(f'AUC: {auc:.4f}, AP: {ap:.4f}', "acc", acc)