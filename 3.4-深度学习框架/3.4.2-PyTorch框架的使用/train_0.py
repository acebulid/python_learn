import torch
import torch.nn as nn 
import torch.optim as optim # 导入优化器
from torchvision import datasets, transforms # 导入数据集和数据预处理库
from torch.utils.data import DataLoader # 数据加载库

# 设置随机种子
torch.manual_seed(21)

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量 
    transforms.Normalize((0.5), (0.5))  # 标准化图像数据 灰度图，只需要一个0.5    -1  -  1
])

# 加载FashionMNIST数据集
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)  # 下载训练集
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform) #  下载测试集

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 对训练集进行打包， 一批次64个图像塞入神经网络训练
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) # 对测试集进行打包

# 定义神经网络模型
class QYNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)   # 定义第一个全连接层 隐藏层的神经元个数为128
        self.fc2 = nn.Linear(128, 10)   # 定义第二个全连接层 输出神经元个数 10 因为我们需要做10分类

    def forward(self, x): # 前向传播
        x = torch.flatten(x, start_dim=1)  # 展平数据，方便进行全连接
        x = torch.relu(self.fc1(x))  # 非线性
        x = self.fc2(x) # 十分类  [0.1,0.2,0.5,0.2,0,0,0,0,0,0]
        return x 

# 初始化模型
model = QYNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵
optimizer = optim.SGD(model.parameters(), lr=0.01) # lr 学习率 用来调整模型收敛速度 0.1 

# 训练模型
epochs = 10
for epoch in range(epochs): # 0-9
    running_loss = 0.0 # 定义初始loss为0
    for inputs, labels in train_loader:
        optimizer.zero_grad() # 梯度清零
        outputs = model(inputs) # 将图片塞进网络训练获得 输出
        loss = criterion(outputs, labels) # 根据输出和标签做对比计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        running_loss += loss.item() # loss值累加

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# 测试模型
correct = 0 # 正确的数量
total = 0 # 样本总数
with torch.no_grad(): # 不用进行梯度计算
    for inputs, labels in test_loader:
        # print(labels.shape)
        outputs = model(inputs) #   [0.1,0.2,0.5,0.2,0,0,0,0,0,0]   2
        # print(outputs.shape)
        _, predicted = torch.max(outputs, 1) # _取到的最大值，可以不要， 我们需要的是最大值对应的索引 也就是label（predicted）
        total += labels.size(0) # 获取当前批次样本数量
        correct += (predicted == labels).sum().item() # 对预测对的值进行累加 

print(f"Accuracy on test set: {correct/total:.2%}")
