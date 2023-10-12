import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import time
import datetime

def _plot_cuda(name, lr=0.1):
    plt.figure()
    plt.title(f"accuracy of epoches for different lr {lr}")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    # 设置随机种子
    torch.manual_seed(77)

    # 数据增强，用于训练集
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转化为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理
    ])

    # 只进行张量转化和归一化，用于测试集
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # 转化为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理
    ])

    # CIFAR-10数据集
    trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=128, shuffle=True, num_workers=2)

    testSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=100, shuffle=True, num_workers=2)


    start = time.time()
    # 创建ResNet-18模型
    net = resnet18()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 使用随机梯度下降（SGD）优化器，设置学习率、动量、权重衰减
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 训练模型，迭代10个周期
    for epoch in range(10):
        # 模型设置为训练模式
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 打印每个周期的平均损失
        print(f"Epoch: {epoch + 1}, Loss: {running_loss / len(trainLoader)}")

        # 模型切换为评估模式
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testLoader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        # 打印模型在测试集上的准确率
        accuracy = correct / total
        print(f"Accuracy on test set: {100 * accuracy}%")
        xy = (epoch+1, accuracy)
        plt.scatter(xy[0], xy[1], c=[5], cmap="terrain")
        plt.annotate('(%d, %.5f)'%xy, xy=xy)
    plt.legend()
    plt.savefig(name, dpi=500)
    plt.close('all')
    end = time.time()
    print(f"elapsed seconds: {end-start}, time {datetime.timedelta(seconds=end-start)}")

def plot_cuda():
    for i, lr in enumerate([0.001, 0.01, 0.1]):
        _plot_cuda(f"./output/accuracy_{i}.png", lr)

def main():
    plot_cuda()

if __name__ == "__main__":
    main()
