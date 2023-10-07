import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

# 加载鸢尾花数据集
data = load_iris()
X = data.data
y = data.target
# 选择两个特征以便可视化
X = X[:, :2]
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def plot_accuracy():
    plt.title("accuracy for different penalties")
    plt.xlabel("penaly value")
    plt.ylabel("accuracy")
    # plt.axis([0.2, 0.8, 15, 30])

    # 创建SVM分类器
    for C in [0.01, 0.02, 0.03, 0.04, 0.05]:
        svm = SVC(kernel='linear', C=C, random_state=42)
        # 拟合模型
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of the Base SVM for C {C}: {accuracy}")

        xy = (C, accuracy)
        plt.scatter(xy[0], xy[1], c=[5], cmap="terrain")
        plt.annotate('(%.2f, %.2f)'%xy, xy=xy)
        # plt.show()
    plt.savefig("./lab_2/output/accuracy.png", dpi=500)
    plt.close('all')


def plot_tsne(raw_data, data):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    data_2d = tsne.fit_transform(data)
    plt.figure(figsize=(16,10))
    target_ids = range(len(raw_data.target_names))

    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, raw_data.target_names):
        plt.scatter(data_2d[y == i, 0], data_2d[y == i, 1], c=c, label=label)
    plt.legend()
    # plt.show()
    plt.savefig("./lab_2/output/tsne.png")
    plt.close('all')


def plot_decision_boundary(X_train, y_train):
    # 函数用于绘制SVM决策边界
    def _plot_decision_boundary(X, y, model, title):
        h = .02 # 步长
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
        np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', s=100)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    # 创建SVM分类器
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    # 拟合模型
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the Base SVM: {accuracy}")

    # 绘制决策边界和数据点
    _plot_decision_boundary(X_train, y_train, svm, "SVM Decision Boundary (Train)")
    plt.tight_layout()
    # plt.show()
    plt.savefig("./lab_2/output/svm.png")
    plt.close('all')

def main():
    import sys
    if len(sys.argv) <= 1:
        plot_tsne(data, X)
        plot_decision_boundary(X_train, y_train)
        plot_accuracy()
        return

    args = sys.argv[1:]
    match args[0]:
        case "tsne":
            plot_tsne(data, X)
        case "bound":
            plot_decision_boundary(X_train, y_train)
        case "accuracy":
            plot_accuracy()
            


if __name__ == "__main__":
    main()
