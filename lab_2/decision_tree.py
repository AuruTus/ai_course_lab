from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
data = load_iris()
X = data.data
y = data.target
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def _plot_tree(max_depth, min_samples_split, min_samples_leaf, name="base_tree"):
    # 创建基础决策树分类器
    base_tree = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    # 拟合模型
    base_tree.fit(X_train, y_train)
    # 预测
    y_pred = base_tree.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the Base Decision Tree: {accuracy}")
    from sklearn.tree import plot_tree
    # 可视化基础决策树
    plt.figure(figsize=(12, 6))
    plt.suptitle("Accuracy of %.2f for %s"%(accuracy, name))
    plot_tree(base_tree, filled=True, feature_names=data.feature_names, class_names=data.target_names)
    # plt.show()
    
    plt.savefig(f"./lab_2/output/{name}.png")
    plt.close("all")

def plot_tree():
    _plot_tree(max_depth=5, min_samples_split=5, min_samples_leaf=5, name="tree_arg_1")
    _plot_tree(max_depth=2, min_samples_split=2, min_samples_leaf=2, name="tree_arg_2")

def plot_accuracy():
    plt.figure()
    plt.title("accuracy for different max_depth")
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")

    for max_depth in range(2, 6):
        # 创建基础决策树分类器
        base_tree = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
        # 拟合模型
        base_tree.fit(X_train, y_train)
        # 预测
        y_pred = base_tree.predict(X_test)
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of the Base Decision Tree: {accuracy}")
        xy = (max_depth, accuracy)
        plt.scatter(xy[0], xy[1], c=[5], cmap="terrain")
        plt.annotate('(%.2f, %.6f)'%xy, xy=xy)
    
    plt.legend()
    # plt.show()
    plt.savefig("./lab_2/output/tree_accuracy.png")
    plt.close("all")



def main():
    import sys
    if len(sys.argv) <= 1:
        plot_tree()
        plot_accuracy()
        return

    args = sys.argv[1:]
    match args[0]:
        case "tree":
            plot_tree()
        case "accuracy":
            plot_accuracy()

if __name__ == "__main__":
    main()