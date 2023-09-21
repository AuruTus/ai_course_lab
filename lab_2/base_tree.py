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
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=42)
# 创建基础决策树分类器
base_tree = DecisionTreeClassifier(random_state=42)
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
plot_tree(base_tree, filled=True,
feature_names=data.feature_names, 
class_names=data.target_names)
# plt.show()
plt.savefig("./lab_2/output/base_tree.png")
