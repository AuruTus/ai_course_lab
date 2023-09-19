# 步骤1：载入已有数据集
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib
import matplotlib.pyplot as plt

boston = load_boston()
X = boston.data  # 特征矩阵
y = boston.target  # 目标值
# 步骤2：数据划分

test_size = []
mse = []
for i in range(3, 8):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=i/10., random_state=42)
    # 步骤3：数据预处理 - 使用最大最小归一化
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 步骤4：载入模型
    model = LinearRegression()
    # 步骤6：训练
    model.fit(X_train_scaled, y_train)
    # 步骤7：测试
    y_pred = model.predict(X_test_scaled)
    # 步骤5：选择指标 - 采用MSE
    _mse = mean_squared_error(y_test, y_pred)
    # 打印MSE

    print("Mean Squared Error (MSE):", _mse)
    test_size.append(test_size)
    mse.append(mse)

# matplotlib.use('Agg')
print("start to draw")
plt.step(test_size, mse)
plt.axis([0.2, 1., 0, 50])
plt.show()
# plt.savefig("matplotlib.png")
