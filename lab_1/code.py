from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process_data(scaler=MinMaxScaler(), _range=None, ratio=None):
    # 步骤1：载入已有数据集
    boston = load_boston()
    X = boston.data  # 特征矩阵
    y = boston.target  # 目标值
    # 步骤2：数据划分

    test_size = []
    mse = []
    # test with test size ratio from 0.3 to 0.7
    if _range is not None:
        for i in _range:
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=i, random_state=42)
            # 步骤3：数据预处理 - 使用最大最小归一化
            # scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train) if scaler is not None else X_train
            X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test

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
            test_size.append(i)
            mse.append(_mse)
    elif ratio:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=ratio, random_state=42)
        # 步骤3：数据预处理 - 使用最大最小归一化
        # scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train) if scaler is not None else X_train
        X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test

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
        test_size.append(ratio)
        mse.append(_mse)
    return (test_size, mse)


def draw_scatter(xset_yset, title, img_location, color):
    (xset, yset) = xset_yset
    # draw discrete points with matplotlib
    print("start to draw", xset, yset)
    # plt.plot(xset, yset)
    # plt.step(xset, yset)
    plt.title(title)
    plt.xlabel('test_size_ratio')
    plt.ylabel('MSE')
    plt.axis([0.2, 0.8, 15, 30])
    for xy in zip(xset, yset):
        # plt.scatter(xset, yset, c=np.full(len(xset), 5), cmap=color)
        plt.scatter(xy[0], xy[1], c=[5], cmap=color)
        # plt.annotate('(%.2f, %.2f)'%xy, xy=xy)
    # plt.show()
    plt.savefig(img_location, dpi=500)
    plt.close('all')

def draw_comparison_plot(scaler=MinMaxScaler(), img_location='./lab_1/output/comparison.png', ratio=0.5):
    # 步骤1：载入已有数据集
    boston = load_boston()
    X = boston.data  # 特征矩阵
    y = boston.target  # 目标值
    # 步骤2：数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=ratio, random_state=42)
    # 步骤3：数据预处理 - 使用最大最小归一化
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train) if scaler is not None else X_train
    X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test

    # 步骤4：载入模型
    model = LinearRegression()
    # 步骤6：训练
    model.fit(X_train_scaled, y_train)
    # 步骤7：测试
    y_pred = model.predict(X_test_scaled)

    _low, _high = 0, 200
    _range = range(_low+1, _high+1)
    plt.plot(_range, y_test[:_high], label="real", color='red')
    plt.plot(_range, y_pred[:_high], label="pred", color='green')
    plt.legend()
    plt.savefig(img_location, dpi=500)

def main():
    draw_scatter(process_data(_range=np.arange(0.3, 0.71, 0.05)), "MSE for different test sizes", "./lab_1/output/problem_1.png", "terrain") # problem 1
    draw_scatter(process_data(_range=np.arange(0.3, 0.71, 0.005)), "MSE for different test sizes", "./lab_1/output/problem_1_pace.png", "terrain") # problem 1
    # draw_scatter(process_data(None, range(3, 8)), "MSE with no scaler for different test sizes", "./lab_1/output/problem_2_none.png", "ocean") # problem 2
    draw_scatter(process_data(None, _range=np.arange(0.3, 0.71, 0.005)), "MSE with no scaler for different test sizes", "./lab_1/output/problem_2_none.png", "ocean") # problem 2
    draw_scatter(process_data(StandardScaler(), _range=np.arange(0.3, 0.71, 0.005)), "MSE with StandardScaler for different test sizes", "./lab_1/output/problem_2.png", "ocean") # problem 2
    # draw_scatter(process_data(ratio=0.5), "MSE for 0.5 test sizes", "./lab_1/output/problem_3.png", "Oranges_r") # problem 3
    draw_comparison_plot(img_location="./lab_1/output/problem_3.png") # problem 3


if __name__ == '__main__':
    print('hello')
    main()
