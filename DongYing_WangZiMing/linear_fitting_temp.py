import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import matplotlib as mpl
import seaborn as sns
import os


# 设置中文支持
def setup_chinese_support():
    """配置Matplotlib支持中文显示"""
    if os.name == 'nt':  # Windows
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    else:  # Mac/Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC']

    plt.rcParams['axes.unicode_minus'] = False


setup_chinese_support()


# 线性拟合，非常抗噪的线性拟合，要丢弃很多数据进行线性拟合
def robust_linear_fit(df, x_col='x', y_col='y', degree=1, fit_intercept=False,
                      max_trials=1000, min_samples=0.5, residual_threshold=0.5,
                      plot=True):
    """
    鲁棒线性拟合函数 - 使用RANSAC算法处理离群点

    参数:
    df: 包含输入数据的DataFrame
    x_col: DataFrame中x数据的列名 (默认'x')
    y_col: DataFrame中y数据的列名 (默认'y')
    degree: 多项式拟合的阶数 (默认1，线性拟合)
    fit_intercept: 是否拟合截距项 (默认False，即拟合过原点的直线)
    max_trials: RANSAC算法的最大迭代次数 (默认1000)
    min_samples: RANSAC算法中内点的最小比例 (默认0.5，即50%)
    residual_threshold: 残差阈值，小于此值的点被视为内点 (默认0.5)
    plot: 是否绘制拟合结果图 (默认True)

    返回:
    model: 拟合的RANSAC模型对象
    slope: 拟合线的斜率（对于线性模型）或多项式系数（对于高阶模型）
    r2: 决定系数（R²），表示拟合优度
    inlier_mask: 布尔数组，标记哪些点是内点（非离群点）
    """

    # 1. 数据准备
    if isinstance(x_col, str):
        X = df[x_col].values.reshape(-1, 1)
    elif isinstance(x_col, int):
        X = df.iloc[:, x_col].values.reshape(-1, 1)
    else:
        raise TypeError('y_col 输入参数类型错误')

    if isinstance(y_col, str):
        y = df[y_col].values
    elif isinstance(y_col, int):
        y = df.iloc[:, y_col].values
    else:
        raise TypeError('y_col 输入参数类型错误')

    # 2. 创建鲁棒回归模型
    # 根据是否拟合截距和多项式阶数创建不同的基础估计器

    # 对于过原点线性回归的特殊处理
    if degree == 1 and not fit_intercept:
        # 创建多项式特征+线性回归的管道
        # 使用PolynomialFeatures生成多项式特征
        # 使用LinearRegression(fit_intercept=False)进行过原点拟合
        base_estimator = make_pipeline(
            PolynomialFeatures(degree),  # 生成多项式特征
            linear_model.LinearRegression(fit_intercept=False)  # 过原点线性回归
        )
    else:
        # 对于其他情况（带截距的线性回归或多项式回归）
        # 直接使用线性回归模型
        base_estimator = linear_model.LinearRegression(fit_intercept=fit_intercept)

    # 创建RANSAC回归模型
    # RANSAC是一种鲁棒回归算法，能有效处理离群点
    model = linear_model.RANSACRegressor(
        base_estimator=base_estimator,  # 使用上面创建的基础估计器
        min_samples=min_samples,  # 内点的最小比例
        residual_threshold=residual_threshold,  # 残差阈值（小于此值为内点）
        max_trials=max_trials  # 最大迭代次数
    )

    # 3. 模型拟合
    # 使用RANSAC算法拟合模型，自动识别内点和离群点
    model.fit(X, y)  # 拟合模型

    # 4. 内点识别
    # 从模型中获取内点掩码（True表示内点，False表示离群点）
    inlier_mask = model.inlier_mask_  # 内点布尔数组

    # 5. 计算拟合指标
    # 获取拟合线的斜率或多项式系数
    if degree == 1:
        # 对于线性模型
        if fit_intercept:
            # 带截距的线性模型：y = ax + b
            # 斜率是第二个系数（第一个系数是截距）
            slope = model.estimator_.coef_[0]  # 斜率
        else:
            # 过原点的线性模型：y = ax
            # 斜率是第一个系数
            slope = model.estimator_.steps[-1][1].coef_[1]  # 斜率
    else:
        # 对于多项式模型
        # 获取所有多项式系数
        slope = model.estimator_.steps[-1][1].coef_  # 多项式系数

    # 使用模型预测所有点
    y_pred = model.predict(X)  # 预测值

    # 计算决定系数R²（仅使用内点）
    # R²衡量模型拟合优度，值越接近1表示拟合越好
    r2 = r2_score(y[inlier_mask], y_pred[inlier_mask])  # 决定系数

    # 6. 结果可视化
    if plot:
        # 调用绘图函数展示拟合结果
        plot_fit_results(df, x_col, y_col, model, inlier_mask, slope, r2, degree, fit_intercept)

    # 返回结果
    return model, slope, r2, inlier_mask


def plot_fit_results(df, x_col, y_col, model, inlier_mask, slope, r2, degree, fit_intercept):
    """
    绘制拟合结果
    """
    X = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values

    plt.figure(figsize=(12, 8))

    # 绘制原始数据点
    plt.scatter(df[x_col][~inlier_mask], df[y_col][~inlier_mask],
                c='gray', alpha=0.3, s=20, label='离群点')
    plt.scatter(df[x_col][inlier_mask], df[y_col][inlier_mask],
                c='blue', alpha=0.6, s=30, label='内点')

    # 绘制拟合线
    X_plot = np.linspace(df[x_col].min(), df[x_col].max(), 300).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, 'r-', linewidth=2.5, label='鲁棒拟合')

    # 添加标注信息
    intercept_info = f"截距: {model.estimator_.steps[-1][1].intercept_:.3f}" if fit_intercept else ""
    title = f"鲁棒线性回归 (阶数: {degree}, 过原点: {not fit_intercept})\n"
    title += f"斜率: {slope[1] if degree > 1 else slope:.4f} {intercept_info}, R²: {r2:.4f}"

    # 设置标题和标签
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # 设置坐标轴范围
    plt.xlim(df[x_col].min() - 5, df[x_col].max() + 5)

    # 添加图例
    plt.legend(loc='best')

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def generate_test_data(slope=0.005, n_points=500, outlier_ratio=0.4, random_state=None):
    """
    生成包含离群点的测试数据

    参数:
    slope: 斜率 (默认0.005)
    n_points: 数据点数量 (默认500)
    outlier_ratio: 离群点比例 (默认0.4)
    random_state: 随机种子

    返回:
    测试数据的DataFrame
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 生成x值 (0-300)
    x = np.random.uniform(0, 300, n_points)

    # 生成符合线性趋势的数据点
    y_main = slope * x

    # 添加噪声
    noise = np.random.normal(0, 0.02, n_points)
    y_main += noise

    # 生成离群点
    n_outliers = int(n_points * outlier_ratio)
    outlier_indices = np.random.choice(np.arange(n_points), n_outliers, replace=False)

    # 创建离群点 (添加随机偏移和噪声)
    for i in outlier_indices:
        outlier_type = np.random.choice([0, 1, 2])
        if outlier_type == 0:  # 垂直偏移
            offset = np.random.uniform(-0.4, 0.4)
            y_main[i] += offset
        elif outlier_type == 1:  # 水平偏移
            offset = np.random.uniform(-100, 100)
            x[i] += offset
        else:  # 完全随机点
            x[i] = np.random.uniform(0, 300)
            y_main[i] = np.random.uniform(-0.1, 0.9)

    # 创建DataFrame
    df = pd.DataFrame({'x': x, 'y': y_main})

    return df


# 测试函数
def test_robust_fit():
    """测试鲁棒线性拟合"""
    # 生成测试数据
    df = generate_test_data(slope=0.005, n_points=300, outlier_ratio=0.45, random_state=42)

    # 应用鲁棒拟合 (过原点)
    print("=== 过原点线性模型 (y = ax) ===")
    model, slope, r2, inliers = robust_linear_fit(
        df, x_col='x', y_col='y',
        degree=1, fit_intercept=False,
        min_samples=0.4, residual_threshold=0.05
    )



if __name__ == "__main__":
    test_robust_fit()