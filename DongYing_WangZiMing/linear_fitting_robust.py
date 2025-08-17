import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import RANSACRegressor


# 设置中文支持
def setup_chinese_support():
    """配置Matplotlib支持中文显示"""
    if os.name == 'nt':  # Windows
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    else:  # Mac/Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC']

    plt.rcParams['axes.unicode_minus'] = False


setup_chinese_support()


def robust_linear_fit(df, x_col='x', y_col='y', degree=1, fit_intercept=True,
                      max_trials=1000, min_samples=0.5, residual_threshold=0.5,
                      plot=True):
    """
    健壮线性拟合函数 - 带截距优化版

    参数:
    df: 包含输入数据的DataFrame
    x_col: x数据列名 (默认'x')
    y_col: y数据列名 (默认'y')
    degree: 多项式阶数 (默认1，线性拟合)
    fit_intercept: 是否拟合截距项 (默认True，即y=ax+b)
    max_trials: RANSAC最大迭代次数 (默认1000)
    min_samples: 内点最小比例 (默认0.5)
    residual_threshold: 残差阈值 (默认0.5)
    plot: 是否绘制结果图 (默认True)

    返回:
    model: 拟合模型
    slope: 斜率(线性)或多项式系数
    intercept: 截距(如果fit_intercept=True)
    r2: 决定系数
    inlier_mask: 内点标记
    """

    # 1. 数据准备
    if isinstance(x_col, str):
        X = df[x_col].values.reshape(-1, 1)
    elif isinstance(x_col, int):
        X = df.iloc[:, x_col].values.reshape(-1, 1)
    else:
        raise TypeError('x_col 输入参数类型错误')

    if isinstance(y_col, str):
        y = df[y_col].values
    elif isinstance(y_col, int):
        y = df.iloc[:, y_col].values
    else:
        raise TypeError('y_col 输入参数类型错误')

    # 2. 创建鲁棒回归模型
    # 根据是否拟合截距和多项式阶数创建不同的基础估计器
    if degree == 1:
        if fit_intercept:
            # 带截距的线性模型：y = ax + b
            base_estimator = linear_model.LinearRegression(fit_intercept=True)
        else:
            # 过原点的线性模型：y = ax
            base_estimator = make_pipeline(
                PolynomialFeatures(degree, include_bias=False),
                linear_model.LinearRegression(fit_intercept=False)
            )
    else:
        # 多项式模型
        base_estimator = make_pipeline(
            PolynomialFeatures(degree),
            linear_model.LinearRegression(fit_intercept=fit_intercept)
        )

    # 创建RANSAC回归模型
    model = RANSACRegressor(
        base_estimator=base_estimator,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=42  # 确保可重复结果
    )

    # 3. 模型拟合
    model.fit(X, y)

    # 4. 内点识别
    inlier_mask = model.inlier_mask_

    # 5. 计算拟合指标
    # 获取拟合线的斜率或多项式系数
    if degree == 1:
        if fit_intercept:
            # 带截距的线性模型
            slope = model.estimator_.coef_[0]
            intercept = model.estimator_.intercept_
        else:
            # 过原点的线性模型
            slope = model.estimator_.steps[-1][1].coef_[0]
            intercept = 0
    else:
        # 多项式模型
        slope = model.estimator_.steps[-1][1].coef_
        intercept = model.estimator_.steps[-1][1].intercept_ if fit_intercept else 0

    # 使用模型预测所有点
    y_pred = model.predict(X)

    # 计算决定系数R²（仅使用内点）
    r2 = r2_score(y[inlier_mask], y_pred[inlier_mask])

    # 6. 结果可视化
    if plot:
        # 调用绘图函数展示拟合结果
        plot_fit_results(df, x_col, y_col, model, inlier_mask, slope, intercept, r2, degree, fit_intercept)

    # 返回结果
    return model, slope, intercept, r2, inlier_mask


def plot_fit_results(df, x_col, y_col, model, inlier_mask, slope, intercept, r2, degree, fit_intercept):
    """绘制拟合结果图"""
    plt.figure(figsize=(10, 6))

    # 绘制原始数据点
    plt.scatter(df[x_col][~inlier_mask], df[y_col][~inlier_mask],
                c='gray', alpha=0.5, s=20, label='离群点')
    plt.scatter(df[x_col][inlier_mask], df[y_col][inlier_mask],
                c='blue', alpha=0.7, s=30, label='内点')

    # 绘制拟合线
    x_min, x_max = df[x_col].min(), df[x_col].max()
    x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_fit = model.predict(x_range)

    plt.plot(x_range, y_fit, 'r-', linewidth=2, label='拟合线')

    # 添加标题和标签
    if degree == 1:
        if fit_intercept:
            equation = f"y = {slope:.4f}x + {intercept:.4f}"
        else:
            equation = f"y = {slope:.4f}x"
    else:
        equation = f"多项式 (阶数: {degree})"

    plt.title(f"健壮线性拟合\n{equation}\nR2 = {r2:.4f}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # 添加图例
    plt.legend()

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)

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
    model, slope, intercept, r2, inliers = robust_linear_fit(
        df, x_col='x', y_col='y',
        degree=1, fit_intercept=True,
        min_samples=0.4, residual_threshold=0.05, plot=True
    )

    # 打印结果
    print(f"拟合斜率: {slope:.6f}")
    print(f"拟合截距: {intercept:.6f}")
    print(f"决定系数 R²: {r2:.4f}")
    print(f"内点数量: {np.sum(inliers)}/{len(df)}")


if __name__ == "__main__":
    test_robust_fit()