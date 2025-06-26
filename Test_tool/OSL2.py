import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# 定义最小二乘法拟合接口,最小二乘法（ Ordinary Least Square，OLS）
def least_squares_fit(df, formula):
    """
    使用最小二乘法拟合R-T关系模型

    参数:
    df -- 包含数据的DataFrame，必须有['DEPTH','R1','R2','TEMP1','TEMP2']列
    formula -- 定义拟合公式的残差函数，形式为 func(params, df)

    返回:
    results -- 包含拟合参数的字典
    """
    # 初始参数猜测（假设在合理范围内）
    initial_guess = [10, 10]

    try:
        # 尝试使用LM算法（无边界约束）
        result = least_squares(
            fun=formula,
            x0=initial_guess,
            args=(df,),
            method='lm',  # Levenberg-Marquardt算法
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10
        )
    except ValueError as e:
        # 如果LM算法报错（可能由于边界约束），则使用支持边界的方法
        if "Method 'lm' doesn't support bounds" in str(e):
            print("切换到支持边界的方法：'trf' (Trust Region Reflective)")
            result = least_squares(
                fun=formula,
                x0=initial_guess,
                args=(df,),
                bounds=([-50, -50], [100, 100]),  # 设置合理的参数范围
                method='trf',  # Trust Region Reflective算法
                ftol=1e-10,
                xtol=1e-10,
                x_scale='jac',
                loss='linear'
            )
        else:
            # 其他错误直接抛出
            raise

    if not result.success:
        raise RuntimeError(f"拟合失败: {result.message}")

    # 提取拟合结果
    X1, X2 = result.x

    # 计算R方值评估拟合效果
    residuals = formula(result.x, df)
    sse = np.sum(residuals ** 2)
    tss = np.sum((df['TEMP1'] - np.mean(df['TEMP1'])) ** 2)
    r_squared = 1 - sse / tss

    # 计算每个点的拟合值
    R_ratio = df['R1'] / df['R2']
    fitted_TEMP1 = X1 + R_ratio * (df['TEMP2'] - X2)

    return {
        'X1': X1,
        'X2': X2,
        'r_squared': r_squared,
        'fitted_values': fitted_TEMP1,
        'residuals': residuals,
        'message': result.message
    }


# 根据物理公式定义的残差函数
def residual_formula(params, df):
    """
    定义物理模型的残差函数
    R1/R2 = (TEMP1-X1)/(TEMP2-X2)
    => 残差 = TEMP1 - [X1 + (R1/R2)(TEMP2 - X2)]

    参数:
    params -- [X1, X2] 待拟合参数
    df -- 包含观测数据的DataFrame

    返回:
    residuals -- 残差数组
    """
    X1, X2 = params
    R_ratio = df['R1'] / df['R2']
    predicted_TEMP1 = X1 + R_ratio * (df['TEMP2'] - X2)
    return df['TEMP1'] - predicted_TEMP1


# 生成模拟数据函数
def generate_simulated_data(X1_true=24.5, X2_true=15.2, num_points=100):
    """
    生成符合物理模型的模拟数据
    物理模型: R1/R2 = (TEMP1 - X1)/(TEMP2 - X2)

    参数:
    X1_true -- 真实参数X1的值
    X2_true -- 真实参数X2的值
    num_points -- 生成的数据点数

    返回:
    df -- 包含模拟数据的DataFrame
    """
    np.random.seed(42)  # 确保结果可复现

    # 生成深度数据（仅作为索引）
    depth = np.linspace(1000, 2000, num_points)

    # 生成TEMP2作为基础温度（10°C到50°C）
    TEMP2 = np.random.uniform(10, 50, num_points)

    # 根据物理模型生成TEMP1
    TEMP1 = np.random.uniform(20, 60, num_points)

    # 计算理论电阻率比例
    ratio = (TEMP1 - X1_true) / (TEMP2 - X2_true)

    # 添加噪声并生成R1和R2
    noise = np.random.normal(1, 0.05, num_points)  # 5%的噪声
    R1 = ratio * noise
    R2 = np.ones(num_points) * noise  # R2包含相同噪声

    # 创建DataFrame
    df = pd.DataFrame({
        'DEPTH': depth,
        'R1': R1,
        'R2': R2,
        'TEMP1': TEMP1,
        'TEMP2': TEMP2
    })

    return df


# 主程序
if __name__ == "__main__":
    # ================== 1. 生成模拟数据 ==================
    true_X1, true_X2 = 24.5, 15.2
    df = generate_simulated_data(true_X1, true_X2, 200)
    print(f"生成模拟数据 (true X1={true_X1}, X2={true_X2})")
    print("数据预览:")
    print(df.head())

    # ================== 2. 进行最小二乘拟合 ==================
    try:
        results = least_squares_fit(df, residual_formula)

        # ================== 3. 打印拟合结果 ==================
        print("\n拟合结果:")
        print(f"拟合参数 X1 = {results['X1']:.4f} (真值: {true_X1})")
        print(f"拟合参数 X2 = {results['X2']:.4f} (真值: {true_X2})")
        print(f"R² = {results['r_squared']:.6f}")
        print(f"拟合状态: {results['message']}")

        # ================== 4. 计算并展示拟合误差 ==================
        X1_error = abs(results['X1'] - true_X1)
        X2_error = abs(results['X2'] - true_X2)
        print(f"\n拟合误差:")
        print(f"X1 绝对误差: {X1_error:.6f}")
        print(f"X2 绝对误差: {X2_error:.6f}")
        print(f"相对误差: X1 {abs(X1_error / true_X1) * 100:.2f}%, X2 {abs(X2_error / true_X2) * 100:.2f}%")

        # ================== 5. 可视化部分结果 ==================
        # 选择前20个点展示
        sample_df = df.head(20).copy()
        sample_df['Fitted_TEMP1'] = results['fitted_values'].iloc[:20]

        print("\n实际值 vs 拟合值 (前20个点):")
        print(sample_df[['TEMP1', 'Fitted_TEMP1']])

        # 创建误差统计
        errors = abs(sample_df['TEMP1'] - sample_df['Fitted_TEMP1'])
        print(f"\n平均绝对误差 (MAE): {errors.mean():.5f} °C")
        print(f"最大绝对误差: {errors.max():.5f} °C")
        print(f"最小绝对误差: {errors.min():.5f} °C")

    except RuntimeError as e:
        print(f"拟合过程中发生错误: {e}")