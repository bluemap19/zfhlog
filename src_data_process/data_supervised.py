import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from src_data_process.data_balanace import smart_balance_dataset
# 设置支持中文的字体，使用黑体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


def supervised_classification(X: pd.DataFrame, y: pd.Series, Norm=False, Type_str={'岩相1':0, '岩相2':1, '岩相3':2, '岩相4':3}, y_type_number=5):
    # data_balanced = smart_balance_dataset(pd.concat([X, y], axis=1), target_col=y.name, method='smote', Type_dict=Type_str)
    # X = data_balanced.iloc[:, :-1]
    # y = data_balanced.iloc[:, -1]
    y = y.astype(int)

    # 数据预处理
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 定义分类器集合
    classifiers = {
        "MLP": make_pipeline(
            StandardScaler() if not Norm else FunctionTransformer(validate=False),  # 避免重复标准化
            MLPClassifier(
                hidden_layer_sizes=(64, 64, 32),  # 增大网络容量
                alpha=1e-4,  # 添加L2正则化
                learning_rate='adaptive',  # 自适应学习率
                max_iter=500,  # 增加迭代次数
                early_stopping=True,
                random_state=42
            )
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=y_type_number)),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        ),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Random Forest": RandomForestClassifier(
            n_estimators=15,
            class_weight='balanced',
            random_state=42
        ),
        "GBM": GradientBoostingClassifier(
            n_estimators=15,
            subsample=0.8,
            random_state=42
        )
    }

    # 模型训练与评估
    results = []
    for name, clf in classifiers.items():
        # 统一训练流程
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # 评估指标计算
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # 存储混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        # 计算每个类别的正确预测数（对角线元素）
        correct_predictions = np.diag(cm)
        # 计算每个类别的总样本数（行和）
        total_samples_per_class = cm.sum(axis=1)
        # 计算每个类别的正确率
        per_class_accuracy = correct_predictions / total_samples_per_class
        # 处理可能的除零错误（当某类别无真实样本时）
        per_class_accuracy = np.nan_to_num(per_class_accuracy, nan=0.0)

        result_base = {
            "Model": name,
            "Accuracy": acc,
            "Precision": report['macro avg']['precision'],
            "Recall": report['macro avg']['recall'],
            "F1": report['macro avg']['f1-score'],
        }
        type_str = list(Type_str.keys())
        for i in range(per_class_accuracy.shape[0]):
            # result_base['ACC_' + type_str[i]] = per_class_accuracy[i]
            result_base[type_str[i]] = per_class_accuracy[i]
        results.append(result_base)

    df_results = pd.DataFrame(results).set_index('Model')
    return df_results, classifiers


def model_predict(classifiers: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    模型批量预测接口
    参数：
    classifiers : dict
        训练好的模型字典，格式为 {"模型名称": 模型对象}
    X : pd.DataFrame
        待预测数据，需确保特征与训练时一致
    返回：
    pd.DataFrame
        预测结果矩阵，每列对应不同模型的预测结果
    """
    # 初始化结果容器
    predictions = pd.DataFrame(index=X.index)

    # 遍历模型进行预测
    for model_name, clf in classifiers.items():
        try:
            # 执行预测（Pipeline自动处理预处理）
            y_pred = clf.predict(X)

            # 存储预测结果
            predictions[model_name] = y_pred.astype(int)

        except Exception as e:
            print(f"模型 {model_name} 预测失败: {str(e)}")
            predictions[model_name] = np.nan  # 标记异常预测

    # 数据验证
    if predictions.isnull().sum().sum() > 0:
        warnings.warn("部分模型预测存在缺失值，请检查输入数据与模型兼容性")

    return predictions