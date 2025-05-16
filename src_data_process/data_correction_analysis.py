import pydot
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, auc,
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, label_binarize, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from src_random_data.create_random_data import get_random_logging_dataframe


# 根据分类的线性关系，分析 输入数据-输出数据 间的相关性
def analyse_attribute_by_random_forest_classifer(X, y, feature_x, path_saved=r'D:\Logging_Data\FILE_TEMP',
                                                 n_classes = 3, random_seed=44, plot_index=[2, 2], figsize=(16, 5),
                                                 axes_title_list = ['Cluster 0'], plot_title_charter='', tree_num=10,
                                                 figure=None, Norm=False):
    """
    :param X: 自变量，输入数据
    :param y: 因变量，输出数据
    :param feature_x: 自变量对应的标签list，其数量必须等于 变量X的列数 即，输入自变量的个数
    :param path_saved: 保存路径
    :param n_classes: 分类的类别个数，类别不一定从0开始，自动识别类别个数
    :param random_seed: 随机种子
    :param plot_index: matplotlib的绘制个数，必须是[m,n]，代表了m*n的画板，必须大于等于与n_classes
    :param figsize:图版的大小
    :return:
    """
    assert X.shape[0] == y.shape[0], "输入XY的形状不符合要求X:{}, y:{}".format(X.shape, y.shape)
    assert plot_index[0]*plot_index[1]>=n_classes, "Plot的子图个{}数小于类别个数n_classes{}".format(plot_index, n_classes)
    if len(axes_title_list) < np.unique(y.ravel()).shape[0]:
        for i in range(np.unique(y.ravel()).shape[0]):
            axes_title_list.append('Cluster '+str(i+1))
        print('Not enough plot title, now add title auto as:{}'.format(axes_title_list))

    print('start analyse by random forest, input X shape:{}, y shape:{}'.format(X.shape, y.shape))

    result_dict = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, train_size=0.6, stratify=y, random_state=random_seed)
    print('data split to shape -X_train:{}, -X_test:{}, -y_train:{}, -y_test:{}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    # Build RF regression model
    # 利用RandomForestClassifier进行模型的构建，n_estimators就是树的个数，
    # random_state是每一个树利用Bagging策略中的Bootstrap进行抽样（即有放回的袋外随机抽样）时，随机选取样本的随机数种子；
    # fit进行模型的训练，predict进行模型的预测，最后一句就是计算预测的误差。
    # # 构建预处理管道
    # random_forest_model = make_pipeline(
    #     StandardScaler(),
    #     RandomForestClassifier(
    #         n_estimators=100,
    #         class_weight='balanced',
    #         max_depth=8,
    #         random_state=random_seed
    #     )
    # )
    random_forest_model = RandomForestClassifier(
        n_estimators=tree_num,  # 增加树的数量
        class_weight='balanced',  # 处理类别不平衡
        max_depth=5,  # 控制过拟合
        random_state=random_seed
    )

    # 修改交叉验证代码
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    scoring = 'f1_weighted'
    scores = cross_val_score(
        random_forest_model,
        X_train,  # 仅在训练集上做CV
        y_train.ravel(),
        cv=cv_strategy,
        scoring=scoring
    )
    # scores = cross_val_score(random_forest_model, X, y.ravel(), cv=5)       # 5-fold 交叉验证
    result_dict['cross_val_score'] = scores
    print("Cross-validation scores: {}, Mean accuracy: {:.4f}".format(scores, scores.mean()))
    # print('Type scores :{}'.format(type(scores)))       # <class 'numpy.ndarray'>

    random_forest_model.fit(X_train, y_train.ravel())
    y_pred = random_forest_model.predict(X_test)
    # 分类任务，模型评估函数
    # classification_report - 函数提供了精度、召回率、F1 分数和支持度（每个类别的样本数）等信息。
    # 其中 前者y_test是真实标签，后者y_pred是模型预测结果
    report_classify = classification_report(y_test, y_pred)
    print('Type report_classify:{}\n {}'.format(type(report_classify), report_classify))
    result_dict['report_classify'] = report_classify

    cm = confusion_matrix(y_test, y_pred)
    result_dict['confusion_matrix'] = cm
    print('classify confusion matrix is:\n{}'.format(cm))

    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R² Score: {r2_score(y_test, y_pred)}")

    # classes = np.unique(y)  # 获取所有类别标签
    classes = [0, 1, 2, 3, 4]  # 获取所有类别标签
    print('classes:{}'.format(classes))
    # 计算每个类别的正确率（召回率）
    class_accuracy = {}
    for i, cls in enumerate(classes):
        true_positive = cm[i, i]  # 对角线元素（正确预测数）
        actual_total = cm[i, :].sum()  # 行总和（真实类别总数）
        accuracy = true_positive / actual_total if actual_total != 0 else 0.0
        class_accuracy[cls] = round(accuracy, 4)  # 保留4位小数
    result_dict['per_class_accuracy'] = class_accuracy
    # # 输出结果
    # for cls, acc in class_accuracy.items():
    #     print(f"类别 {cls} 的正确率：{acc * 100:.2f}%")


    # roc_auc_score - ROC AUC（接收者操作特征曲线下面积）是评估分类模型性能的指标，尤其适用于不平衡数据集。AUC 值越高，模型性能越好。
    y_test = label_binarize(y_test, classes=random_forest_model.classes_)
    y_pred = label_binarize(y_pred, classes=random_forest_model.classes_)
    print('random forest:{}, y_test range:[{}, {}], y range:[{}, {}], y_test data shape:{}'.format(
        random_forest_model.classes_, np.max(y_test), np.min(y_test), np.max(y), np.min(y), y_test.shape))


    # print(y_test.shape, np.max(y_test), np.min(y_test))           # (29, 3) 1 0
    auc_total = roc_auc_score(y_test, y_pred, multi_class='ovr')
    print("Total ROC AUC Score: {}".format(auc_total))
    # print('Type auc_total :{}'.format(type(auc_total)))         # <class 'numpy.float64'>


    ################ 作图,不同类别的AUC曲线图
    # y_test = label_binarize(y_test, classes=random_forest_model.classes_)
    if (n_classes <= 2):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        # 4. 绘制ROC曲线
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Characteristic {}'.format(plot_title_charter))
        plt.legend(loc='lower right')
        plt.show()
    else:
        # 绘制每个类别的ROC曲线
        fig, axes = plt.subplots(plot_index[0], plot_index[1], figsize=figsize)
        colors = ["r", "g", "b", "k", 'y', 'c', 'm', 'k', 'r']
        markers = ["o", "^", "v", "+", '1', 's', '*', 'D', 'H']
        for i in range(n_classes):
            # 计算每个类别的FPR, TPR
            fpr, tpr, thr = roc_curve(y_test[:, i].ravel(), y_pred[:, i].ravel())
            print("classes_{}, fpr:{}, tpr:{}, threshold:{}".format(i, fpr, tpr, thr))
            # 绘制ROC曲线，并计算AUC值
            if plot_index[0] == 1:
                axes[i % plot_index[1]].plot(fpr, tpr, color=colors[i], marker=markers[i], label="AUC: {:.2f}".format(auc(fpr, tpr)))
                axes[i % plot_index[1]].set_xlabel("FPR")
                axes[i % plot_index[1]].set_ylabel("TPR")
                # axes[i % plot_index[1]].set_title("Class_{}".format(random_forest_model.classes_[i]))
                axes[i % plot_index[1]].set_title(axes_title_list[i])
                axes[i % plot_index[1]].legend(loc="lower right")
            else:
                axes[i//plot_index[1], i % plot_index[1]].plot(fpr, tpr, color=colors[i], marker=markers[i], label="AUC: {:.2f}".format(auc(fpr, tpr)))
                axes[i//plot_index[1], i % plot_index[1]].set_xlabel("FPR")
                axes[i//plot_index[1], i % plot_index[1]].set_ylabel("TPR")
                # axes[i//plot_index[1], i % plot_index[1]].set_title("Class_{}".format(random_forest_model.classes_[i]))
                axes[i//plot_index[1], i % plot_index[1]].set_title(axes_title_list[i])
                axes[i//plot_index[1], i % plot_index[1]].legend(loc="lower right")
        fig.suptitle('Different Class ROC Curve {}'.format(plot_title_charter))
        fig.show()

    # ####### Draw decision tree visualizing plot
    # tree_graph_dot_path = path_saved+'\\tree_classifer.dot'
    # tree_graph_png_path = path_saved+'\\tree_classifer.png'
    # random_forest_tree = random_forest_model.estimators_[5]
    # export_graphviz(random_forest_tree, out_file=tree_graph_dot_path, feature_names=feature_x, rounded=True, precision=1)
    # (random_forest_graph,)=pydot.graph_from_dot_file(tree_graph_dot_path)
    # random_forest_graph.write_png(tree_graph_png_path)

    # 各重要性权重计算
    importances = random_forest_model.feature_importances_
    # print('Type importances :{}'.format(type(importances)))     # <class 'numpy.ndarray'>

    # 重要性权重排序，只返回排序后的index，要根据index进行相应的遍历访问
    indices = np.argsort(importances)[::-1]
    importances_N = []
    feature_x_N = []
    # 根据各重要性权重排序结果，重新进行重要性权重排序结果生成
    for f in range(X.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feature_x[indices[f]], importances[indices[f]]))
        importances_N.append(importances[indices[f]])
        feature_x_N.append(feature_x[indices[f]])

    ###################### 作图,不同输入属性的影响因子柱状图
    plt.figure(3)
    plt.clf()
    # 柱状图个数设置
    importance_plot_x_values = list(range(X.shape[1]))
    # # 柱状图高度设置
    # plt.bar(importance_plot_x_values, importances, orientation='vertical')
    # # 柱状图X坐标标签设置
    # plt.xticks(importance_plot_x_values, feature_x, rotation='horizontal')
    plt.bar(importance_plot_x_values, importances_N, orientation='vertical')
    plt.xticks(importance_plot_x_values, feature_x_N, rotation='vertical')
    plt.xlabel('Variable')
    plt.ylabel('Importance')
    plt.title('Variable Importances {}'.format(plot_title_charter))
    plt.show()

    return scores, class_accuracy, auc_total, importances


def data_correction_analyse_by_tree(data_input: pd.DataFrame,
                                    feature_input:[],               # 数据输入的columns参数
                                    feature_target:str='Type',      # 数据目标的column参数
                                    tree_num:int=10,                # 数的个数
                                    # plt:matplot.figure=None,      # 画板信息
                                    y_replace_dict:dict={},         # 类别信息对应的标签信息
                                    title_string:str='',
                                    random_seed:int=42,
                                    path_saved:str='',
                            ) -> pd.DataFrame:
    assert isinstance(data_input, pd.DataFrame)

    # 相关性分析
    X = data_input[feature_input].astype('float64')
    y = data_input[feature_target].astype(np.int32)
    y_classes = np.unique(y)
    y_len = len(y_classes)
    plot_index = {1:[1, 1], 2:[1, 2], 3:[1, 3], 4:[2, 2], 5:[1, 5], 6:[2, 3], 7:[1, 7], 8:[2, 4], 9:[3, 3], 10:[2, 5]}
    figure_dict = {1:(10, 10), 2:(10, 5), 3:(15, 5), 4:(12, 12), 5:(25, 5), 6:(15, 10), 7:(35, 5), 8:(20, 5), 9:(20, 20), 10:(25, 10)}
    print('analysis input X:{} y:{}, y_classes:{}'.format(X.shape, y.shape, y_classes))
    CVS_scores, Class_accuracy, Auc_average, importances = analyse_attribute_by_random_forest_classifer(
        X, y, feature_input, n_classes=y_len,
        plot_index=plot_index[y_len], figsize=figure_dict[y_len],
        random_seed=random_seed, axes_title_list=list(y_replace_dict.keys()),
        tree_num=tree_num,
        plot_title_charter=title_string,
        path_saved=path_saved
    )

    return CVS_scores, Class_accuracy, Auc_average, importances

# a = get_random_logging_dataframe()
# data_correction_analyse_by_tree(a, ['GR', 'AC', 'CNL', 'DEN'], 'Type', 10,
#                                 y_replace_dict={'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5}, title_string='fffffffffuck')
