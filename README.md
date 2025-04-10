# meituan
#美团的数据清洗与分析工程
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from joblib import dump
import numpy as np

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_clean_data(file_path):
    """
    数据加载与清洗函数
    :param file_path: Excel文件路径
    :return: 清洗后的订单数据和基础信息数据
    """
    try:
        # 读取数据
        order_data = pd.read_excel(file_path, sheet_name="表1_门店分站龄的订单规模表")
        store_info = pd.read_excel(file_path, sheet_name="表2_门店基础信息表")
        print("文件读取成功！")

        # 清洗订单数据
        order_data = order_data.copy()
        order_data = order_data.dropna(subset=['站点站龄'])
        order_data['站点站龄'] = order_data['站点站龄'].astype(str).str.replace('M', '').astype(int)
        order_data['开业月份'] = pd.to_datetime(order_data['开业月份'], format='%Y%m')
        order_data = order_data[order_data['站点站龄'].between(0, 21)]  # 过滤有效站龄

        # 清洗基础信息数据
        store_info = store_info.copy()
        store_info = store_info.dropna(subset=['覆盖户数'])
        store_info['覆盖户数'] = store_info['覆盖户数'].astype(str).str.replace(',', '').astype(int)
        store_info['美团月活跃用户数'] = store_info['美团月活跃用户数'].astype(str).str.replace(',', '').astype(int)

        return order_data, store_info

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到，请检查文件路径或U盘连接。")
        raise
    except Exception as e:
        print(f"数据加载或清洗时发生错误：{e}")
        raise


def feature_engineering(order_data, store_info):
    """
    特征工程函数
    :param order_data: 清洗后的订单数据
    :param store_info: 清洗后的基础信息数据
    :return: 合并后的数据集和关键特征矩阵
    """
    # 数据合并
    merged_data = pd.merge(order_data, store_info, on='门店id', how='left')

    # 衍生核心业务特征
    merged_data['L5用户占比'] = (
        merged_data['美团月活_仅居住_L5占比'] +
        merged_data['美团月活_仅工作_L5占比'] +
        merged_data['美团月活_工作&居住_L5占比']
    ) / 3

    merged_data['订单占比_1.5km'] = merged_data['距离1.5km以下日均订单量'] / merged_data['站日均订单量']
    merged_data['总订单量'] = merged_data.filter(like='距离').sum(axis=1)

    # 其他衍生特征
    merged_data['用户价值指数'] = (
        merged_data['美团月活_仅居住_L5占比'] * 2 +
        merged_data['美团月活_仅工作_L5占比'] * 1.5 +
        merged_data['美团月活_工作&居住_L5占比'] * 3
    ) / 3

    merged_data['超市密度'] = merged_data['门店覆盖范围内超市数'] / merged_data['覆盖户数'] * 1000
    merged_data['运营月份'] = (pd.to_datetime('2025-03-17') - merged_data['开业月份']).dt.days // 30

    # 处理数值型特征缺失值
    num_cols = merged_data.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        merged_data[col] = merged_data[col].fillna(merged_data[col].median())

    # 定义关键特征
    key_features = [
        '距离1.5km以下日均订单量', '距离1.5~2.1km日均订单量',
        '距离2.1~2.8km日均订单量', '覆盖户数', '美团月活跃用户数',
        'L5用户占比', '订单占比_1.5km', '用户价值指数',
        '超市密度', '运营月份', '总订单量'
    ]

    # 检查特征存在性
    missing_features = [col for col in key_features if col not in merged_data.columns]
    if missing_features:
        raise ValueError(f"错误：以下特征未定义 - {missing_features}")

    X = merged_data[key_features]
    y = merged_data['站日均订单量']

    return merged_data, X, y


def train_and_evaluate_model(X, y, test_size=0.2, random_state=42):
    """
    模型训练与评估函数
    :param X: 特征矩阵
    :param y: 目标变量
    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 训练好的模型、预测结果和标准化器
    """
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 定义XGBoost模型
    xgb_model = XGBRegressor(random_state=random_state, n_jobs=-1)

    # 定义调整后的超参数网格，增加正则化，减小树的深度等
    param_grid = {
        'n_estimators': [500, 1000, 1500],  # 减少树的数量
        'learning_rate': [0.03, 0.05, 0.1],  # 降低学习率
        'max_depth': [2, 3, 4],  # 减小树的深度
        'subsample': [0.6, 0.7, 0.8],  # 减少样本采样比例
        'colsample_bytree': [0.6, 0.7, 0.8],  # 减少列采样比例
        'reg_alpha': [0.1, 0.5, 1],  # 增加L1正则化
        'reg_lambda': [0.5, 1, 2]  # 增加L2正则化
    }

    # 使用网格搜索寻找最优超参数
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='r2',
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # 训练模型
    grid_search.fit(X_train, y_train)

    # 获取最优模型
    best_model = grid_search.best_estimator_

    # 模型预测
    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)

    # 评估指标
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_pred)
    print(f"最优超参数: {grid_search.best_params_}")
    print(f"训练集R²得分: {r2_train:.4f}")
    print(f"测试集XGBoost模型R²得分: {r2_test:.4f}")

    return best_model, y_pred, scaler


def visualize_feature_importance(model, key_features, figsize=(12, 8)):
    """
    特征重要性可视化函数
    :param model: 训练好的XGBoost模型
    :param key_features: 关键特征列表
    :param figsize: 图形大小
    """
    plt.figure(figsize=figsize)
    ax = xgb.plot_importance(
        model,
        importance_type='gain',
        title="XGBoost特征重要性（按增益排序）",
        xlabel="重要性得分",
        ylabel="特征",
        height=0.6,
        ax=None
    )

    # 修正特征标签顺序（按重要性降序排列）
    feature_order = model.feature_importances_.argsort()[::-1]
    correct_labels = [key_features[i] for i in feature_order]
    ax.set_yticklabels(correct_labels)

    plt.tight_layout()
    plt.show()


def plot_learning_curve(model, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    绘制学习曲线
    :param model: 训练好的模型（如XGBRegressor）
    :param X: 特征矩阵
    :param y: 目标变量
    :param cv: 交叉验证折数
    :param train_sizes: 训练集大小比例
    """
    plt.figure(figsize=(12, 6))

    # 生成学习曲线数据
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )

    # 计算均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # 绘制曲线
    plt.plot(train_sizes, train_mean, label='训练集R²', color='blue')
    plt.plot(train_sizes, val_mean, label='验证集R²', color='red')

    # 绘制误差带
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

    plt.xlabel('训练数据量比例')
    plt.ylabel('R²得分')
    plt.title('学习曲线')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """主函数"""
    # 配置参数
    file_path = "D:/meituan/static.xlsx"

    # 1. 数据加载与清洗
    order_data, store_info = load_and_clean_data(file_path)

    # 2. 特征工程
    merged_data, X, y = feature_engineering(order_data, store_info)

    # 3. 模型训练与评估
    xgb_model, y_pred, scaler = train_and_evaluate_model(X, y)

    # 4. 绘制学习曲线
    plot_learning_curve(xgb_model, X, y)

    # 5. 特征重要性可视化
    visualize_feature_importance(xgb_model, key_features=X.columns)

    # 6. 保存模型和标准化器
    dump(xgb_model, "xgb_model.pkl")
    dump(scaler, "scaler.pkl")


if __name__ == "__main__":
    main()

    
