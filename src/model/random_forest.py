import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # 如果是回归用 RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 回归用 mean_squared_error 等

# 读取数据
df = pd.read_csv('D:\MichaelFairy\data\michael\labeled_reactions_with_similarity.csv')

# 假设最后一列是标签（y），其余是特征（X）
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 如果有非数值型特征，需要做编码
# X = pd.get_dummies(X)  # 简单处理

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
clf = RandomForestClassifier(random_state=42)  # 回归用 RandomForestRegressor
clf.fit(X_train, y_train)

# 预测与评估
y_pred = clf.predict(X_test)
print("准确率：", accuracy_score(y_test, y_pred))  # 回归用 mean_squared_error 等

# 如果需要看特征重要性
print("特征重要性：", clf.feature_importances_)
