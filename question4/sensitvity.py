import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

# Read Dataset to Dataframe
df = pd.read_csv('./question3/dataset.csv', header=None)
print(df.head())

# Divide Features and Labels
X = df.iloc[:, 0:7]
Y = df.iloc[:, 7]


# Transform Dataframe to Numpy Array
X = X.values
Y = Y.values
X, Y = shuffle(X, Y, random_state=42)

model = tf.keras.models.load_model(
    './question3/saved_model',
    custom_objects=None, compile=True)

# 步骤1: 确定目标变量
target_variable = model.output

# 步骤2: 选择输入参数
input_parameters = model.input

# 步骤3: 定义变化范围
# 这里假设对每个输入参数进行±10%的变化
percentage_change = 0.1

# 步骤4: 执行模型评估
original_predictions = model.predict(X)
sensitivity_results = []

for i in range(input_parameters.shape[1]):
    # 将输入参数增加或减少10%
    perturbed_input = np.copy(X)
    perturbed_input[:, i] *= (1 + percentage_change)
    
    # 使用模型预测
    perturbed_predictions = model.predict(perturbed_input)
    
    # 计算目标变量的变化
    sensitivity = np.mean(np.abs(original_predictions - perturbed_predictions), axis=0)
    sensitivity_results.append(sensitivity)

# 步骤5: 分析结果
average_sensitivity = np.mean(sensitivity_results, axis=0)
print(sensitivity_results)
most_sensitive_feature_index = np.argmax(average_sensitivity)

# 步骤6: 验证结果（可选）

# 步骤7: 解释结果
print(f"The most sensitive feature is feature {most_sensitive_feature_index}")

# 步骤8: 提出建议（可选）
# 根据分析结果提出可能的调整、优化或决策建议
sensitivity_results = [0.000247, 0.00200844, 0.00319628, 0.0088265, 0.01488384, 0.01040762, 0.00727562]
plt.bar(['set_disparity', 'breakpoint_won', 'untouchable_shot_times', 'unforced_error_times', 'Momentum_disparity', 'serve_ace_times', 'net_pt_won_times'],sensitivity_results, color=['orange'])
plt.title('Sensitivity analysis results')
plt.ylabel('sensitivity')
plt.xlabel('features')
for i, value in enumerate(sensitivity_results):
    plt.text(i, value + 0.1, str(value), ha='center', va='bottom')
plt.show()
