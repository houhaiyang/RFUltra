
import os
import pandas as pd
import numpy as np
from randomForestUltra.RFUltra import RFUltra,createLongPath
from randomForestUltra.plotROC import plotROC


dataPath = 'D:/BGI/27.MSN/RFUltra'
target_info = pd.read_csv(os.path.join(dataPath, 'data/y_info.csv'), index_col=0)

Y_df = pd.read_csv(os.path.join(dataPath, 'data/Y.csv'), index_col=0)

X_Full_df = pd.read_csv(os.path.join(dataPath, 'data/otu_data.csv'), index_col=0)
X_Full_df = X_Full_df.loc[Y_df.index, :]
X_Full_df = np.log(X_Full_df + 1.0)

save_path = os.path.join(dataPath, 'result/')
createLongPath(save_path)

### 示例一：所有样本的一次实验, bootstrap=False, max_rf_samples=None --------------------
target_list = list(Y_df.columns)  # 可以选择部分
path = os.path.join(save_path, f's_all/')
createLongPath(path)
rf_FR = RFUltra(X_Full_df, Y_df, target_list, path, target_info,
                N_FOLD=4, N_REPEATS=25, bootstrap=False,
                max_rf_samples=None,
                largeSampleSize=False, plotBox=True) # 没有 for循环，可以使用 Tkinter 图形用户界面（GUI）

### 示例二：梯度控制训练集 bootstrap=True 重采样 的 样本数 max_rf_samples --------------------
# 由于每个 变量 y 对应的有效样本数可能不同，所以 将 Y_df 拆分成单个 y_df 处理。
for i in range(Y_df.shape[1]):
    y_df = Y_df.iloc[:, i:i + 1].copy()
    y_df = y_df.loc[y_df.iloc[:, 0].isin([0, 1])] # 有效值
    targetName = y_df.columns[0]
    print(f'[{i+1}, {targetName}]')
    X_df = X_Full_df.loc[y_df.index, :]
    num_train = int(np.floor(X_df.shape[0] * 0.75))
    start, end, step = 100, 211, 50
    target_list = [targetName]
    for num in range(start, min(num_train, end) + 1, step):
        print(f'bootstrap max train sample num: {num}, ----------------------------')
        path = os.path.join(save_path, f's{num}/{targetName}/')
        createLongPath(path)
        rf_FR = RFUltra(X_df, y_df, target_list, path, target_info,
                        N_FOLD=4, N_REPEATS=25,
                        bootstrap=True,
                        max_rf_samples=num,
                        largeSampleSize=False,
                        plotBox=False) # for 循环中禁用 Tkinter 图形用户界面（GUI），否则会报错


### 画ROC图 ----------------------------------------------------------
dataPath = 'D:/BGI/27.MSN/RFUltra/'
Y_df = pd.read_csv(os.path.join(dataPath, 'data/Y.csv'), index_col=0)
target_list = list(Y_df.columns)  # 选择有结果的 target 变量
target_info = pd.read_csv(os.path.join(dataPath, 'data/y_info.csv'), index_col=0)
for i in range(len(target_list)):
    target = target_list[i]
    pathModelOutput = os.path.join(dataPath, f'result/s_all/ModelOutput/{target}/')
    print(f'{i + 1}, {target}, {pathModelOutput}')
    # 读取数据
    y_pred = pd.read_csv(os.path.join(pathModelOutput, 'y_pred.csv'), index_col=None, header=None)
    y_true = pd.read_csv(os.path.join(pathModelOutput, 'y_true.csv'), index_col=None, header=None)
    y_pred_shuffled = pd.read_csv(os.path.join(pathModelOutput, 'y_pred_shuffled.csv'), index_col=None, header=None)
    y_true_shuffled = pd.read_csv(os.path.join(pathModelOutput, 'y_true_shuffled.csv'), index_col=None, header=None)

    outPath = os.path.join(dataPath, 'result/s_all/ROCs/')
    createLongPath(outPath)
    title = "Classification of " + target_info.loc[target, "plot_name"]
    plotROC(target, outPath, title, y_pred, y_true, y_pred_shuffled, y_true_shuffled)

