
<div align=center><img width="597" height="249" src="docs/logo2.png"/></div>


# Random Forest Ultra

[![PyPI - Downloads](https://img.shields.io/pypi/dm/randomForestUltra?label=randomForestUltra%20on%20PyPi)](https://pypi.org/manage/project/randomforestultra/releases/) 

- Author: Haiyang Hou
- Date: 2023-11-08
- Version: v0.3.0
- If you want to use this package, please indicate the source and tell me in "lssues". Free use.

## Current function

* Support for multi-target variables (binary)

* Support for multiple rounds of k-fold cross-validation

* Support for feature importance calculation in the feature matrix

* Support for bootstrap gradient resampling

* Support for calculating mean AUC and computing P-value through random permutation

* Support for plotting average ROC curves

<div align=center><img width="800" height="800" src="docs/RFUltra.png"/></div>


## Description

This project is based on Python 3.7+ and developed using PyCharm on Windows 10+ (11).

-------------

## Installation
Requirements: python>=3.7, numpy, pandas, tqdm, seaborn, matplotlib, scipy, scikit-learn.

Pay attention to the installation of scikit-learn package.

##### 1st | Install numpy scipy matplotlib
```commandline
pip install numpy pandas tqdm seaborn matplotlib scipy -i https://pypi.mirrors.ustc.edu.cn/simple/
```
##### 2nd | Install scikit-learn
```commandline
pip uninstall sklearn # Delete sklearn-0.0.post10.dist-info
pip install scikit-learn -i https://pypi.mirrors.ustc.edu.cn/simple/
```
##### 3rd | Install randomForestUltra

Install through PyPI:
```commandline
pip install randomForestUltra
```
Install through local:
```commandline
pip install dist/randomForestUltra-0.3.0-py3-none-any.whl
```

-------------

## Example usage

Please refer to test.py for an example.


#### Import dependency package
```python
import os
import pandas as pd
import numpy as np
from randomForestUltra.RFUltra import RFUltra,createLongPath
from randomForestUltra.plotROC import plotROC
```
#### Prepare data
```python
dataPath = 'Your_project_path_prefix/RFUltra'
target_info = pd.read_csv(os.path.join(dataPath, 'data/y_info.csv'), index_col=0)

Y_df = pd.read_csv(os.path.join(dataPath, 'data/Y.csv'), index_col=0)

X_Full_df = pd.read_csv(os.path.join(dataPath, 'data/otu_data.csv'), index_col=0)
X_Full_df = X_Full_df.loc[Y_df.index, :]
X_Full_df = np.log(X_Full_df + 1.0)

save_path = os.path.join(dataPath, 'result/')
createLongPath(save_path)
```

#### Example 1: One experiment of all samples
```python
target_list = list(Y_df.columns)
path = os.path.join(save_path, f's_all/')
createLongPath(path)
rf_FR = RFUltra(X_Full_df, Y_df, target_list, path, target_info,
                N_FOLD=4, N_REPEATS=25, bootstrap=False,
                max_rf_samples=None,
                largeSampleSize=False, plotBox=True) # Without a for loop, you can use the GUI.
```

#### Example 2: Control the sample size in the training set bootstrap
```python
for i in range(Y_df.shape[1]):
    y_df = Y_df.iloc[:, i:i + 1].copy()
    y_df = y_df.loc[y_df.iloc[:, 0].isin([0, 1])] # effective value
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
                        plotBox=False) # Disable the GUI in the for loop, otherwise an error will be reported.
```


##### Example 3: Draw ROC diagram
```python
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
```

<div align=center><img width="400" height="400" src="docs/ROC.png"/></div>

-------------

### Additional information
This project is referenced from: https://github.com/jacksklar/AGPMicrobiomeHostPredictions, A broad search for significant microbiome-modifying host variables from the American Gut Project public microbiome data
