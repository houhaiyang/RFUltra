
import os
import sys
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg') # Set up a non-interactive backend !
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef
from tqdm import tqdm
import seaborn as sns

num_iterations = 100

RANDOM_STATE_RF = 347945
RANDOM_STATE_CV = 124213

metrics = ["p_val", "n_samples", "auc_mean", "auc_std", "auc_median",  "shuffled_auc_mean", "shuffled_auc_std",  "shuffled_auc_median", "acc_mean", "acc_std",
           "shuffled_matthews_mean", "shuffled_matthews_std" ,"matthews_mean", "matthews_std", "shuffled_accuracy_mean", "shuffled_accuracy_std"]

def createLongPath(target_path):
    def create_directory(path):
        try:
            os.makedirs(path)
            print("Directory created: ", path)

        except FileExistsError:
            # print("Directory already exists: ", path)
            pass
    target_dir = os.path.split(target_path)[0]
    path_parts = []
    folder = 'test'
    while folder != '':
        target_dir, folder = os.path.split(target_dir)
        if folder != '':
            path_parts.insert(0, folder)
            # print(path_parts)
    for part in path_parts:
        current_path = os.path.join(target_dir, part)
        target_dir = current_path
        # print(current_path)
        create_directory(current_path)

# target_path = 'D:/BGI/02.CSMP/00.ref/AGPMicrobiomeHostPredictions/newcohortsforrf/alc_matched/ROCs/y3_rf.png'
# createLongPath(target_path)
def empiricalPVal(statistic, null_dist):
    ###number of shuffled iterations where performance is >= standard iteration performance
    count = len([val for val in null_dist if val >= statistic])
    p_val = (count + 1 ) /float(len(null_dist) + 1)
    return p_val

class modelResults:
    def __init__(self):
        self.tprs = []
        self.aucs = []
        self.importances = []
        self.accuracy = []
        self.matthews = []
        self.shuffled_accuracy = []
        self.shuffled_aucs = []
        self.shuffled_matthews = []
        self.shuffled_tprs = []
        self.mean_fpr = np.linspace(0, 1, 101)

    def getMetrics(self, cohort_n):
        metrics = pd.Series([])
        metrics.loc["n_samples"] = cohort_n
        metrics.loc["auc_mean"] = np.mean(self.aucs)
        metrics.loc["auc_std"] = np.std(self.aucs)
        metrics.loc["auc_median"] = np.median(self.aucs)
        metrics.loc["shuffled_auc_mean"] = np.mean(self.shuffled_aucs)
        metrics.loc["shuffled_auc_std"] = np.std(self.shuffled_aucs)
        metrics.loc["shuffled_auc_median"] = np.median(self.shuffled_aucs)
        metrics.loc["p_val"] = np.mean([empiricalPVal(stat, self.shuffled_aucs) for stat in self.aucs])
        metrics.loc["acc_mean"] = np.mean(self.accuracy)
        metrics.loc["acc_std"] = np.std(self.accuracy)
        metrics.loc["matthews_mean"] = np.mean(self.matthews)
        metrics.loc["matthews_std"] = np.std(self.matthews)
        metrics.loc["shuffled_matthews_std"] = np.std(self.shuffled_matthews)
        metrics.loc["shuffled_matthews_mean"] = np.mean(self.shuffled_matthews)
        metrics.loc["shuffled_accuracy_mean"] = np.mean(self.shuffled_accuracy)
        metrics.loc["shuffled_accuracy_std"] = np.std(self.shuffled_accuracy)
        return metrics

    def getImportances(self, col_names):
        avg_imps = np.stack(self.importances)
        avg_imps = pd.DataFrame(avg_imps, columns=col_names).mean(axis=0)
        return avg_imps

class RFClassificationUltra:
    def __init__(self, X_df, target_name, cohort, plot, save, title,
                 N_FOLD = 4, N_REPEATS = 25, bootstrap=True, max_rf_samples=0.7, largeSampleSize=False):
        self.X_df = X_df
        self.target_name = target_name
        self.cohort = cohort
        self.plot = plot
        self.save = save
        self.title = title
        self.largeSampleSize = largeSampleSize
        self.N_FOLD = N_FOLD
        self.N_REPEATS = N_REPEATS
        self.max_rf_samples = max_rf_samples
        self.bootstrap = bootstrap


    def classifyFeature(self):
        X, y = self.buildDataSubset()
        list_y_pred,list_y_true,list_y_pred_shuffled, list_y_true_shuffled = self.GroupCV(X, y)
        # 标记
        return list_y_pred,list_y_true,list_y_pred_shuffled, list_y_true_shuffled

    ## Preprocess questionnaire matched-pair cohort for classification
    ##      - taxonomic relative abundance data is log-transformed with a pseudocount of 1
    ##      - abundance data is not normally distributed so this transformation makes it more suitable for classification
    def buildDataSubset(self):
        X = self.X_df
        y = self.cohort[self.target_name].astype(float)
        print(y.value_counts())
        # X = np.log(X + 1.0) # 需要提前预处理
        ## limit cohorts to 2000 samples
        max_samples = 2000
        if len(X) <= max_samples:
            return X, y
        else:
            if not self.largeSampleSize:
                print('Sample size is too large, please specify [largeSampleSize=False], otherwise only the first 2000 samples will be selected.')
                X = X[:max_samples, :]
                y = y[:max_samples]
            return X, y


        ## 25 iteration 4-fold cross validation

    ##      - pairs must be in consecutive rows, are kept grouped between training and test to maintain this balance
    ## 3 standard machine learning classifiers: random forests, ridge-logistic regression, SVM
    ##      - classifiers chosen for performing well with high-dimensional, low sample data that is noisy, and zero-inflated
    ## Target variable shuffled and model trained over same split of data to assess ability for classifier to find signal in noise
    ## Shuffled performance used to obtain significance non-shuffled standard classifiers

    def GroupCV(self, X, y):
        self.rf = modelResults()
        ##100 iterations Group Shuffle-Split Cross Validation (matched case-control pairs remain stratified)
        cv = RepeatedStratifiedKFold(n_splits=self.N_FOLD, n_repeats=self.N_REPEATS,
                                     random_state=RANDOM_STATE_CV)  # 75/25 training/test split for each iteration
        # 创建列表记录每一次的训练结果
        list_y_pred,list_y_true = [],[]
        list_y_pred_shuffled, list_y_true_shuffled = [], []

        # 计算迭代总数
        total_iterations = cv.get_n_splits(X, y)
        # 创建tqdm进度条
        progress_bar = tqdm(total=total_iterations, desc="Progress", position=0, ncols=80)

        for fold_num, (train, test) in enumerate(cv.split(X, y)):
            # 标准化数据
            scaler = StandardScaler().fit(X[train])
            X_train = scaler.transform(X[train])
            X_test = scaler.transform(X[test])
            y_shuffled = shuffle(y)
            # RANDOM FOREST:
            if not self.max_rf_samples==None:
                if X_train.shape[0] < self.max_rf_samples:
                    print('Error ! X_train.shape[0] < max_rf_samples')
                    sys.exit(1)
            y_pred,y_true = self.trainModel(X_train, X_test, y[train], y[test],
                                            shuffle=False, model_type=self.rf,
                                            bootstrap=self.bootstrap,
                                            max_rf_samples=self.max_rf_samples)
            y_pred_shuffled,y_true_shuffled = self.trainModel(X_train, X_test,
                                                              y_shuffled[train], y_shuffled[test],
                                                              shuffle=True, model_type=self.rf,
                                                              bootstrap=self.bootstrap,
                                                              max_rf_samples=self.max_rf_samples)
            list_y_pred.append(y_pred)
            list_y_true.append(y_true)
            list_y_pred_shuffled.append(y_pred_shuffled)
            list_y_true_shuffled.append(y_true_shuffled)


            # 更新进度条
            progress_bar.update(1)


        # if self.plot:
        #     self.rf.plotROC(self.target_name, self.save, self.title, "rf")
        # rf_result = self.rf
        print()
        print()

        return list_y_pred,list_y_true, list_y_pred_shuffled,list_y_true_shuffled #,rf_result

    ##train one of three classifiers on CV iteration
    ##returns performance metrics, feature importances, saves to classifier object
    def trainModel(self, X_train, X_test, y_train, y_test, shuffle, model_type, bootstrap=True, max_rf_samples=0.7):
        # X_train, X_test, y_train, y_test, shuffle, model_type=X_train, X_test, y[train], y[test], False, modelResults()
        if model_type == self.rf:
            max_features = int(np.minimum(np.sqrt(X_train.shape[1]), 100)) # 超过10000个特征时，限制max_features=100
            alg = RandomForestClassifier(n_estimators=512,
                                         min_samples_leaf=1,
                                         n_jobs=-1,
                                         bootstrap=bootstrap,
                                         max_samples=max_rf_samples,
                                         max_features=max_features,
                                         class_weight='balanced',
                                         random_state=RANDOM_STATE_RF)
            alg.fit(X_train, y_train)
            imp = alg.feature_importances_

        # 需要保存下来
        y_pred = alg.predict_proba(X_test)[:, 1]
        y_pred_class = alg.predict(X_test)
        y_true = y_test


        ##Performance Metrics:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        matthew = matthews_corrcoef(y_true, y_pred_class)
        acc = accuracy_score(y_true, y_pred_class)
        if shuffle:
            # results for shuffled target variable (corrupted) dataset used as null hypothesis
            model_type.shuffled_tprs.append(np.interp(self.rf.mean_fpr, fpr, tpr))
            model_type.shuffled_tprs[-1][0] = 0.0
            model_type.shuffled_aucs.append(roc_auc)
            model_type.shuffled_matthews.append(matthew)
            model_type.shuffled_accuracy.append(acc)
        else:
            # results of classifier on cohort
            model_type.importances.append(imp)
            model_type.tprs.append(np.interp(model_type.mean_fpr, fpr, tpr))
            model_type.tprs[-1][0] = 0.0
            model_type.aucs.append(roc_auc)
            model_type.accuracy.append(acc)
            model_type.matthews.append(matthew)

        return y_pred,y_true #,model_type


## Aggregate classification results of all questionnaire variables into single csv file for analysis/comparison
class QuestionnaireResults():
    def __init__(self, iters, col_names, model_name, save_path):
        self.iters = iters
        self.col_names = col_names
        self.model_name = model_name
        self.save_path = save_path
        self.model_results = pd.DataFrame([], columns=metrics)
        self.model_importances = pd.DataFrame([], columns=col_names)
        self.model_aucs = pd.DataFrame([], columns=range(iters))
        self.model_shuffled_aucs = pd.DataFrame([], columns=range(iters))
        # self.model_shuffled_aucs = pd.DataFrame([], columns=range(iters), dtype=float)

    def AppendModelRes(self, model_obj, cohort_n, target_name):
        self.model_results.loc[target_name, :] = model_obj.getMetrics(cohort_n)
        self.model_importances.loc[target_name, :] = model_obj.getImportances(self.col_names)
        self.model_aucs.loc[target_name, :] = model_obj.aucs
        self.model_shuffled_aucs.loc[target_name, :] = model_obj.shuffled_aucs

    def SaveModelDF(self):
        csvPath = os.path.join(self.save_path, f'{self.model_name}_results.csv')
        self.model_results.to_csv(csvPath)
        csvPath = os.path.join(self.save_path, f'AUCs/{self.model_name}_aucs.csv')
        createLongPath(csvPath)
        self.model_aucs.to_csv(csvPath)
        self.model_shuffled_aucs.to_csv(os.path.join(self.save_path, f'AUCs/{self.model_name}_shuffled_aucs.csv'))
        csvPath = os.path.join(self.save_path, f'{self.model_name}_importances.csv')
        createLongPath(csvPath)
        self.model_importances.to_csv(csvPath)

### Plot boxplot of distribution of AUC results from cross-validation for top performing variables
def PlotFeatureBox(model_results, model_aucs, path, model, target_info):
    temp = model_results[(model_results["p_val"] <= 0.05) & (model_results["auc_mean"] >= 0.65)].sort_values \
        ("auc_median", ascending = False)
    boxplotdata = model_aucs.loc[temp.index, :].values
    boxplotdata = pd.DataFrame(boxplotdata, index = target_info.loc[temp.index, "plot_name"]).T
    sns.boxplot(data = boxplotdata, notch = False, showfliers=False, palette = "Blues_r", orient = "h")
    plt.xlabel("AUC")
    plt.xlim(0.5, 1.0)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'auc_dists_{model}.pdf'))
    # plt.show()
    plt.close()

def saveModelOutput(target_name, save_path,
                        list_y_pred, list_y_true, list_y_pred_shuffled, list_y_true_shuffled):
        result_y_pred = pd.DataFrame(list_y_pred)
        # list_y_true是包含100个一维ndarray的list
        dfs = [arr.reset_index(drop=True) for arr in list_y_true]
        result_y_true = pd.DataFrame(dfs).reset_index(drop=True)

        result_y_pred_shuffled = pd.DataFrame(list_y_pred_shuffled)
        dfs = [arr.reset_index(drop=True) for arr in list_y_true_shuffled]
        result_y_true_shuffled = pd.DataFrame(dfs).reset_index(drop=True)

        dir_path = os.path.join(save_path, f'ModelOutput/{target_name}/')
        csvPath = os.path.join(dir_path, f'y_pred.csv')
        createLongPath(csvPath)
        result_y_pred.to_csv(csvPath, header=False, index=False)
        result_y_true.to_csv(os.path.join(dir_path, f'y_true.csv'), header=False, index=False)
        result_y_pred_shuffled.to_csv(os.path.join(dir_path, f'y_pred_shuffled.csv'), header=False, index=False)
        result_y_true_shuffled.to_csv(os.path.join(dir_path, f'y_true_shuffled.csv'), header=False, index=False)

        modelOutput = {
            'result_y_pred': result_y_pred,
            'result_y_true': result_y_true,
            'result_y_pred_shuffled': result_y_pred_shuffled,
            'result_y_true_shuffled': result_y_true_shuffled
        }
        return modelOutput

# def plot_roc(modelOutput, save_path, title, target_name, modelType="rf"):
#     # 后续用R语言绘图，先不用 python
#     print(save_path)


def checkData(X_Full_df, Y_df, target_list, save_path):
    # 判断X_Full_df是否为数值型数据框
    if not isinstance(X_Full_df, pd.DataFrame) or not X_Full_df.dtypes.apply(pd.api.types.is_numeric_dtype).all():
        print('X_Full_df must be a numeric data frame !')
        return False
    # 判断Y_df中是否只包含0、-1和1
    if not isinstance(Y_df, pd.DataFrame) or not all(Y_df.isin([0, -1, 1]).all()):
        print('Y_df can only contain 0, -1 and 1 !')
        return False
    # 判断target_list是否为字符串列表
    if not isinstance(target_list, list) or not all(isinstance(item, str) for item in target_list):
        print('target_list must be a list of strings !')
        return False
    # 判断save_path是否为文件夹路径
    if not isinstance(save_path, str) or not os.path.isdir(save_path):
        print(f'save_path does not exist or is not a directory: {save_path}')
        return False

    return True


def RFUltra(X_Full_df, Y_df, target_list, save_path, target_info,
            N_FOLD=4, N_REPEATS=25, bootstrap=True, max_rf_samples=0.7,
            largeSampleSize=False, plotBox=False):
    # 先判断数据是否标准
    if not checkData(X_Full_df, Y_df, target_list, save_path):
        print("There is a problem with the format of the input data.")
        return False
    else:
        print("Input data check complete.")

    col_names = X_Full_df.columns # 为了计算 importances
    rf_FR = QuestionnaireResults(num_iterations, col_names, "rf", save_path)
    for y_target in target_list: # 尝试变为多线程
        target_name = y_target # .split(".")[0]
        if target_name == "":
            continue
        print(f'[{target_name}] --- --- ---')
        cohort = Y_df[target_name]
        cohort = pd.DataFrame(cohort)
        cohort = cohort[(cohort[target_name] == 0) | (cohort[target_name] == 1)]

        # cohort.index = cohort["num"]
        cohort_n = len(cohort)
        X_df = X_Full_df.loc[cohort.index, :].astype(float).values
        title = "Classification of " + target_info.loc[target_name, "plot_name"]
        CohClass = RFClassificationUltra(X_df=X_df, target_name=target_name, cohort=cohort, plot=False,
                                         save=True, title=title, N_FOLD=N_FOLD, N_REPEATS=N_REPEATS,
                                         bootstrap=bootstrap,
                                         max_rf_samples=max_rf_samples,
                                         largeSampleSize=largeSampleSize)
        list_y_pred,list_y_true,list_y_pred_shuffled, list_y_true_shuffled = CohClass.classifyFeature() # 执行
        saveModelOutput(target_name, save_path,
                        list_y_pred, list_y_true, list_y_pred_shuffled, list_y_true_shuffled)
        # modelOutput = saveModelOutput(target_name, save_path,
        #                 list_y_pred, list_y_true, list_y_pred_shuffled, list_y_true_shuffled)
        # plot_roc(modelOutput, save_path, title, target_name) # 后续用R语言绘图，先不用 python

        # print(CohClass.rf.importances)
        rf_FR.AppendModelRes(CohClass.rf, cohort_n, target_name)

    rf_FR.SaveModelDF()
    # Plot performance of binary cohorts (disease, lifestyle, etc.)
    if plotBox:
        PlotFeatureBox(rf_FR.model_results, rf_FR.model_aucs, save_path, "rf", target_info)

    return rf_FR


if __name__ == "__main__":
    dataPath = 'D:/BGI/27.MSN/RFUltra'
    target_info = pd.read_csv(os.path.join(dataPath, 'data/y_info.csv'), index_col=0)

    Y_df = pd.read_csv(os.path.join(dataPath, 'data/Y.csv'), index_col=0)
    target_list = list(Y_df.columns)  # 可以选择部分

    X_Full_df = pd.read_csv(os.path.join(dataPath, 'data/otu_data.csv'), index_col=0)
    X_Full_df = X_Full_df.loc[Y_df.index, :]
    X_Full_df = np.log(X_Full_df + 1.0)

    save_path = os.path.join(dataPath, 'result/')
    # rf_FR = RFUltra(X_Full_df, Y_df, target_list, save_path, target_info)
    rf_FR = RFUltra(X_Full_df, Y_df, target_list, save_path, target_info,
                    N_FOLD = 4, N_REPEATS = 25, bootstrap=True,
                    max_rf_samples=0.7, largeSampleSize=False, plotBox=True)
