# -*- coding:utf-8 -*-
# @Time    : 2021/12/2 15:07
# @Author  : tianjinmouth, Celeste Liu

"""
Learning Triggers for Heterogeneous Treatment Effects
https://arxiv.org/pdf/1902.00087.pdf
"""
import warnings

warnings.filterwarnings("ignore")
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy.stats as stats
#from utils import check_parameters, df2array, increasing_constrain
#from visualization_tree_json import rf_generate_json

# 刘老师，大概整理了一下，也不知道怎么写。您先看看!
# 1.DecisionTree class可能得改，如果多了一些要存的东西的话
# 2.HonestTreeClassifier我在里面已经加了treatment必须是0，1，2，3的判断了，并且已经新增了self.B=0或者1的判断，1是binary
# 3.tree_node_summary感觉可以不改，evaluate_obj里面能计算F，Fc，H的数据即可，把CTL里面的util一些东西搬过来可能就可以了
# 4.evaluate_obj里面通过self.honest和self.B区分不同计算方法
# 5.control_name=0不确定
# 6.prune不确定是否适用
# 7.gain里面的p和norm_I不确定
# 8.先测HonestTreeClassifier，然后测HonestRandomForestClassifier，最后再看能不能plot，plot不好改就算了

def increasing_constrain(currentNodeSummary):
    """
    强制单调性约束函数
    :param currentNodeSummary: 节点的各个Treatment的信息
    :return: int (1单调递增，-1单调递减)
    """
    keys = list(currentNodeSummary.keys())
    keys.remove(0)
    x = sorted(keys)
    x.append(0)
    y = []
    for i in x:
        # print(isinstance(currentNodeSummary[i], list))
        if isinstance(currentNodeSummary[i], list):
            y.append(currentNodeSummary[i][2])
        else:
            y.append(currentNodeSummary[i])
    return is_decreasing_list(y)

def df2array(x):
    """
    dataframe转多维数组
    :param x:
    :return:
    """
    if isinstance(x, pd.DataFrame):
        return x.values
    else:
        return x

def is_decreasing_list(y):
    for i in range(len(y) - 1):
        if y[i] < y[i + 1]:
            return 0
    return 1

def check_parameters(parameters_dict):
    """
    检查搜索参数是否合法
    :param parameters_dict:
    :return:
    """
    for key in parameters_dict.keys():
        if key == 'max_depth':
            for item in parameters_dict[key]:
                assert 1 <= item <= 15, "树的深度超出指定范围"
        elif key == 'max_features':
            for item in parameters_dict[key]:
                assert 0.1 <= item <= 1.0, "特征采样比例超出指定范围"
        elif key == 'min_samples_leaf':
            for item in parameters_dict[key]:
                assert 0 < item, "叶节点最小样本数应该大于0"
        elif key == 'min_samples_treatment':
            for item in parameters_dict[key]:
                assert 0 < item, "叶节点每个Treatment最小样本数应该大于0"
        elif key == 'n_reg':
            for item in parameters_dict[key]:
                assert 0 < item, "正则化参数应该大于0"
        else:
            raise KeyError()


class DecisionTree:
    """ Tree Node Class

    Tree node class to contain all the statistics of the tree node.

    Parameters
    ----------

    col : int, optional (default = -1)
        The column index for splitting the tree node to children nodes.

    value : float, optional (default = None)
        The value of the feature column to split the tree node to children nodes.

    trueBranch : object of DecisionTree
        The true branch tree node (feature > value).

    falseBranch : object of DecisionTree
        The false branch tree node (feature <= value).

    results : dictionary
        The classification probability Pr(1) for each experiment group in the tree node.

    summary : dictionary
        Summary statistics of the tree nodes, including impurity, sample size, uplift score, etc.

    maxDiffTreatment : string
        The treatment name generating the maximum difference between treatment and control group.

    maxDiffSign : float
        The sign of the maximum difference (1. or -1.).

    nodeSummary : dictionary
        Summary statistics of the tree nodes {treatment: [y_mean, n]}, where y_mean stands for the target metric mean
        and n is the sample size.

    backupResults : dictionary
        The conversion probabilities in each treatment in the parent node {treatment: y_mean}. The parent node
        information is served as a backup for the children node, in case no valid statistics can be calculated from the
        children node, the parent node information will be used in certain cases.

    bestTreatment : string
        The treatment name providing the best uplift (treatment effect).

    upliftScore : list
        The uplift score of this node: [max_Diff, p_value], where max_Diff stands for the maximum treatment effect, and
        p_value stands for the p_value of the treatment effect.

    matchScore : float
        The uplift score by filling a trained tree with validation dataset or testing dataset.

    """

    def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None,
                 results=None, summary=None, maxDiffTreatment=None,
                 maxDiffSign=1., nodeSummary=None, backupResults=None,
                 bestTreatment=None, upliftScore=None, matchScore=None):
        self.col = col
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results  # None for nodes, not None for leaves
        self.summary = summary
        # the treatment with max( |p(y|treatment) - p(y|control)| )
        self.maxDiffTreatment = maxDiffTreatment
        # the sign for p(y|maxDiffTreatment) - p(y|control)
        self.maxDiffSign = maxDiffSign
        self.nodeSummary = nodeSummary
        self.backupResults = backupResults
        self.bestTreatment = bestTreatment
        self.upliftScore = upliftScore
        # match actual treatment for validation and testing
        self.matchScore = matchScore

# Honest Tree Classifier
class HonestTreeClassifier:
    """ Honest Tree Classifier for Classification Task.

    A uplift tree classifier estimates the individual treatment effect by modifying the loss function in the
    classification trees.

    The uplift tree classifier is used in uplift random forest to construct the trees in the forest.

    Parameters
    ----------

    max_features: int, optional (default=10)
        The number of features to consider when looking for the best split.

    max_depth: int, optional (default=5)
        The maximum depth of the tree.

    min_samples_leaf: int, optional (default=100)
        The minimum number of samples required to be split at a leaf node.

    min_samples_treatment: int, optional (default=10)
        The minimum number of samples required of the experiment group to be split at a leaf node.

    n_reg: int, optional (default=10)
        The regularization parameter defined in Rzepakowski et al. 2012, the weight (in terms of sample size) of the
        parent node influence on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.

    control_name: string
        The name of the control group (other experiment groups will be regarded as treatment groups)

    normalization: boolean, optional (default=True)
        The normalization factor defined in Rzepakowski et al. 2012, correcting for tests with large number of splits
        and imbalanced treatment and control splits

    """

    def __init__(self, max_features=None, max_depth=3, min_samples_leaf=100,
                 min_samples_treatment=10, n_reg=100,
                 control_name=0, normalization=False, is_constrained=False,
                 is_continuous_method=False, show_log=False, B=0, honest=False,
                 weight=0.5, cnt_prune=0, seed=0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.max_features = max_features
        self.fitted_uplift_tree = None
        self.control_name = control_name
        self.normalization = normalization
        self.is_constrained = is_constrained
        self.show_log = show_log
        self.B = B
        self.honest = honest
        self.weight = weight
        self.cnt_prune = cnt_prune
        self.seed = seed

    def fit(self, X, treatment, y):
        """ Fit the uplift model.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.

        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        Returns
        -------
        self : object
        """
        assert len(X) == len(y) and len(X) == len(treatment), 'Data length must be equal for X, treatment, and y.'
        assert min(treatment.value_counts().index) == 0, 'Treatment should be 0, 1, 2, ... and control group must be zero'

        if treatment.value_counts().shape[0] == 2:
            self.B = 1
        elif treatment.value_counts().shape[0] > 2:
            self.B = 0

        if self.honest:
            # ----------------------------------------------------------------
            # Split data
            # ----------------------------------------------------------------
            train_x, val_x, train_y, val_y, train_t, val_t = train_test_split(X, y, treatment, random_state=self.seed,
                                                                              shuffle=True,
                                                                              test_size=0.7)
            # get honest/estimation portion
            train_x, est_x, train_y, est_y, train_t, est_t = train_test_split(train_x, train_y, train_t, shuffle=True,
                                                                              random_state=self.seed, test_size=0.8)
        else:
            pass

        self.treatment_group = list(set(treatment))
        # 返回训练好的树模型
        if self.honest:
            fitted_uplift_tree = self._honest_fit(
                train_x, train_t, train_y, val_x, val_t, val_y, est_x, est_t, est_y,
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                depth=1, min_samples_treatment=self.min_samples_treatment,
                n_reg=self.n_reg, parentNodeSummary=None, is_constrained=self.is_constrained, show_log=self.show_log
            )
        else:
            fitted_uplift_tree = self._fit(
                X, treatment, y,
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                depth=1, min_samples_treatment=self.min_samples_treatment,
                n_reg=self.n_reg, parentNodeSummary=None, is_constrained=self.is_constrained, show_log=self.show_log
            )
        self.fitted_uplift_tree = fitted_uplift_tree
        return fitted_uplift_tree

    # Prune Trees
    def prune(self, X, treatment, y, minGain=0.0001, rule='maxAbsDiff'):
        """ Prune the uplift model.
        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        minGain : float, optional (default = 0.0001)
            The minimum gain required to make a tree node split. The children
            tree branches are trimmed if the actual split gain is less than
            the minimum gain.
        rule : string, optional (default = 'maxAbsDiff')
            The prune rules. Supported values are 'maxAbsDiff' for optimizing
            the maximum absolute difference, and 'bestUplift' for optimizing
            the node-size weighted treatment effect.
        Returns
        -------
        self : object
        """
        assert len(X) == len(y) and len(X) == len(treatment), 'Data length must be equal for X, treatment, and y.'

        self.pruneTree(X, treatment, y,
                       tree=self.fitted_uplift_tree,
                       rule=rule,
                       minGain=minGain,
                       notify=False,
                       n_reg=self.n_reg,
                       parentNodeSummary=None)
        return self

    def pruneTree(self, X, treatment, y, tree, rule='maxAbsDiff', minGain=0., notify=False, n_reg=0, parentNodeSummary=None):
        """Prune one single tree node in the uplift model.
        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        minGain : float, optional (default = 0.0001)
            The minimum gain required to make a tree node split. The children tree branches are trimmed if the actual
            split gain is less than the minimum gain.
        rule : string, optional (default = 'maxAbsDiff')
            The prune rules. Supported values are 'maxAbsDiff' for optimizing the maximum absolute difference, and
            'bestUplift' for optimizing the node-size weighted treatment effect.
        Returns
        -------
        self : object
        """
        # Current Node Summary for Validation Data Set
        currentNodeSummary = self.tree_node_summary(
            treatment, y, min_samples_treatment=self.min_samples_treatment,
            n_reg=n_reg, parentNodeSummary=parentNodeSummary
        )
        tree.nodeSummary = currentNodeSummary
        # Divide sets for child nodes
        X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment, y, tree.col, tree.value)

        # recursive call for each branch
        if tree.trueBranch.results is None:
            self.pruneTree(X_l, w_l, y_l, tree.trueBranch, rule, minGain,
                           notify, n_reg,
                           parentNodeSummary=currentNodeSummary)
        if tree.falseBranch.results is None:
            self.pruneTree(X_r, w_r, y_r, tree.falseBranch, rule, minGain,
                           notify, n_reg,
                           parentNodeSummary=currentNodeSummary)

        # merge leaves (potentially)
        if (tree.trueBranch.results is not None and
                tree.falseBranch.results is not None):
            if rule == 'maxAbsDiff':
                # Current D
                if (tree.maxDiffTreatment in currentNodeSummary and
                        self.control_name in currentNodeSummary):
                    currentScoreD = tree.maxDiffSign * (currentNodeSummary[tree.maxDiffTreatment][0]
                                                        - currentNodeSummary[self.control_name][0])
                else:
                    currentScoreD = 0

                # trueBranch D
                trueNodeSummary = self.tree_node_summary(
                    w_l, y_l, min_samples_treatment=self.min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )
                if (tree.trueBranch.maxDiffTreatment in trueNodeSummary and
                        self.control_name in trueNodeSummary):
                    trueScoreD = tree.trueBranch.maxDiffSign * (trueNodeSummary[tree.trueBranch.maxDiffTreatment][0]
                                                                - trueNodeSummary[self.control_name][0])
                    trueScoreD = (
                            trueScoreD
                            * (trueNodeSummary[tree.trueBranch.maxDiffTreatment][1]
                               + trueNodeSummary[self.control_name][1])
                            / (currentNodeSummary[tree.trueBranch.maxDiffTreatment][1]
                               + currentNodeSummary[self.control_name][1])
                    )
                else:
                    trueScoreD = 0

                # falseBranch D
                falseNodeSummary = self.tree_node_summary(
                    w_r, y_r, min_samples_treatment=self.min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )
                if (tree.falseBranch.maxDiffTreatment in falseNodeSummary and
                        self.control_name in falseNodeSummary):
                    falseScoreD = (
                            tree.falseBranch.maxDiffSign *
                            (falseNodeSummary[tree.falseBranch.maxDiffTreatment][0]
                             - falseNodeSummary[self.control_name][0])
                    )

                    falseScoreD = (
                            falseScoreD *
                            (falseNodeSummary[tree.falseBranch.maxDiffTreatment][1]
                             + falseNodeSummary[self.control_name][1])
                            / (currentNodeSummary[tree.falseBranch.maxDiffTreatment][1]
                               + currentNodeSummary[self.control_name][1])
                    )
                else:
                    falseScoreD = 0

                if ((trueScoreD + falseScoreD) - currentScoreD <= minGain or
                        (trueScoreD + falseScoreD < 0.)):
                    self.cnt_prune += 1
                    tree.trueBranch, tree.falseBranch = None, None
                    tree.results = tree.backupResults

            elif rule == 'bestUplift':
                # Current D
                if (tree.bestTreatment in currentNodeSummary and
                        self.control_name in currentNodeSummary):
                    currentScoreD = (
                            currentNodeSummary[tree.bestTreatment][0]
                            - currentNodeSummary[self.control_name][0]
                    )
                else:
                    currentScoreD = 0

                # trueBranch D
                trueNodeSummary = self.tree_node_summary(
                    w_l, y_l, min_samples_treatment=self.min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )
                if (tree.trueBranch.bestTreatment in trueNodeSummary and
                        self.control_name in trueNodeSummary):
                    trueScoreD = (
                            trueNodeSummary[tree.trueBranch.bestTreatment][0]
                            - trueNodeSummary[self.control_name][0]
                    )
                else:
                    trueScoreD = 0

                # falseBranch D
                falseNodeSummary = self.tree_node_summary(
                    w_r, y_r, min_samples_treatment=self.min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )
                if (tree.falseBranch.bestTreatment in falseNodeSummary and
                        self.control_name in falseNodeSummary):
                    falseScoreD = (
                            falseNodeSummary[tree.falseBranch.bestTreatment][0]
                            - falseNodeSummary[self.control_name][0]
                    )
                else:
                    falseScoreD = 0
                gain = ((1. * len(y_l) / len(y) * trueScoreD
                         + 1. * len(y_r) / len(y) * falseScoreD)
                        - currentScoreD)
                if gain <= minGain or (trueScoreD + falseScoreD < 0.):
                    self.cnt_prune += 1
                    tree.trueBranch, tree.falseBranch = None, None
                    tree.results = tree.backupResults
        return self

    def predict(self, X, full_output=False):
        '''
        Returns the recommended treatment group and predicted optimal
        probability conditional on using the recommended treatment group.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        full_output : bool, optional (default=False)
            Whether the UpliftTree algorithm returns upliftScores, pred_nodes
            alongside the recommended treatment group and p_hat in the treatment group.

        Returns
        -------
        df_res : DataFrame, shape = [num_samples, (num_treatments + 1)]
            A DataFrame containing the predicted delta in each treatment group,
            the best treatment group and the maximum delta.

        '''

        p_hat_optimal = []
        treatment_optimal = []
        pred_nodes = {}
        upliftScores = []
        for xi in range(len(X)):
            pred_leaf, upliftScore = self.classify(X.iloc[[xi]], self.fitted_uplift_tree, dataMissing=False)
            # Predict under uplift optimal treatment
            opt_treat = max(pred_leaf, key=pred_leaf.get)
            p_hat_optimal.append(pred_leaf[opt_treat])
            treatment_optimal.append(opt_treat)
            if full_output:
                if xi == 0:
                    for key_i in pred_leaf:
                        pred_nodes[key_i] = [pred_leaf[key_i]]
                else:
                    for key_i in pred_leaf:
                        pred_nodes[key_i].append(pred_leaf[key_i])
                upliftScores.append(upliftScore)
        if full_output:
            return treatment_optimal, p_hat_optimal, upliftScores, pred_nodes
        else:
            return treatment_optimal, p_hat_optimal

    def divideSet(self, X, treatment, y, column, value):
        '''
        Tree node split.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        column : int
                The column used to split the data.
        value : float or int
                The value in the column for splitting the data.

        Returns
        -------
        (X_l, X_r, treatment_l, treatment_r, y_l, y_r) : list of ndarray
                The covariates, treatments and outcomes of left node and the right node.
        '''
        # for int and float values
        if isinstance(value, int) or isinstance(value, float):
            filt = X.iloc[:, column] >= value
        else:  # for strings
            filt = X.iloc[:, column] == value

        return X[filt], X[~filt], treatment[filt], treatment[~filt], y[filt], y[~filt]

    def ace_trigger(self, y, t, B, trigger):
        if B == 0:
            treat = t >= trigger
        elif B == 1:
            treat = t >= 0.5
        control = ~treat

        yt = y[treat]
        yc = y[control]

        mu1 = 0.0
        mu0 = 0.0
        if yt.shape[0] != 0:
            mu1 = np.mean(yt)
        if yc.shape[0] != 0:
            mu0 = np.mean(yc)

        return mu1 - mu0

    def tau_squared_trigger(self, outcome, treatment, min_size=1, is_honest=False, outcome_val=None, treatment_val=None):
        """Continuous case"""
        total = outcome.shape[0]

        return_val = (-np.inf, -np.inf)

        if total == 0:
            return return_val

        unique_treatment = np.unique(treatment)

        if unique_treatment.shape[0] == 1:
            return return_val
        # 要改
        unique_treatment = (unique_treatment[1:] + unique_treatment[:-1]) / 2
        unique_treatment = unique_treatment[1:]

        yy = np.tile(outcome, (unique_treatment.shape[0], 1))
        tt = np.tile(treatment, (unique_treatment.shape[0], 1))

        x = np.transpose(np.transpose(tt) > unique_treatment)

        tt[x] = 1
        tt[np.logical_not(x)] = 0

        treat_num = np.sum(tt == 1, axis=1)
        cont_num = np.sum(tt == 0, axis=1)
        min_size_idx = np.where(np.logical_and(
            treat_num >= min_size, cont_num >= min_size))

        unique_treatment = unique_treatment[min_size_idx]
        tt = tt[min_size_idx]
        yy = yy[min_size_idx]

        if tt.shape[0] == 0:
            return return_val

        y_t_m = np.sum((yy * (tt == 1)), axis=1) / np.sum(tt == 1, axis=1)
        y_c_m = np.sum((yy * (tt == 0)), axis=1) / np.sum(tt == 0, axis=1)

        if is_honest:
            total_val = outcome_val.shape[0]

            yyv = np.tile(outcome_val, (unique_treatment.shape[0], 1))
            ttv = np.tile(treatment_val, (unique_treatment.shape[0], 1))

            xv = np.transpose(np.transpose(ttv) > unique_treatment)

            ttv[xv] = 1
            ttv[np.logical_not(xv)] = 0

            ttv = ttv[min_size_idx]
            yyv = yyv[min_size_idx]

            treat_num = np.sum(ttv == 1, axis=1)
            cont_num = np.sum(ttv == 0, axis=1)
            min_size_idx = np.where(np.logical_and(
                treat_num >= min_size, cont_num >= min_size))

            unique_treatment = unique_treatment[min_size_idx]
            ttv = ttv[min_size_idx]
            yyv = yyv[min_size_idx]

            if tt.shape[0] == 0:
                return return_val

            y_t_m_v = np.sum((yyv * (ttv == 1)), axis=1) / np.sum(ttv == 1, axis=1)
            y_c_m_v = np.sum((yyv * (ttv == 0)), axis=1) / np.sum(ttv == 0, axis=1)

            train_effect = y_t_m - y_c_m
            train_err = train_effect ** 2

            val_effect = y_t_m_v - y_c_m_v
            # val_err = val_effect ** 2

            train_mse = (1 - self.weight) * (total * train_err)
            cost = self.weight * total_val * np.abs(train_effect - val_effect)
            obj = (train_mse - cost) / (np.abs(total - total_val) + 1)

            argmax_obj = np.argmax(obj)
            best_effect = train_effect[argmax_obj]
            best_split = unique_treatment[argmax_obj]
        else:
            effect = y_t_m - y_c_m
            err = effect ** 2

            max_err = np.argmax(err)

            best_effect = effect[max_err]
            best_split = unique_treatment[max_err]

        return best_effect, best_split

    def variance_trigger(self, y, treatment, B, trigger):
        """
        Calculate Variance.
        """
        # ----------------------------------------------------------------
        # Variance
        # ----------------------------------------------------------------
        treat_vect = treatment
        if B == 0:
            treat = treat_vect >= trigger
        elif B == 1:
            treat = treat_vect == 1
        control = ~treat

        if y.shape[0] == 0:
            return np.array([np.inf, np.inf])

        yt = y[treat]
        yc = y[control]

        if yt.shape[0] == 0:
            var_t = np.var(y)
        else:
            var_t = np.var(yt)

        if yc.shape[0] == 0:
            var_c = np.var(y)
        else:
            var_c = np.var(yc)

        return var_t, var_c

    def evaluate_obj(self, treatment, y, B, honest, val_treatment=None, val_y=None):
        """
        Calculate obj.
        """
        # ----------------------------------------------------------------
        # Obj
        # ----------------------------------------------------------------
        if B == 0:
            if honest:
                total_train = y.shape[0]
                total_val = val_y.shape[0]

                return_val = (-np.inf, -np.inf)

                if total_train == 0 or total_val == 0:
                    return return_val

                train_effect, best_trigger = self.tau_squared_trigger(y, treatment, 2, honest, val_y, val_treatment)
                if train_effect <= -np.inf:
                    return return_val

                val_effect = self.ace_trigger(val_y, val_treatment, B, best_trigger)
                if val_effect <= -np.inf:
                    return return_val

                # 要改
                train_mse = (1 - self.weight) * (total_train * train_effect ** 2)
                cost = self.weight * total_val * np.abs(train_effect - val_effect)
                best_obj = (train_mse - cost) / (np.abs(total_train - total_val) + 1)
            else:
                total_train = y.shape[0]
                return_val = (-np.inf, -np.inf)

                if total_train == 0:
                    return return_val

                train_effect, best_trigger = self.tau_squared_trigger(y, treatment, 2)

                if train_effect <= -np.inf:
                    return return_val

                train_mse = total_train * (train_effect ** 2)
                best_obj = train_mse
        if B == 1:
            best_trigger = -np.inf
            if honest:
                total_train = y.shape[0]
                total_val = val_y.shape[0]

                train_effect = self.ace_trigger(y, treatment, B, 0.5)
                val_effect = self.ace_trigger(val_y, val_treatment, B, 0.5)

                # 要改
                train_mse = (1 - self.weight) * (total_train * train_effect ** 2)
                cost = self.weight * total_val * np.abs(train_effect - val_effect)

                best_obj = (train_mse - cost) / (np.abs(total_train - total_val) + 1)
            else:
                total_train = y.shape[0]

                train_effect = self.ace_trigger(y, treatment, B, 0.5)

                train_mse = total_train * (train_effect ** 2)

                best_obj = train_mse

        return best_obj, best_trigger

    @staticmethod
    def entropyH(p, q=None):
        '''
        Entropy

        Entropy calculation for normalization.

        Args
        ----
        p : float
            The probability used in the entropy calculation.

        q : float, optional, (default = None)
            The second probability used in the entropy calculation.

        Returns
        -------
        entropy : float
        '''
        if q is None and p > 0:
            return -p * np.log(p)
        elif q > 0:
            return -p * np.log(q)
        else:
            return 0

    @staticmethod
    def kl_divergence(pk, qk):
        '''
        Calculate KL Divergence for binary classification.

        sum(np.array(pk) * np.log(np.array(pk) / np.array(qk)))

        Args
        ----
        pk : float
            The probability of 1 in one distribution.
        qk : float
            The probability of 1 in the other distribution.

        Returns
        -------
        S : float
            The KL divergence.
        '''
        if qk < 0.1 ** 6:
            qk = 0.1 ** 6
        elif qk > 1 - 0.1 ** 6:
            qk = 1 - 0.1 ** 6
        # 某个treatment下分为0和为1两种情况考虑
        S = pk * np.log(pk / qk) + (1 - pk) * np.log((1 - pk) / (1 - qk))
        return S

    def normI(self, currentNodeSummary, leftNodeSummary, rightNodeSummary, control_name, alpha=0.9):
        '''
        Normalization factor.

        Args
        ----
        currentNodeSummary : dictionary
            The summary statistics of the current tree node.

        leftNodeSummary : dictionary
            The summary statistics of the left tree node.

        rightNodeSummary : dictionary
            The summary statistics of the right tree node.

        control_name : string
            The control group name.

        alpha : float
            The weight used to balance different normalization parts.

        Returns
        -------
        norm_res : float
            Normalization factor.
        '''
        norm_res = 0
        # n_t, n_c: sample size for all treatment, and control
        # pt_a, pc_a: % of treatment is in left node, % of control is in left node
        n_c = currentNodeSummary[control_name][1]
        n_c_left = leftNodeSummary[control_name][1]
        n_t = []
        n_t_left = []
        for treatment_group in currentNodeSummary:
            if treatment_group != control_name:
                n_t.append(currentNodeSummary[treatment_group][1])
                if treatment_group in leftNodeSummary:
                    n_t_left.append(leftNodeSummary[treatment_group][1])
                else:
                    n_t_left.append(0)
        pt_a = 1. * np.sum(n_t_left) / (np.sum(n_t) + 0.1)
        pc_a = 1. * n_c_left / (n_c + 0.1)
        # Normalization Part 1
        norm_res += (
                alpha * self.entropyH(1. * np.sum(n_t) / (np.sum(n_t) + n_c), 1. * n_c / (np.sum(n_t) + n_c))
                * self.kl_divergence(pt_a, pc_a)
        )
        # Normalization Part 2 & 3
        for i in range(len(n_t)):
            pt_a_i = 1. * n_t_left[i] / (n_t[i] + 0.1)
            norm_res += (
                    (1 - alpha) * self.entropyH(1. * n_t[i] / (n_t[i] + n_c), 1. * n_c / (n_t[i] + n_c))
                    * self.kl_divergence(1. * pt_a_i, pc_a)
            )
            norm_res += (1. * n_t[i] / (np.sum(n_t) + n_c) * self.entropyH(pt_a_i))
        # Normalization Part 4
        norm_res += 1. * n_c / (np.sum(n_t) + n_c) * self.entropyH(pc_a)

        # Normalization Part 5
        norm_res += 0.5
        return norm_res

    def tree_node_summary(self, treatment, y, min_samples_treatment=10, n_reg=100, parentNodeSummary=None):
        '''
        Tree node summary statistics.

        Args
        ----
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        min_samples_treatment: int, optional (default=10)
            The minimum number of samples required of the experiment group t be split at a leaf node.
        n_reg :  int, optional (default=10)
            The regularization parameter defined in Rzepakowski et al. 2012,
            the weight (in terms of sample size) of the parent node influence
            on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.
        parentNodeSummary : dictionary
            Node summary statistics of the parent tree node.

        Returns
        -------
        nodeSummary : dictionary
            The node summary of the current tree node.
        '''
        # returns {treatment_group: p(1)}
        results = self.group_uniqueCounts(treatment, y)
        # node Summary: {treatment_group: [p(1), size]}
        nodeSummary = {}
        # iterate treatment group
        for r in results:
            n1 = results[r][1]
            ntot = results[r][0] + n1
            if parentNodeSummary is None:  # 父节点为空
                y_mean = n1 / ntot
                real_y_mean = y_mean
            elif ntot > min_samples_treatment:
                # 计算左右子树的y_hat(公式3.8)
                y_mean = (n1 + parentNodeSummary[r][0] * n_reg) / (ntot + n_reg)
                real_y_mean = n1 / ntot
            else:  # 小于最少样本数等于父节点的值
                y_mean = parentNodeSummary[r][0]
                real_y_mean = y_mean
            nodeSummary[r] = [y_mean, ntot, real_y_mean]

        return nodeSummary

    def group_uniqueCounts(self, treatment, y):
        '''
        Count sample size by experiment group.

        Args
        ----
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        Returns
        -------
        results : dictionary
                The control and treatment sample size.
        '''
        results = {}
        for t in self.treatment_group:
            filt = treatment == t
            n_t = y[filt].sum()
            results[t] = (filt.sum() - n_t, n_t)
        return results

    def uplift_classification_results(self, treatment, y):
        '''
        Classification probability for each treatment in the tree node.

        Args
        ----
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        Returns
        -------
        res : dictionary
            The probability of 1 in each treatment in the tree node.
        '''
        results = self.group_uniqueCounts(treatment, y)
        res = {}
        for r in results:
            p = float(results[r][1]) / (results[r][0] + results[r][1])  # Y / CNT
            res[r] = round(p, 6)
        return res

    def _fit(self, X, treatment, y, max_depth=10,
                             min_samples_leaf=100, depth=1,
                             min_samples_treatment=10, n_reg=100,
                             parentNodeSummary=None, flag=-1, is_constrained=True, show_log=False):
        '''
        Train the uplift decision tree.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        max_depth: int, optional (default=10)
            The maximum depth of the tree.
        min_samples_leaf: int, optional (default=100)
            The minimum number of samples required to be split at a leaf node.
        depth : int, optional (default = 1)
            The current depth.
        min_samples_treatment: int, optional (default=10)
            The minimum number of samples required of the experiment group to be split at a leaf node.
        n_reg: int, optional (default=10)
            The regularization parameter defined in Rzepakowski et al. 2012,
            the weight (in terms of sample size) of the parent node influence
            on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.
        parentNodeSummary : dictionary, optional (default = None)
            Node summary statistics of the parent tree node.

        Returns
        -------
        object of DecisionTree class
        '''

        if len(X) == 0:
            return DecisionTree()

        # Current Node Info and Summary
        # 返回当前节点(父节点)的不同的组(control,treatment1,...)的y(gmv)均值,样本数
        currentNodeSummary = self.tree_node_summary(treatment, y,
                                                    min_samples_treatment=min_samples_treatment,
                                                    n_reg=n_reg,
                                                    parentNodeSummary=parentNodeSummary)
        currentScore, currentTrigger = self.evaluate_obj(treatment, y, self.B, self.honest, None, None)

        # Prune Stats
        # 剪枝操作（针对纯节点和负增益）
        maxAbsDiff = 0
        maxDiff = -1.
        bestTreatment = self.control_name
        suboptTreatment = self.control_name
        maxDiffTreatment = self.control_name
        maxDiffSign = 0
        # 计算每个treatment组与control组的差异得到uplift score
        for treatment_group in currentNodeSummary:
            if treatment_group != self.control_name:
                diff = (currentNodeSummary[treatment_group][0]
                        - currentNodeSummary[self.control_name][0])
                if abs(diff) >= maxAbsDiff:
                    maxDiffTreatment = treatment_group
                    maxDiffSign = np.sign(diff)
                    maxAbsDiff = abs(diff)
                if diff >= maxDiff:
                    maxDiff = diff
                    suboptTreatment = treatment_group
                    if diff > 0:
                        bestTreatment = treatment_group
        if maxDiff > 0:
            pt = currentNodeSummary[bestTreatment][0]
            nt = currentNodeSummary[bestTreatment][1]
            pc = currentNodeSummary[self.control_name][0]
            nc = currentNodeSummary[self.control_name][1]
            p_value = (1. - stats.norm.cdf((pt - pc) / np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc))) * 2
        else:
            pt = currentNodeSummary[suboptTreatment][0]
            nt = currentNodeSummary[suboptTreatment][1]
            pc = currentNodeSummary[self.control_name][0]
            nc = currentNodeSummary[self.control_name][1]
            p_value = (1. - stats.norm.cdf((pc - pt) / np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc))) * 2
        upliftScore = [maxDiff, p_value]

        bestGain = 0.0
        bestAttribute = None

        # last column is the result/target column, 2nd to the last is the treatment group
        columnCount = X.shape[1]
        assert 0 < self.max_features <= 1.0
        max_features = self.max_features * columnCount
        # 随机选取一些列（默认10个特征）进行树的构建、划分，这里RF采用放回采样
        for col in list(np.random.choice(a=range(columnCount), size=int(max_features), replace=False)):
            columnValues = X.iloc[:, col]
            # unique values
            lsUnique = np.unique(columnValues)

            if (isinstance(lsUnique[0], int) or
                    isinstance(lsUnique[0], float)):
                if len(lsUnique) > 10:  # 找到一组数的分位数值
                    lspercentile = np.percentile(columnValues, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                else:
                    lspercentile = np.percentile(lsUnique, [10, 50, 90])
                lsUnique = np.unique(lspercentile)
            # 统计特征的唯一值数量
            for value in lsUnique:
                # 划分左右子树
                X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment, y, col, value)
                # check the split validity on min_samples_leaf  372
                if len(X_l) < min_samples_leaf or len(X_r) < min_samples_leaf:
                    continue
                # summarize notes
                # Gain -- Entropy or Gini
                # 评分值 = 左、右子树样本数的比例
                leftNodeSummary = self.tree_node_summary(w_l, y_l,
                                                         min_samples_treatment=min_samples_treatment,
                                                         n_reg=n_reg,
                                                         parentNodeSummary=currentNodeSummary)

                rightNodeSummary = self.tree_node_summary(w_r, y_r,
                                                          min_samples_treatment=min_samples_treatment,
                                                          n_reg=n_reg,
                                                          parentNodeSummary=currentNodeSummary)

                # check the split validity on min_samples_treatment
                if set(leftNodeSummary.keys()) != set(rightNodeSummary.keys()):
                    continue
                node_mst = 10 ** 8
                for ti in leftNodeSummary:
                    node_mst = np.min([node_mst, leftNodeSummary[ti][1]])
                    node_mst = np.min([node_mst, rightNodeSummary[ti][1]])
                # 保证每个treatment样本数充足
                if node_mst < min_samples_treatment:
                    continue
                # evaluate the split
                # 计算增益
                # 计算单调性
                if is_constrained:
                    K_L = increasing_constrain(leftNodeSummary)
                    K_R = increasing_constrain(rightNodeSummary)
                    if K_L == 0 or K_R == 0:
                        continue

                if (self.control_name in leftNodeSummary and
                        self.control_name in rightNodeSummary):
                    leftScore1, leftTrigger1 = self.evaluate_obj(w_l, y_l, self.B, self.honest)
                    rightScore2, rightTrigger2 = self.evaluate_obj(w_r, y_r, self.B, self.honest)

                    gain = leftScore1 + rightScore2 - currentScore
                    if self.normalization:
                        norm_factor = self.normI(currentNodeSummary,
                                                 leftNodeSummary,
                                                 rightNodeSummary,
                                                 self.control_name,
                                                 alpha=0.9)
                    else:
                        norm_factor = 1
                    gain = gain / norm_factor
                else:
                    gain = 0
                if gain > bestGain and len(X_l) > min_samples_leaf and len(X_r) > min_samples_leaf:
                    bestGain = gain
                    bestAttribute = (col, value)
                    best_set_left = [X_l, w_l, y_l]
                    best_set_right = [X_r, w_r, y_r]

        dcY = {'impurity': '%.3f' % currentScore, 'samples': '%d' % len(X), 'group_size': ''}
        # Add treatment size
        for treatment_group in currentNodeSummary:
            dcY['group_size'] += ' ' + str(treatment_group) + ': ' + str(currentNodeSummary[treatment_group][1])
        dcY['upliftScore'] = [round(upliftScore[0], 8), round(upliftScore[1], 8)]
        dcY['matchScore'] = round(upliftScore[0], 8)

        # 打印树的分裂情况
        if bestAttribute and show_log:
            if depth == 1 and flag == -1:
                print("root node:", depth, bestAttribute[0], bestAttribute[1])
            else:
                if flag == 1:
                    print("left node:", depth, bestAttribute[0], bestAttribute[1])
                else:
                    print("right node:", depth, bestAttribute[0], bestAttribute[1])
        # 如果增益大于0,递归分裂
        if bestGain > 0 and depth < max_depth:
            trueBranch = self._fit(
                *best_set_left, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary, flag=1, is_constrained=self.is_constrained, show_log=show_log
            )
            falseBranch = self._fit(
                *best_set_right, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary, flag=0, is_constrained=self.is_constrained, show_log=show_log
            )

            return DecisionTree(
                col=bestAttribute[0], value=bestAttribute[1],
                trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY,
                maxDiffTreatment=maxDiffTreatment, maxDiffSign=maxDiffSign,
                nodeSummary=currentNodeSummary,
                backupResults=self.uplift_classification_results(treatment, y),
                bestTreatment=bestTreatment, upliftScore=upliftScore
            )
        else:  # 增益小于等于0的话或者大于最大深度的话结束分裂
            return DecisionTree(
                    results=self.uplift_classification_results(treatment, y),
                    summary=dcY, maxDiffTreatment=maxDiffTreatment,
                    maxDiffSign=maxDiffSign, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )

    def _honest_fit(self, train_x, train_t, train_y, val_x, val_t, val_y, est_x, est_t, est_y, max_depth=10,
                             min_samples_leaf=100, depth=1,
                             min_samples_treatment=10, n_reg=100,
                             parentNodeSummary=None, flag=-1, is_constrained=True, show_log=False):
        '''
        Train the uplift decision tree.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        max_depth: int, optional (default=10)
            The maximum depth of the tree.
        min_samples_leaf: int, optional (default=100)
            The minimum number of samples required to be split at a leaf node.
        depth : int, optional (default = 1)
            The current depth.
        min_samples_treatment: int, optional (default=10)
            The minimum number of samples required of the experiment group to be split at a leaf node.
        n_reg: int, optional (default=10)
            The regularization parameter defined in Rzepakowski et al. 2012,
            the weight (in terms of sample size) of the parent node influence
            on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.
        parentNodeSummary : dictionary, optional (default = None)
            Node summary statistics of the parent tree node.

        Returns
        -------
        object of DecisionTree class
        '''

        if len(train_x) == 0:
            return DecisionTree()

        # Current Node Info and Summary
        # 返回当前节点(父节点)的不同的组(control,treatment1,...)的y(gmv)均值,样本数
        currentNodeSummary = self.tree_node_summary(train_t, train_y,
                                                    min_samples_treatment=min_samples_treatment,
                                                    n_reg=n_reg,
                                                    parentNodeSummary=parentNodeSummary)
        currentScore, currentTrigger = self.evaluate_obj(train_t, train_y, self.B, self.honest, val_t, val_y)
        currentVar, _ = self.variance_trigger(train_t, train_y, self.B, currentTrigger)
        # Prune Stats
        # 剪枝操作（针对纯节点和负增益）
        maxAbsDiff = 0
        maxDiff = -1.
        bestTreatment = self.control_name
        suboptTreatment = self.control_name
        maxDiffTreatment = self.control_name
        maxDiffSign = 0
        # 计算每个treatment组与control组的差异得到uplift score
        for treatment_group in currentNodeSummary:
            if treatment_group != self.control_name:
                diff = (currentNodeSummary[treatment_group][0]
                        - currentNodeSummary[self.control_name][0])
                if abs(diff) >= maxAbsDiff:
                    maxDiffTreatment = treatment_group
                    maxDiffSign = np.sign(diff)
                    maxAbsDiff = abs(diff)
                if diff >= maxDiff:
                    maxDiff = diff
                    suboptTreatment = treatment_group
                    if diff > 0:
                        bestTreatment = treatment_group
        if maxDiff > 0:
            pt = currentNodeSummary[bestTreatment][0]
            nt = currentNodeSummary[bestTreatment][1]
            pc = currentNodeSummary[self.control_name][0]
            nc = currentNodeSummary[self.control_name][1]
            p_value = (1. - stats.norm.cdf((pt - pc) / np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc))) * 2
        else:
            pt = currentNodeSummary[suboptTreatment][0]
            nt = currentNodeSummary[suboptTreatment][1]
            pc = currentNodeSummary[self.control_name][0]
            nc = currentNodeSummary[self.control_name][1]
            p_value = (1. - stats.norm.cdf((pc - pt) / np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc))) * 2
        upliftScore = [maxDiff, p_value]

        bestGain = 0.0
        bestAttribute = None

        # last column is the result/target column, 2nd to the last is the treatment group
        columnCount = train_x.shape[1]
        assert 0 < self.max_features <= 1.0
        max_features = self.max_features * columnCount
        # 随机选取一些列（默认10个特征）进行树的构建、划分，这里RF采用放回采样
        for col in list(np.random.choice(a=range(columnCount), size=int(max_features), replace=False)):
            columnValues = train_x.iloc[:, col]
            # unique values
            lsUnique = np.unique(columnValues)

            if (isinstance(lsUnique[0], int) or
                    isinstance(lsUnique[0], float)):
                if len(lsUnique) > 10:  # 找到一组数的分位数值
                    lspercentile = np.percentile(columnValues, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                else:
                    lspercentile = np.percentile(lsUnique, [10, 50, 90])
                lsUnique = np.unique(lspercentile)
            # 统计特征的唯一值数量
            train_to_est_ratio = len(train_x) / len(est_x)
            for value in lsUnique:
                # 划分左右子树
                X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(train_x, train_t, train_y, col, value)
                val_X_l, val_X_r, val_w_l, val_w_r, val_y_l, val_y_r = self.divideSet(val_x, val_t, val_y, col, value)
                est_X_l, est_X_r, est_y_l, est_y_r, est_t_l, est_t_r = self.divideSet(est_x, est_y, est_t, col, value)

                # check the split validity on min_samples_leaf  372
                if len(X_l) < min_samples_leaf or len(X_r) < min_samples_leaf:
                    continue
                if len(est_X_l) < min_samples_leaf or len(est_X_r) < min_samples_leaf:
                    continue
                # summarize notes
                # Gain -- Entropy or Gini
                # 评分值 = 左、右子树样本数的比例
                leftNodeSummary = self.tree_node_summary(w_l, y_l,
                                                         min_samples_treatment=min_samples_treatment,
                                                         n_reg=n_reg,
                                                         parentNodeSummary=currentNodeSummary)

                rightNodeSummary = self.tree_node_summary(w_r, y_r,
                                                          min_samples_treatment=min_samples_treatment,
                                                          n_reg=n_reg,
                                                          parentNodeSummary=currentNodeSummary)

                # check the split validity on min_samples_treatment
                if set(leftNodeSummary.keys()) != set(rightNodeSummary.keys()):
                    continue
                node_mst = 10 ** 8
                for ti in leftNodeSummary:
                    node_mst = np.min([node_mst, leftNodeSummary[ti][1]])
                    node_mst = np.min([node_mst, rightNodeSummary[ti][1]])
                # 保证每个treatment样本数充足
                if node_mst < min_samples_treatment:
                    continue
                # evaluate the split
                # 计算增益
                # 计算单调性
                if is_constrained:
                    K_L = increasing_constrain(leftNodeSummary)
                    K_R = increasing_constrain(rightNodeSummary)
                    if K_L == 0 or K_R == 0:
                        continue

                if (self.control_name in leftNodeSummary and
                        self.control_name in rightNodeSummary):
                    leftScore1, leftTrigger1 = self.evaluate_obj(w_l, y_l, self.B, self.honest, val_w_l, val_y_l)
                    rightScore2, rightTrigger2 = self.evaluate_obj(w_r, y_r, self.B, self.honest, val_w_r, val_y_r)
                    # ----------------------------------------------------------------
                    # Honest penalty
                    # ----------------------------------------------------------------
                    var_treat1, var_control1 = self.variance_trigger(y_l, w_l, self.B, leftTrigger1)
                    var_treat2, var_control2 = self.variance_trigger(y_r, w_r, self.B, rightTrigger2)
                    treated = w_l >= leftTrigger1
                    left_nt = w_l[treated].shape[0]
                    treated = w_r >= rightTrigger2
                    right_nt = w_r[treated].shape[0]

                    left_treated_share = left_nt / train_x.shape[0] if left_nt > 0 else 1.0
                    right_treated_share = right_nt / train_x.shape[0] if right_nt > 0 else 1.0

                    left_var = (1 + train_to_est_ratio) * (
                            (var_treat1 / left_treated_share) + (var_control1 / (1-left_treated_share)))
                    right_var = (1 + train_to_est_ratio) * (
                            (var_treat2 / right_treated_share) + (var_control2 / (1-right_treated_share)))

                    gain = (leftScore1 + rightScore2 - currentScore) - (left_var + right_var - currentVar)
                    if self.normalization:
                        norm_factor = self.normI(currentNodeSummary,
                                                 leftNodeSummary,
                                                 rightNodeSummary,
                                                 self.control_name,
                                                 alpha=0.9)
                    else:
                        norm_factor = 1
                    gain = gain / norm_factor
                else:
                    gain = 0
                if gain > bestGain and len(X_l) > min_samples_leaf and len(X_r) > min_samples_leaf:
                    bestGain = gain
                    bestAttribute = (col, value)
                    best_set_left = [X_l, w_l, y_l, val_X_l, val_w_l, val_y_l, est_X_l, est_t_l, est_y_l]
                    best_set_right = [X_r, w_r, y_r, val_X_r, val_w_r, val_y_r, est_X_r, est_t_r, est_y_r]

        dcY = {'impurity': '%.3f' % currentScore, 'samples': '%d' % len(train_x), 'group_size': ''}
        # Add treatment size
        for treatment_group in currentNodeSummary:
            dcY['group_size'] += ' ' + str(treatment_group) + ': ' + str(currentNodeSummary[treatment_group][1])
        dcY['upliftScore'] = [round(upliftScore[0], 8), round(upliftScore[1], 8)]
        dcY['matchScore'] = round(upliftScore[0], 8)

        # 打印树的分裂情况
        if bestAttribute and show_log:
            if depth == 1 and flag == -1:
                print("root node:", depth, bestAttribute[0], bestAttribute[1])
            else:
                if flag == 1:
                    print("left node:", depth, bestAttribute[0], bestAttribute[1])
                else:
                    print("right node:", depth, bestAttribute[0], bestAttribute[1])
        # 如果增益大于0,递归分裂
        if bestGain > 0 and depth < max_depth:
            print('best gain:', bestGain)
            print('trigger:', leftTrigger1, rightTrigger2)
            print(dcY)
            trueBranch = self._honest_fit(
                *best_set_left, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary, flag=1, is_constrained=self.is_constrained, show_log=show_log
            )
            falseBranch = self._honest_fit(
                *best_set_right, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary, flag=0, is_constrained=self.is_constrained, show_log=show_log
            )

            return DecisionTree(
                col=bestAttribute[0], value=bestAttribute[1],
                trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY,
                maxDiffTreatment=maxDiffTreatment, maxDiffSign=maxDiffSign,
                nodeSummary=currentNodeSummary,
                backupResults=self.uplift_classification_results(train_t, train_y),
                bestTreatment=bestTreatment, upliftScore=upliftScore
            )
        else:  # 增益小于等于0的话或者大于最大深度的话结束分裂
            return DecisionTree(
                    results=self.uplift_classification_results(train_t, train_y),
                    summary=dcY, maxDiffTreatment=maxDiffTreatment,
                    maxDiffSign=maxDiffSign, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )

    def classify(self, observations, tree, dataMissing=False):
        '''
        Classifies (prediction) the observations according to the tree.

        Args
        ----
        observations : list of list
            The internal data format for the training data (combining X, Y, treatment).

        dataMissing: boolean, optional (default = False)
            An indicator for if data are missing or not.

        Returns
        -------
        tree.results, tree.upliftScore :
            The results in the leaf node.
        '''
        def classifyWithoutMissingData(observations, tree):
            '''
            Classifies (prediction) the observations according to the tree, assuming without missing data.

            Args
            ----
            observations : list of list
                The internal data format for the training data (combining X, Y, treatment).

            Returns
            -------
            tree.results, tree.upliftScore :
                The results in the leaf node.
            '''
            if tree.results is not None:  # leaf
                return tree.results, tree.upliftScore
            else:  # 递归查找分裂的叶子节点
                v = observations.iloc[0, tree.col]
                if isinstance(v, int) or isinstance(v, float):
                    if v >= tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
                else:
                    if v == tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
            return classifyWithoutMissingData(observations, branch)

        def classifyWithMissingData(observations, tree):
            '''
            Classifies (prediction) the observations according to the tree, assuming with missing data.

            Args
            ----
            observations : list of list
                The internal data format for the training data (combining X, Y, treatment).

            Returns
            -------
            tree.results, tree.upliftScore :
                The results in the leaf node.
            '''
            if tree.results is not None:  # leaf
                return tree.results
            else:
                v = observations.iloc[0, tree.col]
                if v is None:
                    tr = classifyWithMissingData(observations, tree.trueBranch)
                    fr = classifyWithMissingData(observations, tree.falseBranch)
                    tcount = sum(tr.values())
                    fcount = sum(fr.values())
                    tw = float(tcount) / (tcount + fcount)
                    fw = float(fcount) / (tcount + fcount)

                    # Problem description: http://blog.ludovf.net/python-collections-defaultdict/
                    result = defaultdict(int)
                    for k, v in tr.items():
                        result[k] += v * tw
                    for k, v in fr.items():
                        result[k] += v * fw
                    return dict(result)
                else:
                    branch = None
                    if isinstance(v, int) or isinstance(v, float):
                        if v >= tree.value:
                            branch = tree.trueBranch
                        else:
                            branch = tree.falseBranch
                    else:
                        if v == tree.value:
                            branch = tree.trueBranch
                        else:
                            branch = tree.falseBranch
                return classifyWithMissingData(observations, branch)

        # function body
        if dataMissing:
            return classifyWithMissingData(observations, tree)
        else:
            return classifyWithoutMissingData(observations, tree)


# Honest Random Forests
class HonestRandomForestClassifier:
    """ Honest Random Forest for Classification Task.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the uplift random forest.

    max_features: int, optional (default=10)
        The number of features to consider when looking for the best split.

    random_state: int, optional (default=2019)
        The seed used by the random number generator.

    max_depth: int, optional (default=5)
        The maximum depth of the tree.

    min_samples_leaf: int, optional (default=100)
        The minimum number of samples required to be split at a leaf node.

    min_samples_treatment: int, optional (default=10)
        The minimum number of samples required of the experiment group to be split at a leaf node.

    n_reg: int, optional (default=10)
        The regularization parameter defined in Rzepakowski et al. 2012, the
        weight (in terms of sample size) of the parent node influence on the
        child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.

    control_name: string
        The name of the control group (other experiment groups will be regarded as treatment groups)

    normalization: boolean, optional (default=True)
        The normalization factor defined in Rzepakowski et al. 2012,
        correcting for tests with large number of splits and imbalanced
        treatment and control splits

    Outputs
    ----------
    df_res: pandas dataframe
        A user-level results dataframe containing the estimated individual treatment effect.
    """

    # https://causalml.readthedocs.io/en/latest/methodology.html#uplift-tree
    def __init__(self,
                 seed=0,
                 n_estimators=20,
                 max_features=0.5,
                 random_state=2020,
                 max_depth=5,
                 min_samples_leaf=100,
                 min_samples_treatment=10,
                 n_reg=10,
                 control_name=0,
                 normalization=False,
                 is_constrained=False,
                 is_continuous_method=False,
                 show_log=False,
                 n_jobs=None,
                 B=0,
                 honest=False,
                 weight=0.5,
                 is_prune=False):
        """
        Initialize the HonestRandomForestClassifier class.
        """
        self.classes_ = {}
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.control_name = control_name
        self.is_constrained = is_constrained
        self.show_log = show_log
        self.n_jobs = n_jobs
        self.B = B
        self.honest = honest
        self.weight = weight
        self.is_prune = is_prune
        self.seed = seed
        self.aucc_score = None
        self.qini_score = None
        # Create forest
        # 创建n个随机森林分类器
        self.uplift_forest = []

        para_dict = {
            'max_depth': [self.max_depth],
            'max_features': [self.max_features],
            'min_samples_leaf': [self.min_samples_leaf],
            'min_samples_treatment': [self.min_samples_treatment],
            'n_reg': [self.n_reg],
        }
        check_parameters(para_dict)

        for _ in range(n_estimators):
            uplift_tree = HonestTreeClassifier(
                seed=self.seed,
                max_features=self.max_features, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_treatment=self.min_samples_treatment,
                n_reg=self.n_reg,
                control_name=self.control_name,
                normalization=normalization,
                is_constrained=self.is_constrained,
                is_continuous_method=is_continuous_method,
                show_log=self.show_log,
                B=self.B,
                honest=self.honest,
                weight=self.weight)

            self.uplift_forest.append(uplift_tree)

    def fit(self, X, treatment, y):
        """
        Fit the UpliftRandomForestClassifier.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.

        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        """
        # 数据类型转换
        # X = df2array(X)
        # treatment = df2array(treatment)
        # y = df2array(y)
        # 随机种子
        np.random.seed(self.random_state)
        # Get treatment group keys
        # 获得所有的treatment组的名字
        treatment_group_keys = list(set(treatment))
        treatment_group_keys.remove(self.control_name)
        treatment_group_keys.sort()
        self.classes_ = {}
        for i, treatment_group_key in enumerate(treatment_group_keys):
            self.classes_[treatment_group_key] = i

        # 随机森林多棵树并行计算
        if not self.n_jobs or self.n_jobs <= 1:
            for tree_i in range(len(self.uplift_forest)):
                bt_index = np.random.choice(len(X), len(X))
                x_train_bt = X.iloc[bt_index,:]
                y_train_bt = y[bt_index]
                treatment_train_bt = treatment[bt_index]
                self.uplift_forest[tree_i].fitted_uplift_tree = self.uplift_forest[tree_i].fit(x_train_bt, treatment_train_bt, y_train_bt)
                if self.is_prune:
                    self.uplift_forest[tree_i].fitted_uplift_tree = self.uplift_forest[tree_i].prune(x_train_bt, treatment_train_bt, y_train_bt).fitted_uplift_tree
        else:
            from joblib import Parallel, delayed, parallel_backend
            tasks = []
            if self.n_jobs >= 32:
                self.n_jobs = 32
            for tree_i in range(len(self.uplift_forest)):
                bt_index = np.random.choice(len(X), len(X))
                x_train_bt = X.iloc[bt_index,:]
                y_train_bt = y[bt_index]
                treatment_train_bt = treatment[bt_index]
                tree = self.uplift_forest[tree_i]
                tasks.append(delayed(self.multiple_thread_rf)(tree_i, tree, x_train_bt, treatment_train_bt, y_train_bt))
            with parallel_backend("loky", n_jobs=self.n_jobs):
                for result in Parallel(prefer="processes", n_jobs=self.n_jobs, pre_dispatch='1 * n_jobs')(tasks):
                    self.uplift_forest[result[0]].fitted_uplift_tree = result[1]
                    self.uplift_forest[result[0]].cnt_prune = result[2]

    def multiple_thread_rf(self, tree_i=None, tree=None, X=None, treatment=None, y=None):
        fitted_tree = tree.fit(X, treatment, y)
        cnt_prune = 0
        if self.is_prune:
            fitted_tree = tree.prune(X, treatment, y).fitted_uplift_tree
            cnt_prune = self.uplift_forest[tree_i].cnt_prune
        return tree_i, fitted_tree, cnt_prune

    def predict(self, X, n_jobs=None):
        '''
        Returns the recommended treatment group and predicted optimal
        probability conditional on using the recommended treatment group.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        full_output : bool, optional (default=False)
            Whether the UpliftTree algorithm returns upliftScores, pred_nodes
            alongside the recommended treatment group and p_hat in the treatment group.

        Returns
        -------
        df_res : DataFrame, shape = [num_samples, (num_treatments + 1)]
            A DataFrame containing the predicted delta in each treatment group,
            the best treatment group and the maximum delta.
            :param n_jobs:

        '''
        # 数据格式转换
        # X = df2array(X)

        df_res = pd.DataFrame()
        y_pred_ensemble = dict()
        y_pred_list = np.zeros((X.shape[0], len(self.classes_)))
        tree_list = []

        # Make prediction by each tree
        y_pred_full_dict = {}
        if n_jobs and n_jobs > 1:
            """
            多线程预测
            """
            from joblib import Parallel, delayed, parallel_backend
            tasks = []
            n_jobs = min(n_jobs, 32)

            for tree_i in range(len(self.uplift_forest)):
                tasks.append(delayed(self.multiple_thread_predict)(tree_i, X))

            with parallel_backend("loky", n_jobs=-1):
                for result in Parallel(prefer="processes", n_jobs=n_jobs, pre_dispatch='1 * n_jobs')(tasks):
                    y_pred_full_dict[result[0]] = result[1]

        for tree_i in range(len(self.uplift_forest)):  # 每棵树的结果进行加和得到最后的
            tree_list.append(self.uplift_forest[tree_i].fitted_uplift_tree)
            if len(y_pred_full_dict) > 0:
                y_pred_full = y_pred_full_dict[tree_i]
            else:
                _, _, _, y_pred_full = self.uplift_forest[tree_i].predict(X=X, full_output=True)

            if tree_i == 0:
                for treatment_group in y_pred_full:
                    y_pred_ensemble[treatment_group] = (
                            np.array(y_pred_full[treatment_group]) / len(self.uplift_forest)
                    )
            else:
                for treatment_group in y_pred_full:
                    y_pred_ensemble[treatment_group] = (
                            np.array(y_pred_ensemble[treatment_group])
                            + np.array(y_pred_full[treatment_group]) / len(self.uplift_forest)
                    )

        # Summarize results into dataframe
        for treatment_group in y_pred_ensemble:
            df_res[treatment_group] = y_pred_ensemble[treatment_group]

        df_res['recommended_treatment'] = df_res.apply(np.argmax, axis=1)

        # Calculate delta
        delta_cols = []
        for treatment_group in y_pred_ensemble:
            if treatment_group != self.control_name:
                delta_cols.append('delta_%s' % (treatment_group))
                df_res['delta_%s' % (treatment_group)] = df_res[treatment_group] - df_res[self.control_name]
                # Add deltas to results list
                y_pred_list[:, self.classes_[treatment_group]] = df_res['delta_%s' % (treatment_group)].values
        df_res['max_delta'] = df_res[delta_cols].max(axis=1)
        return y_pred_list

    def multiple_thread_predict(self, tree_i=None, X=None):
        _, _, _, y_pred_full = self.uplift_forest[tree_i].predict(X=X, full_output=True)
        return tree_i, y_pred_full


def test():
    raw_train = pd.read_csv('/Users/didi/Desktop/data_train_1030.csv')
    raw_test = pd.read_csv('/Users/didi/Desktop/data_test_1030.csv')
    print("数据集比例：", len(raw_train), len(raw_test))
    # 训练数据
    # raw_train = raw_train[(raw_train['group_type'] == 'treatment_0') | (raw_train['group_type'] == 'control')].reset_index(drop=True)
    # raw_test = raw_test[(raw_test['group_type'] == 'treatment_0') | (raw_test['group_type'] == 'control')].reset_index(drop=True)
    y_train = raw_train['label']
    cost_train = raw_train['cost']
    treatment_train = raw_train['group_type']
    # 删除与特征无关的列(此处根据情况适当修改代码)
    x_train = raw_train.drop(columns=['cost', 'group_type', 'label', 'date'])
    # 测试数据
    y_test = raw_test['label']
    cost_test = raw_test['cost']
    treatment_test = raw_test['group_type']
    # 删除与特征无关的列
    x_test = raw_test.drop(columns=['cost', 'group_type', 'label', 'date'])
    # treatment组
    t_groups = list(set(list(treatment_train.value_counts().keys())) - {'control'})
    t_groups.sort(reverse=False)

    treatment_unique = list(set(treatment_train) - {"control"})
    treatment_unique.sort(reverse=False)
    treat_dict = dict()
    treat_dict['control'] = 0
    for i in range(len(treatment_unique)):
        treat_dict[treatment_unique[i]] = i + 1
    t_train = treatment_train.apply(lambda x: treat_dict[x])
    n = 5
    HTC = HonestRandomForestClassifier(n_estimators=n, B=0, max_depth=4, max_features=0.5, min_samples_leaf=50,
                                       min_samples_treatment=10, n_jobs=32,is_prune=True)
    HTC.fit(x_train, t_train, y_train)
    pred_train = HTC.predict(x_train)
    pred = HTC.predict(x_test)
    cnt = []
    for i in range(n):
        cnt.append(HTC.uplift_forest[i].cnt_prune)
    print('max prune nodes:', max(cnt))
    print('min prune nodes:', min(cnt))
    print('avg prune nodes:', np.mean(cnt))
    # pd.DataFrame(pred_train).to_csv('/Users/didi/Desktop/pred_train.csv')
    # pd.DataFrame(pred).to_csv('/Users/didi/Desktop/pred_test.csv')


if __name__ == "__main__":
    test()
