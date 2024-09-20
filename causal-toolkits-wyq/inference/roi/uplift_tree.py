from __future__ import print_function
from collections import defaultdict
import numpy as np
import scipy.stats as stats


class UpliftTree:
    """ Tree Node Class

    Tree node class to contain all the statistics of the tree node.

    Parameters
    ----------

    col : int, optional (default = -1)
        The column index for splitting the tree node to children nodes.

    value : float, optional (default = None)
        The value of the feature column to split the tree node to children nodes.

    trueBranch : object of UpliftTree
        The true branch tree node (feature > value).

    falseBranch : object of UpliftTree
        The flase branch tree node (feature > value).

    results : dictionary
        The classification probability Pr(1) for each experiment group in the tree node.

    summary : dictionary
        Summary statistics of the tree nodes, including impurity, sample size, uplift score, etc.

    maxDiffTreatment : string
        The treatment name generating the maximum difference between treatment and control group.

    maxDiffSign : float
        The sign of the maxium difference (1. or -1.).

    nodeSummary : dictionary
        Summary statistics of the tree nodes {treatment: [y_mean, n]}, where y_mean stands for the target metric mean
        and n is the sample size.

    backupResults : dictionary
        The conversion proabilities in each treatment in the parent node {treatment: y_mean}. The parent node
        information is served as a backup for the children node, in case no valid statistics can be calculated from the
        children node, the parent node information will be used in certain cases.

    bestTreatment : string
        The treatment name providing the best uplift (treatment effect).

    upliftScore : list
        The uplift score of this node: [max_Diff, p_value], where max_Diff stands for the maxium treatment effect, and
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


# Uplift Tree Classifier
class UpliftTreeRegressor:
    """ Uplift Tree Classifier for Classification Task.

    A uplift tree classifier estimates the individual treatment effect by modifying the loss function in the
    classification trees.

    The uplift tree classifer is used in uplift random forest to construct the trees in the forest.

    Parameters
    ----------

    evaluationFunction : string
        Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS'.

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
                 min_samples_treatment=10, n_reg=100, evaluationFunction='KL',
                 control_name=None, normalization=False, x_names=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.max_features = max_features
        self.x_names = x_names

        assert evaluationFunction in ["DDP", "KL", "ED", "Chi", "CTS", "Roi",
                                      "Net"], 'evaluation function must be in ["DDP", "KL", "ED", "Chi", "CTS", "Roi", "Net"] '

        if evaluationFunction == 'KL':
            self.evaluationFunction = self.evaluate_KL
        elif evaluationFunction == 'ED':
            self.evaluationFunction = self.evaluate_ED
        elif evaluationFunction == 'Chi':
            self.evaluationFunction = self.evaluate_Chi
        elif evaluationFunction == 'CTS':
            self.evaluationFunction = self.evaluate_CTS
        elif evaluationFunction == 'Roi':
            self.evaluationFunction = self.evaluate_Roi
        elif evaluationFunction == 'Net':
            self.evaluationFunction = self.evaluate_Net
        elif evaluationFunction == 'DDP':
            self.evaluationFunction = self.evaluate_DDP
        else:
            self.evaluationFunction = self.evaluate_Roi
        self.fitted_uplift_tree = None
        self.control_name = control_name
        self.normalization = normalization

    def fit(self, X, treatment, y, c):
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

        print("hello world ~")
        rows = [list(X[i]) + [treatment[i]] + [y[i]] + [c[i]] for i in range(len(X))]
        resTree = self.growDecisionTreeFrom(
            rows, evaluationFunction=self.evaluationFunction,
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            depth=1, min_samples_treatment=self.min_samples_treatment,
            n_reg=self.n_reg, parentNodeSummary=None
        )
        self.fitted_uplift_tree = resTree
        return self

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

        rows = [list(X[i]) + [treatment[i]] + [y[i]] for i in range(len(X))]
        self.pruneTree(rows,
                       tree=self.fitted_uplift_tree,
                       rule=rule,
                       minGain=minGain,
                       evaluationFunction=self.evaluationFunction,
                       notify=False,
                       n_reg=self.n_reg,
                       parentNodeSummary=None)
        return self

    def pruneTree(self, rows, tree, rule='maxAbsDiff', minGain=0.,
                  evaluationFunction=None, notify=False, n_reg=0,
                  parentNodeSummary=None):
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
            rows, min_samples_treatment=self.min_samples_treatment,
            n_reg=n_reg, parentNodeSummary=parentNodeSummary
        )
        tree.nodeSummary = currentNodeSummary
        # Divide sets for child nodes
        (set1, set2) = self.divideSet(rows, tree.col, tree.value)

        # recursive call for each branch
        if tree.trueBranch.results is None:
            self.pruneTree(set1, tree.trueBranch, rule, minGain,
                           evaluationFunction, notify, n_reg,
                           parentNodeSummary=currentNodeSummary)
        if tree.falseBranch.results is None:
            self.pruneTree(set2, tree.falseBranch, rule, minGain,
                           evaluationFunction, notify, n_reg,
                           parentNodeSummary=currentNodeSummary)

        # merge leaves (potentionally)
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
                    set1, min_samples_treatment=self.min_samples_treatment,
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
                    set2, min_samples_treatment=self.min_samples_treatment,
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
                    set1, min_samples_treatment=self.min_samples_treatment,
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
                    set2, min_samples_treatment=self.min_samples_treatment,
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
                gain = ((1. * len(set1) / len(rows) * trueScoreD
                         + 1. * len(set2) / len(rows) * falseScoreD)
                        - currentScoreD)
                if gain <= minGain or (trueScoreD + falseScoreD < 0.):
                    tree.trueBranch, tree.falseBranch = None, None
                    tree.results = tree.backupResults
        return self

    def fill(self, X, treatment, y):
        """ Fill the data into an existing tree.

        This is a higher-level function to transform the original data inputs
        into lower level data inputs (list of list and tree).

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

        rows = [list(X[i]) + [treatment[i]] + [y[i]] for i in range(len(X))]
        self.fillTree(rows, tree=self.fitted_uplift_tree)
        return self

    def fillTree(self, rows, tree):
        """ Fill the data into an existing tree.

        This is a lower-level function to execute on the tree filling task.

        Args
        ----
        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).

        tree : object
            object of DecisionTree class

        Returns
        -------
        self : object
        """
        # Current Node Summary for Validation Data Set
        currentNodeSummary = self.tree_node_summary(rows, min_samples_treatment=0, n_reg=0, parentNodeSummary=None)
        tree.nodeSummary = currentNodeSummary
        # Divide sets for child nodes
        (set1, set2) = self.divideSet(rows, tree.col, tree.value)

        # recursive call for each branch
        if tree.trueBranch is not None:
            self.fillTree(set1, tree.trueBranch)
        if tree.falseBranch is not None:
            self.fillTree(set2, tree.falseBranch)

        # Update Information

        # matchScore
        matchScore = (currentNodeSummary[tree.bestTreatment][0] - currentNodeSummary[self.control_name][0])
        tree.matchScore = round(matchScore, 4)
        tree.summary['matchScore'] = round(matchScore, 4)

        # Samples, Group_size
        tree.summary['samples'] = len(rows)
        tree.summary['group_size'] = ''
        for treatment_group in currentNodeSummary:
            tree.summary['group_size'] += ' ' + treatment_group + ': ' + str(currentNodeSummary[treatment_group][1])
        # classProb
        if tree.results is not None:
            tree.results = self.uplift_classification_results(rows)
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
            pred_leaf, upliftScore = self.classify(X[xi], self.fitted_uplift_tree, dataMissing=False)
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

    def divideSet(self, rows, column, value):
        '''
        Tree node split.

        Args
        ----

        rows : list of list
               The internal data format.

        column : int
                The column used to split the data.

        value : float or int
                The value in the column for splitting the data.

        Returns
        -------
        (list1, list2) : list of list
                The left node (list of data) and the right node (list of data).
        '''
        splittingFunction = None

        # for int and float values
        if isinstance(value, int) or isinstance(value, float):
            splittingFunction = lambda row: row[column] >= value
        else:  # for strings
            splittingFunction = lambda row: row[column] == value
        list1 = [row for row in rows if splittingFunction(row)]
        list2 = [row for row in rows if not splittingFunction(row)]
        return (list1, list2)

    def group_uniqueCounts(self, rows):
        '''
        Count sample size by experiment group.

        Args
        ----

        rows : list of list
               The internal data format.

        Returns
        -------
        results : dictionary
                The control and treatment sample size.
        '''
        results = {}
        for row in rows:
            r = row[-3]
            y = row[-2]
            c = row[-1]
            if r not in results:
                results[r] = [0, 0, 0]

            results[r][0] += 1
            results[r][1] += y
            results[r][2] += c

        return results

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
        S = pk * np.log(pk / qk) + (1 - pk) * np.log((1 - pk) / (1 - qk))
        return S

    def evaluate_KL(self, nodeSummary, control_name):
        '''
        Calculate KL Divergence as split evaluation criterion for a given node.

        Args
        ----

        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary()
            method.

        control_name : string
            The control group name.

        Returns
        -------
        d_res : KL Divergence
        '''
        if control_name not in nodeSummary:
            return 0
        c_distribution = nodeSummary[control_name][6] / float(nodeSummary[control_name][1])
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_name:
                t_distribution = nodeSummary[treatment_group][6] / float(nodeSummary[treatment_group][1])
                for i in range(len(t_distribution)):
                    t = t_distribution[i] if t_distribution[i] > 0 else 0.1 ** 6
                    c = c_distribution[i] if c_distribution[i] > 0 else 0.1 ** 6
                    d_res += t * np.log(t / c)
        return d_res

    @staticmethod
    def evaluate_ED(nodeSummary, control_name):
        '''
        Calculate Euclidean Distance as split evaluation criterion for a given node.

        Args
        ----

        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary()
            method.

        control_name : string
            The control group name.

        Returns
        -------
        d_res : Euclidean Distance
        '''
        if control_name not in nodeSummary:
            return 0
        c_distribution = nodeSummary[control_name][6]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_name:
                t_distribution = nodeSummary[control_name][6]
                for i in range(len(t_distribution)):
                    t = t_distribution[i] if t_distribution[i] > 0 else 0.1 ** 6
                    c = c_distribution[i] if c_distribution[i] > 0 else 0.1 ** 6
                    d_res += (t - c) ** 2
        return d_res

    @staticmethod
    def evaluate_Chi(nodeSummary, control_name):
        '''
        Calculate Chi-Square statistic as split evaluation criterion for a given node.

        Args
        ----

        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary() method.

        control_name : string
            The control group name.

        Returns
        -------
        d_res : Chi-Square
        '''
        if control_name not in nodeSummary:
            return 0
        c_distribution = nodeSummary[control_name][6]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_name:
                t_distribution = nodeSummary[control_name][6]
                for i in range(len(t_distribution)):
                    t = t_distribution[i] if t_distribution[i] > 0 else 0.1 ** 6
                    c = c_distribution[i] if c_distribution[i] > 0 else 0.1 ** 6
                    d_res += (t - c) ** 2 / t
        return d_res

    @staticmethod
    def evaluate_CTS(currentNodeSummary):
        '''
        Calculate CTS (conditional treatment selection) as split evaluation criterion for a given node.

        Args
        ----

        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary() method.

        control_name : string
            The control group name.

        Returns
        -------
        d_res : Chi-Square
        '''
        mu = 0.0
        # iterate treatment group
        for r in currentNodeSummary:
            mu = max(mu, currentNodeSummary[r][0])
        return -mu

    @staticmethod
    def evaluate_Roi(nodeSummary, control_name):

        if control_name not in nodeSummary:
            return 0
        c_gmv = float(nodeSummary[control_name][0])
        c_cost = float(nodeSummary[control_name][2])

        t_gmv_total = 0
        t_cost_total = 0
        t_pas_total = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_name:
                t_gmv_total += nodeSummary[treatment_group][3]
                t_cost_total += nodeSummary[treatment_group][4]
                t_pas_total += nodeSummary[treatment_group][1]

        t_gmv = float(t_gmv_total / t_pas_total)
        t_cost = float(t_cost_total / t_pas_total)

        if t_cost - c_cost <= 0:
            return 0.0

        d_res = (t_gmv - c_gmv) / (t_cost - c_cost)
        return d_res

    @staticmethod
    def evaluate_Net(nodeSummary, control_name):

        alpha = 1.0
        if control_name not in nodeSummary:
            return 0
        c_gmv = float(nodeSummary[control_name][0])
        c_cost = float(nodeSummary[control_name][2])

        t_gmv_total = 0
        t_cost_total = 0
        t_pas_total = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_name:
                t_gmv_total += nodeSummary[treatment_group][3]
                t_cost_total += nodeSummary[treatment_group][4]
                t_pas_total += nodeSummary[treatment_group][1]

        t_gmv = float(t_gmv_total / t_pas_total)
        t_cost = float(t_cost_total / t_pas_total)

        d_res = (t_gmv - c_gmv) - alpha * (t_cost - c_cost)
        return d_res

    @staticmethod
    def evaluate_DDP(nodeSummary, control_name):

        alpha = 1.0
        if control_name not in nodeSummary:
            return 0
        c_gmv = float(nodeSummary[control_name][0])

        t_gmv_total = 0
        t_pas_total = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_name:
                t_gmv_total += nodeSummary[treatment_group][3]
                t_pas_total += nodeSummary[treatment_group][1]

        t_gmv = float(t_gmv_total / t_pas_total)

        d_res = (t_gmv - c_gmv) / c_gmv
        return d_res

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

    def tree_node_summary(self, rows, min_samples_treatment=10, n_reg=100, parentNodeSummary=None):
        '''
        Tree node summary statistics.

        Args
        ----

        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).

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

        yValues = [row[-2] for row in rows]
        lspercentile = np.percentile(yValues, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        lsUnique = list(set(lspercentile))
        lsUnique.sort()
        if lsUnique[0] == 0:
            lsUnique[0] = 0.00001

        nodeSummary = {}
        for row in rows:
            r = row[-3]
            y = row[-2]
            c = row[-1]
            if r not in nodeSummary:
                nodeSummary[r] = [0, 0, 0, 0, [0] * (len(lsUnique) + 1)]

            nodeSummary[r][0] += 1
            nodeSummary[r][1] += y
            nodeSummary[r][2] += c
            nodeSummary[r][3] += y * y

            bucket_index = len(lsUnique)
            for i, val in enumerate(lsUnique):
                if y < val:
                    bucket_index = i
                    break
            nodeSummary[r][4][bucket_index] += 1

        res = {}
        for r in nodeSummary:
            sample = nodeSummary[r][0]
            gmv = nodeSummary[r][1]
            cost = nodeSummary[r][2]
            gmv_square = nodeSummary[r][3]
            distribution = nodeSummary[r][4]
            mean_gmv = gmv / sample
            mean_cost = cost / sample
            mean_gmv_square = gmv_square / sample
            sigma_gmv = mean_gmv_square - mean_gmv ** 2
            res[r] = [mean_gmv, sample, mean_cost, gmv, cost, sigma_gmv, distribution]
        return res

    def uplift_classification_results(self, rows):

        '''
        Classification probability for each treatment in the tree node.

        Args
        ----

        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).

        Returns
        -------
        res : dictionary
            The probability of 1 in each treatment in the tree node.
        '''

        results = self.group_uniqueCounts(rows)
        res = {}
        for r in results:
            if r == self.control_name:
                res[r] = 0.0
            else:
                t_gmv = float(results[r][0])
                t_cost = float(results[r][2])
                c_gmv = float(results[self.control_name][0])
                c_cost = float(results[self.control_name][2])
                roi = (t_gmv - c_gmv) / (t_cost - c_cost)
                res[r] = round(roi, 6)

        return res

    def growDecisionTreeFrom(self, rows, evaluationFunction, max_depth=10,
                             min_samples_leaf=100, depth=1,
                             min_samples_treatment=10, n_reg=100,
                             parentNodeSummary=None):
        '''
        Train the uplift decision tree.

        Args
        ----

        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).

        evaluationFunction : string
            Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS'.

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

        if len(rows) == 0:
            return UpliftTree()

        # Current Node Info and Summary
        currentNodeSummary = self.tree_node_summary(
            rows, min_samples_treatment=min_samples_treatment, n_reg=n_reg, parentNodeSummary=parentNodeSummary
        )
        if evaluationFunction == self.evaluate_CTS:
            currentScore = evaluationFunction(currentNodeSummary)
        else:
            currentScore = evaluationFunction(currentNodeSummary, control_name=self.control_name)

        # Prune Stats
        maxAbsDiff = 0
        maxDiff = -10000000.
        bestTreatment = self.control_name
        suboptTreatment = self.control_name
        maxDiffTreatment = self.control_name
        maxDiffSign = 0
        for treatment_group in currentNodeSummary:
            if treatment_group != self.control_name:
                diff = currentNodeSummary[treatment_group][0] - currentNodeSummary[self.control_name][0]
                if abs(diff) >= maxAbsDiff:
                    maxDiffTreatment = treatment_group
                    maxDiffSign = np.sign(diff)
                    maxAbsDiff = abs(diff)
                if diff >= maxDiff:
                    maxDiff = diff
                    suboptTreatment = treatment_group
                    if diff > 0:
                        bestTreatment = treatment_group

        sigma_t = currentNodeSummary[bestTreatment][5]
        n_t = currentNodeSummary[bestTreatment][1]
        sigma_c = currentNodeSummary[self.control_name][5]
        n_c = currentNodeSummary[self.control_name][1]
        p_value = 1.96 * np.sqrt(sigma_t / n_t + sigma_c / n_c)

        upliftScore = [maxDiff, p_value]

        bestGain = 0.0
        bestAttribute = None
        bestSets = None

        # last column is the cost, 2nd to the last is the treatment group, 3rd to the last is the result/target column
        columnCount = len(rows[0]) - 3
        if (self.max_features and self.max_features > 0 and self.max_features <= columnCount):
            max_features = self.max_features
        else:
            max_features = columnCount

        randomCols = list(np.random.choice(a=range(columnCount), size=max_features, replace=False))
        randomCols.sort()
        print("第", depth, "层采样的随机特征: ", list(map(lambda x: self.x_names[x], randomCols)))
        for col in randomCols:
            columnValues = [row[col] for row in rows]
            # unique values
            lsUnique = list(set(columnValues))

            if (isinstance(lsUnique[0], int) or
                    isinstance(lsUnique[0], float)):
                if len(lsUnique) > 10:
                    lspercentile = np.percentile(columnValues, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                else:
                    lspercentile = np.percentile(lsUnique, [10, 50, 90])
                lsUnique = list(set(lspercentile))

            lsUnique.sort()
            print("\n第", depth, "层第", col, "个【特征】", self.x_names[col], " : ", lsUnique)
            for value in lsUnique:
                (set1, set2) = self.divideSet(rows, col, value)
                # check the split validity on min_samples_leaf  372
                if (len(set1) < min_samples_leaf or len(set2) < min_samples_leaf):
                    continue
                # summarize notes
                # Gain -- Entropy or Gini
                p = float(len(set1)) / len(rows)
                leftNodeSummary = self.tree_node_summary(
                    set1, min_samples_treatment=min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )

                rightNodeSummary = self.tree_node_summary(
                    set2, min_samples_treatment=min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )
                # check the split validity on min_samples_treatment
                if set(leftNodeSummary.keys()) != set(rightNodeSummary.keys()):
                    continue
                node_mst = 10 ** 8
                for ti in leftNodeSummary:
                    node_mst = np.min([node_mst, leftNodeSummary[ti][1]])
                    node_mst = np.min([node_mst, rightNodeSummary[ti][1]])
                if node_mst < min_samples_treatment:
                    continue
                # evaluate the split

                if evaluationFunction == self.evaluate_CTS:
                    leftScore1 = evaluationFunction(leftNodeSummary)
                    rightScore2 = evaluationFunction(rightNodeSummary)
                    gain = (currentScore - p * leftScore1 - (1 - p) * rightScore2)
                else:
                    leftScore1 = evaluationFunction(leftNodeSummary, control_name=self.control_name)
                    rightScore2 = evaluationFunction(rightNodeSummary, control_name=self.control_name)
                    gain = np.abs(leftScore1 - rightScore2)
                    # gain = (p * leftScore1 + (1 - p) * rightScore2 - currentScore)
                    if self.normalization:
                        norm_factor = self.normI(currentNodeSummary,
                                                 leftNodeSummary,
                                                 rightNodeSummary,
                                                 self.control_name,
                                                 alpha=0.9)
                    else:
                        norm_factor = 1
                    gain = gain / norm_factor
                    print(
                        "\n%s:%s,left_score:%.3f,right_score:%.3f,current_score:%.3f,gain:%.3f,best_gain:%.3f,best_attribute:%s" % (
                            self.x_names[col], value, leftScore1, rightScore2, currentScore, gain, bestGain,
                            bestAttribute))
                    print(" - curr  node summary: treatment->", currentNodeSummary['treatment'], " control->",
                          currentNodeSummary['control'])
                    print(" - left  node summary: treatment->", leftNodeSummary['treatment'], " control->",
                          leftNodeSummary['control'])
                    print(" - right node summary: treatment->", rightNodeSummary['treatment'], " control->",
                          rightNodeSummary['control'])

                if (gain > bestGain and len(set1) > min_samples_leaf and
                        len(set2) > min_samples_leaf):
                    bestGain = gain
                    bestAttribute = (col, value)
                    bestSets = (set1, set2)

        dcY = {'impurity': '%.3f' % currentScore, 'samples': '%d' % len(rows)}
        # Add treatment size
        dcY['group_size'] = ''
        for treatment_group in currentNodeSummary:
            dcY['group_size'] += ' ' + treatment_group + ': ' + str(currentNodeSummary[treatment_group][1])
        dcY['upliftScore'] = [round(upliftScore[0], 4), round(upliftScore[1], 4)]
        dcY['matchScore'] = round(upliftScore[0], 4)

        if bestGain > 0 and depth < max_depth:
            trueBranch = self.growDecisionTreeFrom(
                bestSets[0], evaluationFunction, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )
            falseBranch = self.growDecisionTreeFrom(
                bestSets[1], evaluationFunction, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )

            return UpliftTree(
                col=bestAttribute[0], value=bestAttribute[1],
                trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY,
                maxDiffTreatment=maxDiffTreatment, maxDiffSign=maxDiffSign,
                nodeSummary=currentNodeSummary,
                backupResults=self.uplift_classification_results(rows),
                bestTreatment=bestTreatment, upliftScore=upliftScore
            )
        else:
            if evaluationFunction == self.evaluate_CTS:
                return UpliftTree(
                    results=self.uplift_classification_results(rows),
                    summary=dcY, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )
            else:
                print("\n************ results: ", self.uplift_classification_results(rows))
                return UpliftTree(
                    results=self.uplift_classification_results(rows),
                    summary=dcY, maxDiffTreatment=maxDiffTreatment,
                    maxDiffSign=maxDiffSign, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )

    def classify(self, observations, tree, dataMissing=False):
        '''
        Classifies (prediction) the observationss according to the tree.

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
            Classifies (prediction) the observationss according to the tree, assuming without missing data.

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
            else:
                v = observations[tree.col]
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
            return classifyWithoutMissingData(observations, branch)

        def classifyWithMissingData(observations, tree):
            '''
            Classifies (prediction) the observationss according to the tree, assuming with missing data.

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
                v = observations[tree.col]
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
