"""
Describe :
Time :
Author: liwuzhuang@didiglobal.com
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import pydotplus


class TreeNode:
    def __init__(self, value=None, branch_left=None, branch_right=None, summary=None, gain=0):
        self.value = value
        self.branch_left = branch_left
        self.branch_right = branch_right
        self.summary = summary
        self.gain = gain


class PeopleUpliftTree:
    def __init__(self, max_depth=6, min_samples_leaf=40000, function_score="default", function_gain="default"):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.node = None
        self.feature_importance = None
        self.idx = 0

        if function_score == "default":
            self.get_score = self.function_score_default
        elif function_score == "user-defined":
            self.get_score = self.function_score_user_defined

        if function_gain == "default":
            self.get_gain = self.function_gain_default
        elif function_gain == "user-defined":
            self.get_gain = self.function_gain_user_defined

    def fit(self, x, treatment, y, cost=None):
        assert len(x) == len(y) and len(x) == len(treatment), 'Data length must be equal for X, treatment, and y.'
        # assert set(treatment) == {0, 1}, '0 means control name, 1 means treatment, they must be 0 and 1.'

        self.node = self.grow_tree(x, treatment, y, cost, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, depth=1)
        self.feature_importance = self.total_gain(self.node)

    def grow_tree(self, x, treatment, y, cost=None, max_depth=10, min_samples_leaf=100, depth=1):

        if len(x) == 0:
            return TreeNode()

        # 节点的数据统计
        score_current = self.get_score(treatment, y, cost)
        summary = {'n_samples': len(x),
                   'n_samples_treatment': (treatment == 1).sum(),
                   'n_samples_control': (treatment == 0).sum(),
                   'score': score_current
                   }

        # 获取特征的分割点
        values_unique = list(set(x))
        if isinstance(values_unique[0], int) or isinstance(values_unique[0], float):
            if len(values_unique) > 10:
                values_percentile = np.percentile(x, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                values_unique = np.unique(values_percentile)

        # 寻找最优分割点
        gain_best = 0.0
        p = 0.0
        value_best = None
        data_best_left = [[], [], [], []]
        data_best_right = [[], [], [], []]

        for value_split in values_unique:
            x_left, x_right, w_left, w_right, y_left, y_right, cost_left, cost_right = self.split_data(x, treatment, y, value_split, cost)

            if len(x_left) < min_samples_leaf or len(x_right) < min_samples_leaf:
                continue
            if set(w_left) != set(w_right):
                continue

            p = len(y_left)/len(y)
            score_left = p * self.get_score(w_left, y_left, cost_left)
            score_right = (1-p) * self.get_score(w_right, y_right, cost_right)
            gain = self.get_gain(score_left, score_right, score_current)

            if gain > gain_best:
                gain_best = gain
                value_best = value_split
                data_best_left = [x_left, w_left, y_left, cost_left]
                data_best_right = [x_right, w_right, y_right, cost_right]

        if gain_best > 0 and depth < max_depth:
            self.idx += 1
            branch_left = self.grow_tree(*data_best_left, max_depth, min_samples_leaf, depth + 1)
            branch_right = self.grow_tree(*data_best_right, max_depth, min_samples_leaf, depth + 1)
            return TreeNode(value=value_best, branch_left=branch_left, branch_right=branch_right, summary=summary, gain=gain_best)
        else:
            return TreeNode(summary=summary)

    @staticmethod
    def function_score_default(treatment, y, cost=None):
        g_t = treatment == 1
        g_c = treatment == 0

        if cost is not None:
            cost_delta = cost[g_t].mean() - cost[g_c].mean()
            y_delta = y[g_t].mean() - y[g_c].mean()
            score = y_delta / cost_delta
        else:
            y_delta = y[g_t].mean() - y[g_c].mean()
            score = y_delta

        return score

    @staticmethod
    def function_score_user_defined(w, y, cost=None):
        return ((np.mean(w*y)-np.mean(w)*np.mean(y))/(np.mean(w*w)-(np.mean(w))**2))

    @staticmethod
    def function_gain_default(score_left, score_right, score_parent):
        return max(score_left, score_right) - score_parent

    @staticmethod
    def function_gain_user_defined(score_left, score_right, score_parent=None):
        return (score_left + score_right - score_parent)

    @staticmethod
    def split_data(x, treatment, y, value, cost=None):
        if isinstance(value, int) or isinstance(value, float):
            flag = x >= value
        else:  # for strings
            flag = x == value
        if cost is not None:
            return x[flag], x[~flag], treatment[flag], treatment[~flag], y[flag], y[~flag], cost[flag], cost[~flag]
        else:
            return x[flag], x[~flag], treatment[flag], treatment[~flag], y[flag], y[~flag], None, None

    def total_gain(self, tree):
        if tree is None:
            return 0
        tmp = tree.gain
        tmp += self.total_gain(tree.branch_left)
        tmp += self.total_gain(tree.branch_right)
        return tmp

    @staticmethod
    def plot_tree(tree, x_name, score_name):

        nodes_data_tree = defaultdict(list)

        def to_string(is_split, tree, bBranch, szParent="null", indent='', indexParent=0, x_name="cnt"):
            if tree.value is None:
                nodes_data_tree[is_split].append(['leaf', "leaf", szParent, bBranch,
                                                  str(round(float(tree.summary['score']), 2)),
                                                  str(round(float(tree.summary['n_samples']) / 10000, 1)) + "w",
                                                  indexParent])
            else:
                if isinstance(tree.value, int) or isinstance(tree.value, float):
                    decision = '%s >= %s' % (x_name, tree.value)
                else:
                    decision = '%s == %s' % (x_name, tree.value)

                indexOfLevel = len(nodes_data_tree[is_split])
                to_string(is_split + 1, tree.branch_left, True, decision, indent + '\t\t', indexOfLevel, x_name)
                to_string(is_split + 1, tree.branch_right, False, decision, indent + '\t\t', indexOfLevel, x_name)
                nodes_data_tree[is_split].append([is_split + 1, decision, szParent, bBranch,
                                                  str(round(float(tree.summary['score']), 2)),
                                                  str(round(float(tree.summary['n_samples']) / 10000, 1)) + "w",
                                                  indexParent])

        to_string(0, tree, None, x_name=x_name)

        dots = ['digraph Tree {',
                'node [shape=box, style="filled, rounded", fontname=helvetica] ;',
                'edge [fontname=helvetica] ;'
                ]
        i_node = 0
        dcParent = {}
        for nSplit in range(len(nodes_data_tree.items())):
            lsY = nodes_data_tree[nSplit]
            indexOfLevel = 0
            for lsX in lsY:
                iSplit, decision, szParent, bBranch, score, n_samples, indexParent = lsX

                if type(iSplit) is int:
                    szSplit = '%d-%d' % (iSplit, indexOfLevel)
                    dcParent[szSplit] = i_node
                    dots.append('%d [label=< %s : %s<br/> n_samples : %s<br/> %s<br/>> ] ;' % (
                        i_node, score_name, score, n_samples, decision.replace('>=', '&ge;').replace('?', '')
                    ))
                else:
                    dots.append('%d [label=< %s : %s<br/> n_samples : %s<br/>>, fillcolor="%s"] ;' % (
                        i_node, score_name, score, n_samples, "green"
                    ))

                if szParent != 'null':
                    if bBranch:
                        szAngle = '45'
                        szHeadLabel = 'True'
                    else:
                        szAngle = '-45'
                        szHeadLabel = 'False'
                    szSplit = '%d-%d' % (nSplit, indexParent)
                    p_node = dcParent[szSplit]
                    if nSplit == 1:
                        dots.append('%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;' % (p_node,
                                                                                                       i_node, szAngle,
                                                                                                       szHeadLabel))
                    else:
                        dots.append('%d -> %d ;' % (p_node, i_node))
                i_node += 1
                indexOfLevel += 1
        dots.append('}')
        dot_data = '\n'.join(dots)
        graph = pydotplus.graph_from_dot_data(dot_data)
        from IPython.display import Image
        Image(graph.create_png())
        return graph


def test():
    raw_train = pd.read_csv('../../results/data_train_1030.csv')
    raw_test = pd.read_csv('../../results/data_test_1030.csv')
    print("数据集比例：", len(raw_train), len(raw_test))
    # 训练数据
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
    treatment = treatment_train.apply(lambda x: treat_dict[x])

    tree = PeopleUpliftTree(max_depth=5, min_samples_leaf=10, function_score="user-defined", function_gain="user-defined")
    # tree.fit(np.array(x_train['eta']), np.array(treatment), np.array(y_train))
    tree.fit(np.array(x_train['eta']), np.array(treatment), np.array(y_train))
    print(tree.idx)


if __name__ == "__main__":
    test()
