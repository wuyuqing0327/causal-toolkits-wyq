import warnings

warnings.filterwarnings("ignore")
import pickle
from time import ctime
import gc
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait
from causalml.inference.tree.models_ import UpliftRandomForestClassifier
from causalml.metrics.visualize_ import plot_all

"""
寻找RF模型的最佳参数
"""


class SearchBestParametersRF(object):
    def __init__(self, parameters_dict, n_estimators, evaluationFunction, is_constrained, base_path, max_workers,
                 x_train, x_test, treatment_train, treatment_test, y_train, y_test, cost_test, t_groups):
        self.x_train = x_train
        self.x_test = x_test
        self.treatment_train = treatment_train
        self.treatment_test = treatment_test
        self.y_train = y_train
        self.y_test = y_test
        self.cost_test = cost_test
        self.t_groups = t_groups
        self.parameters_dict = parameters_dict
        self.n_estimators = n_estimators
        self.evaluationFunction = evaluationFunction
        self.is_constrained = is_constrained
        self.base_path = base_path
        self.max_workers = max_workers

    def get_best_parameters(self):
        """
        多进程并行查找
        """
        print('开始查找时间:', ctime())
        tasks = []
        executor = ProcessPoolExecutor(max_workers=self.max_workers)
        for max_depth in self.parameters_dict['max_depth']:
            for max_features in self.parameters_dict['max_features']:
                for min_samples_leaf in self.parameters_dict['min_samples_leaf']:
                    for min_samples_treatment in self.parameters_dict['min_samples_treatment']:
                        for n_reg in self.parameters_dict['n_reg']:
                            if min_samples_treatment * len(self.t_groups) <= min_samples_leaf:
                                tasks.append(
                                    executor.submit(self.getAUCC, max_depth, max_features, min_samples_leaf, min_samples_treatment, n_reg))

        print("已开启进程数:", len(tasks))
        wait(tasks)
        print('结束时间:', ctime())
        resultList_RF = []
        for i in range(len(tasks)):
            resultList_RF.append(tasks[i].result())
        executor.shutdown()
        return resultList_RF

    def getAUCC(self, max_depth, max_features, min_samples_leaf, min_samples_treatment, n_reg):
        """
        获取单个模型的结果
        :param max_depth:
        :param max_features:
        :param min_samples_leaf:
        :param min_samples_treatment:
        :param n_reg:
        :return:
        """
        learner_rf = UpliftRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_treatment=min_samples_treatment,
            n_reg=n_reg,
            control_name='control',
            evaluationFunction=self.evaluationFunction,
            is_constrained=self.is_constrained
        )
        learner_rf.fit(self.x_train, self.treatment_train, self.y_train)
        cate = learner_rf.predict(self.x_test, is_generate_json=False)
        cate = pd.DataFrame(cate, columns=self.t_groups)
        qini_scores = []
        aucc_scores = []
        for group in self.t_groups:
            aucc_score, qini_score = plot_all(cate=cate, treatment_groups=self.t_groups, treatment_test=self.treatment_test,
                                              y_test=self.y_test, cost_test=self.cost_test, title="rf-learner-test multi-treatment uplift curve",
                                              select_treatment_group=group, is_find_best_parameters=1)
            qini_scores.append(qini_score)
            aucc_scores.append(aucc_score)
        learner_rf.aucc_score = np.mean(aucc_scores)
        learner_rf.qini_score = np.mean(qini_scores)
        res_list = [self.evaluationFunction, max_depth, max_features, min_samples_leaf, min_samples_treatment, n_reg]
        # 保存模型
        file_name = self.base_path + '_'.join(map(str, res_list)) + '.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(learner_rf, f)
        del learner_rf, cate
        gc.collect(generation=0)  # 垃圾回收
        return [max_depth, max_features, min_samples_leaf, min_samples_treatment, n_reg, np.mean(aucc_scores)]
