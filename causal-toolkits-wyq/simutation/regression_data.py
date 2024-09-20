#!/usr/bin/env python
# coding: utf-8

import datetime
import joblib
import time
import pandas as pd
import math
import numpy as np
import os
import random

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['axes.unicode_minus'] = False   # （解决坐标轴负数的负号显示问题）

# find pyspark
# import findspark
# findspark.init()
# from pyspark.sql import Row
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, from_json, get_json_object
# spark = SparkSession.builder.master("yarn").appName("simulation_guoyanchen") \
# .config("spark.yarn.queue", "root.pingtaijishubu_yewupingtaibu_alg_prod") \
# .config("spark.driver.memory", "8g") \
# .config("spark.executor.memory", "8g") \
# .config("mapred.input.dir.recursive", "TRUE") \
# .config("hive.mapred.supports.subdirectories", "TRUE") \
# .enableHiveSupport() \
# .getOrCreate()

from causalml.simutation.features import extend_passenger_features, environments_features, order_features, cat_features_names, \
    extend_num_features_names
Num = 31
passengers_features = extend_passenger_features(num=Num)
num_features_names = extend_num_features_names(Num)


class Tools:

    def plot_hist(self, data, save_path, xlabel=None, ylabel=None, title=None, bins=100):
        plt.cla()
        plt.hist(data, bins)
        plt.xlabel(xlabel, size=14)
        plt.ylabel(ylabel, size=14)
        plt.title(title)
        plt.grid()
        # plt.show()
        plt.savefig(save_path + '.jpg')


class Config:
    """
    数据配置模块
    用户用户量，时间，输出地址
    """
    def __init__(self, p_num, start_data, test_days, mean_lam, std_lam,
                 mean_delta_lam, std_delta_lam):
        self.p_num = p_num  # 用户用户量
        self.user_name = 'guoyanchen'
        self.start_date = start_data
        self.test_days = test_days  # 时间单位：天
        self.hive_path = 'gulfstream.car_bonus_simulation_data_gyc'  # 为了与主表区分开，建议后缀姓名缩写

        self.mean_lam = mean_lam  # 实验时间内人均打车单量
        self.std_lam = std_lam  # 实验时间内打车单量标准差
        self.mean_delta_lam = mean_delta_lam  # 实验时间内人均提升单量
        self.std_delta_lam = std_delta_lam  # 实验时间内提升单量标准差, 此处做转换时另人均提升单量全部大于0


class BaseInfo:

    def __init__(self, config):
        self.config = config

    def main_base_info(self):
        """
        生成用户的基础特征
        :return:
        """
        # 将特征从dict转为四个list
        p_names, p_types, p_distributions, p_parameters = self.find_passengers_features(passengers_features)
        # e_names, e_types, e_distributions, e_parameters = \
        #     self.find_environments_features(environments_features, self.config.test_days)

        # 按照概率分布生成乘客信息，其中包含打车金特征(企业的成本)； 环境信息
        passengers = self.get_passengers(p_names, p_types, p_distributions, p_parameters)
        environments = self.get_environments()
        # print('乘客信息：')
        # print(passengers.head())
        # print('环境信息')
        # print(environments.head())

        # 连续型特征归一化
        num_features = passengers.loc[:, num_features_names]
        num_features = self.stand_scaler(num_features)
        passengers.loc[:, num_features_names] = num_features

        # 将类别特征转换为one hot表征，16列 -> 135列; 将连续型特征进行归一化
        # 这一步进行了特征的筛选，有一部分特征没有加进来
        cat_features = passengers.loc[:, cat_features_names]
        cat_features = pd.get_dummies(cat_features)
        process_passengers = pd.concat([cat_features, num_features], axis=1)
        return passengers, process_passengers, environments

    # 将用户、环境、订单信息从dict转为list
    @staticmethod
    def find_passengers_features(p_features):
        """
        将用户特征从dict转为list
        :param p_features:
        :return:
        """
        p_temp_names = ['p_var0%s_name' % i if i < 10 else 'p_var%s_name' % i for i in range(100)]
        p_temp_types = ['p_var0%s_type' % i if i < 10 else 'p_var%s_type' % i for i in range(100)]
        p_temp_distributions = ['p_var0%s_distribution' % i if i < 10 else 'p_var%s_distribution' % i for i in
                                range(100)]
        p_temp_parameters = ['p_var0%s_parameter' % i if i < 10 else 'p_var%s_parameter' % i for i in range(100)]

        p_names, p_types, p_distributions, p_parameters = [], [], [], []
        for a, b, c, d in zip(p_temp_names, p_temp_types, p_temp_distributions, p_temp_parameters):
            try:
                p_names.append(p_features[a])
                p_types.append(p_features[b])
                p_distributions.append(p_features[c])
                p_parameters.append(p_features[d])
            except:
                pass
        if not len(p_names) == len(p_types) == len(p_distributions) == len(p_parameters):
            print('请检查用户特征定义是否完整')
        return p_names, p_types, p_distributions, p_parameters

    @staticmethod
    def find_environments_features(e_features, test_days):
        """
        将环境特征从dict转为list格式
        :param e_features:
        :param test_days:
        :return:
        """
        # 找出所有有效环境特征
        e_num = test_days * 24
        e_temp_names = ['e_var0%s_name' % i if i < 10 else 'e_var%s_name' % i for i in range(100)]
        e_temp_types = ['e_var0%s_type' % i if i < 10 else 'e_var%s_type' % i for i in range(100)]
        e_temp_distributions = ['e_var0%s_distribution' % i if i < 10 else 'e_var%s_distribution' % i for i in
                                range(100)]
        e_temp_parameters = ['e_var0%s_parameter' % i if i < 10 else 'e_var%s_parameter' % i for i in range(100)]

        e_names, e_types, e_distributions, e_parameters = [], [], [], []
        for a, b, c, d in zip(e_temp_names, e_temp_types, e_temp_distributions, e_temp_parameters):
            try:
                e_names.append(e_features[a])
                e_types.append(e_features[b])
                e_distributions.append(e_features[c])
                e_parameters.append(e_features[d])
            except:
                pass
        if not len(e_names) == len(e_types) == len(e_distributions) == len(e_parameters):
            print('请检查环境特征定义是否完整')
        return e_names, e_types, e_distributions, e_parameters

    @staticmethod
    def find_orders_features(o_features):
        """
        将订单信息从dict转为list
        :return:
        :param o_features:
        :return:
        """
        o_temp_names = ['o_var0%s_name' % i if i < 10 else 'o_var%s_name' % i for i in range(100)]
        o_temp_types = ['o_var0%s_type' % i if i < 10 else 'o_var%s_type' % i for i in range(100)]
        o_temp_distributions = ['o_var0%s_distribution' % i if i < 10 else 'o_var%s_distribution' % i for i in
                                range(100)]
        o_temp_parameters = ['o_var0%s_parameter' % i if i < 10 else 'o_var%s_parameter' % i for i in range(100)]

        o_names, o_types, o_distributions, o_parameters = [], [], [], []
        for a, b, c, d in zip(o_temp_names, o_temp_types, o_temp_distributions, o_temp_parameters):
            try:
                o_names.append(o_features[a])
                o_types.append(o_features[b])
                o_distributions.append(o_features[c])
                o_parameters.append(o_features[d])
            except:
                pass
        if not len(o_names) == len(o_types) == len(o_distributions) == len(o_parameters):
            print('请检查用户特征定义是否完整')
        return o_names, o_types, o_distributions, o_parameters

    @staticmethod
    def get_cost(passengers):
        """
        cost值函数：打车金发放额度
        :param passengers:
        :return:
        """
        cost = pd.DataFrame(np.zeros((len(passengers), 1)))

        # cost  = np.random.choice([1, 2, 3, 4, 5], len(passengers), [0.5, 0.3, 0.1, 0.05, 0.05])
        # cost 服从poisson分布：9.7 - 21.2
        cost = np.random.poisson(150, len(passengers)) / 10

        return cost

    def get_passengers(self, p_names, p_types, p_distributions, p_parameters):
        """
        生成用户数据
        :param config:
        :param p_names:
        :param p_types:
        :param p_distributions:
        :param p_parameters:
        :return:
        """
        passengers = pd.DataFrame()

        # 生成用户的分组特征
        passengers['pid'] = np.random.choice(self.config.p_num * 10, self.config.p_num, replace=False)
        passengers['train_or_test'] = np.random.choice(['train', 'test'], len(passengers), p=[0.9, 0.1])
        passengers['group_type'] = np.array(passengers.index % 2)

        # 打车金作为treatment，不应再出现在用户的特征中
        # 生成用户的打车金发放特征
        # # give_cost = poisson(150) / 10
        # passengers['fake_give_cost'] = self.get_cost(passengers)
        # # receive_cost =  int(give_cost * rate**2 * 10) / 10
        # passengers['fake_receive_cost'] = (passengers['fake_give_cost'] * (
        #             np.random.random_sample(len(passengers)) ** 2) * 10).apply(int) / 10
        # # spend_cost = int(receive_cost * rate**3 * 10) / 10
        # passengers['fake_spend_cost'] = (passengers['fake_receive_cost'] * (
        #             np.random.random_sample(len(passengers)) ** 3) * 10).apply(int) / 10
        # fake_costs = passengers[['pid', 'fake_give_cost', 'fake_receive_cost', 'fake_spend_cost']].copy()
        # fake_costs.index = fake_costs.pid
        # # 将控制组group_type == 1 的give_cost, receive_cost, spend_cost设置为0
        # temp = passengers.fake_give_cost.copy()
        # temp[passengers.group_type == 1] = 0
        # passengers['give_cost'] = temp  # 将group_type == 1的give_cost设置为0
        # temp = passengers.fake_receive_cost.copy()
        # temp[passengers.group_type == 1] = 0
        # passengers['receive_cost'] = temp
        # temp = passengers.fake_spend_cost.copy()
        # temp[passengers.group_type == 1] = 0
        # passengers['spend_cost'] = temp

        # 生成其他基础特征
        passengers = self.add_columns_obey_prob(passengers, p_names, p_distributions, p_parameters)
        passengers.index = passengers.pid
        return passengers

    def get_environments(self):
        """
        按照概率分布生成每个小时的环境信息，包括：天气(晴、雨)，等待时间， 响应率
        :param config:
        :param e_names:
        :param e_types:
        :param e_distributions:
        :param e_parameters:
        :return:
        """
        # 生成环境数据
        environments = pd.DataFrame()
        start_date = datetime.datetime.strptime(self.config.start_date, '%Y-%m-%d')
        delta_days = datetime.timedelta(days=self.config.test_days)
        end_date = start_date + delta_days
        time_seq = pd.date_range(start_date, end_date, freq='H')[:-1]
        environments['dummy_date'] = [str(i).split(' ')[0] for i in time_seq]
        environments['dummy_time'] = [str(i).split(' ')[1] for i in time_seq]
        # environments = add_columns_obey_prob(environments, e_names, e_distributions, e_parameters)
        return environments

    @staticmethod
    def stand_scaler(num):
        """
        归一化连续型特征，标准正态分布
        :param num:
        :return:
        """
        tmp = num
        for i in num.columns:
            mean = num[i].mean()
            std = num[i].std()
            tmp[i] = (num[i] - mean) / std
        return tmp

    def add_columns_obey_prob(self, data, names, distributions, parameters):
        """
        根据设定的数据分布，生成数据，并添加到dataframe中
        :param data:
        :param names:
        :param distributions:
        :param parameters:
        :return:
        """
        for name, dis, par in zip(names, distributions, parameters):
            if dis == '正态分布':
                data[name] = self.zhengtai(par, len(data))  # 一次生成所有的数据
            elif dis == '指数分布':
                data[name] = self.zhishu(par, len(data))
            elif dis == '等概率抽样':
                data[name] = self.denggailv(par, len(data))
            elif dis == '不等概率抽样':
                data[name] = self.budenggailv(par, len(data))
            elif dis == '泊松分布':
                data[name] = self.posong(par, len(data))
            elif dis == '均匀分布':
                data[name] = self.junyun(par, len(data))
            elif dis == '帕累托分布':
                data[name] = self.pareto(par, len(data))
            else:
                print('代码中还没有添加这个分布')
        return data

    # 正态分布
    def zhengtai(self, parameter, num):
        return np.random.normal(parameter[0], parameter[1], num)

    # 指数分布
    def zhishu(self, parameter, num):
        return np.random.exponential(parameter[0], num)

    # 等概率抽样
    def denggailv(self, parameter, num):
        return np.random.choice(parameter[0], num)

    # 不等概率抽样
    def budenggailv(self, parameter, num):
        m = sum(parameter[1])
        p = np.array(parameter[1]) / m
        return np.random.choice(parameter[0], num, p=p)

    # 泊松分布
    def posong(self, parameter, num):
        return np.random.poisson(parameter[0], num)

    # 均匀分布
    def junyun(self, parameter, num):
        return np.random.uniform(parameter[0], parameter[1], num)

    # 帕累托分布
    def pareto(self, parameter, num):
        return np.random.pareto(parameter[0], num)


class OrderInfor:

    def __init__(self, config):
        self.config = config
        self.base = BaseInfo(config)
        self.tools = Tools()

    def main_order_info(self, passengers, process_passengers, environments, uplift_threshold, func_mode):
        """
        订单信息的主流程
        :param passengers:
        :param process_passengers:
        :param environments:
        :param uplift_threshold:
        :param func_mode:
        :return:
        """
        if not os.path.exists('splited_feature.pkl'):
            self.split_features(cat_features_names, num_features_names, split_proportion=[0.6, 0.1, 0.1, 0.2])
        splited_features = joblib.load('splited_feature.pkl')
        base_lams = self.get_base_lams(environments, passengers,
                                       splited_features['informative'] + splited_features['mix'])
        delta_lams = self.get_delta_lams(environments, passengers, splited_features['uplift'] + splited_features['mix'],
                                         uplift_threshold, func_mode)

        delta_lams_copy = delta_lams.copy()
        delta_lams_copy[passengers.group_type == 1] = 0  # 将控制组的增量设为0
        total_lams = base_lams + delta_lams_copy  # 用户的所有单数为基础单数 + 提升单数
        total_lams = total_lams.where(total_lams > 0, 0)  # 将 <0 的单量设置为0

        # 泊松分布：单位时间内随机事件发生的次数的概率分布，参数为单位时间内随机时间的平均发生次数。泊松采样生成订单量
        # 根据概率进行采样，在均值附近进行波动。对每个小时的订单量进行操作是为了增加随机性。
        hour_orders1 = total_lams.apply(np.random.poisson)  # apply函数作用于每一列

        # 为每个订单生成gmv等信息
        o_names, o_types, o_distributions, o_parameters = self.base.find_orders_features(order_features)
        orders_data1 = self.get_data(hour_orders1, passengers, environments, [o_names, o_distributions, o_parameters])
        print('length of order data:', len(orders_data1))

        # changed: process_passengers -> passengers
        passengers_data1 = self.generate_passengers(passengers, orders_data1)
        passengers_data1['ground_truth'] = delta_lams.sum(axis=1)  # 生成真实的ground_truth值

        self.analyze_data(passengers_data1, hour_orders1, process_passengers, self.config)

        # self.tools.plot_hist(base_lams.sum(axis=1), 'fig/base_Y_distribution', xlabel='base part of Y',
        #                      ylabel=' # of orders', title='Distribution of base orders')
        # self.tools.plot_hist(delta_lams.sum(axis=1), 'fig/uplift_Y_distribution', xlabel='uplift part of Y',
        #                      ylabel=' # of orders', title='Distribution of uplift orders')
        # self.tools.plot_hist(hour_orders1.sum(axis=1), 'fig/order_num_distribution', xlabel='number of orders',
        #                      ylabel=' # of users', title='Distribution of orders')

        return passengers_data1['gmv'], passengers_data1['group_type'], passengers_data1['ground_truth'], \
               base_lams.sum(axis=1)

    @staticmethod
    def split_features(cat_names, num_names, split_proportion=[0.6, 0.1, 0.1, 0.2]):
        """
        将特征分为：informative(4/5)、uplift related(1/10)、mix(1/10)、irrelevant(误差~N(0,1）)
        特征还分为：category、numerical
        实现分层抽样。
        :param cat_names:
        :param num_names:
        :param split_proportion:  list，[informative_prop, uplift_prop, mix_prop, irrelevant]
        :return: {informative_f, uplift_f, mix_f, irr_f}
        """
        # print(len(cat_names), len(num_names))  # 16, 34
        splited_l = []
        splited_feature = {}
        used = []  # 记录哪些特征已经被选过了
        for i in range(len(split_proportion)):
            splited_l.append([])
            for features in [cat_names, num_names]:
                features_num = len(features)
                features = list(set(features) - set(used))
                if i == len(split_proportion):  # 最后一步，将剩下的所有特征都加进去
                    splited_l[i].append(features)
                else:
                    selected = random.sample(features, int(features_num*split_proportion[i]))
                    used.extend(selected)
                    splited_l[i].append(selected)
        # print(splited_l)  # 29, 4, 4, 9

        splited_feature['informative'] = splited_l[0]
        splited_feature['uplift'] = splited_l[1]
        splited_feature['mix'] = splited_l[2]
        splited_feature['irr'] = splited_l[3]
        joblib.dump(splited_feature, 'splited_feature.pkl')  # 持久化

    @staticmethod
    def get_base_lam(env, pas, features):
        """
        lambda值函数：生成所有用户所有时刻，基础打车次数：特征的线性组合
        :param env:
        :param pas:
        :param features: 对基础单量有影响的feature：informative + mix
        :return:
        """
        cat_f = features[0] + features[2]
        num_f = features[1] + features[3]
        print('对基础订单量有作用的特征:')
        print(cat_f + num_f)
        pas_cat_data = pas.loc[:, cat_f]
        pas_cat_data = pd.get_dummies(pas_cat_data)   # 将选出的特征进行onehot表征
        pas_num_data = pas.loc[:, num_f]
        new_pas = pd.concat([pas.pid, pas_cat_data, pas_num_data], axis=1)

        lam = np.zeros(len(pas))
        # 随机生成每个特征的重要性
        importants = list(np.random.choice(list(range(-50, 50)), len(new_pas.columns)-1))
        # print(len(importants), len(new_pas.columns))  # 125, 126
        for i, j in zip(list(new_pas.columns)[1:], importants):
            # 之后再根据每个特征的取值 * 该特征的重要性，得到每个乘客总的的打车次数
            lam += new_pas[i] * j
        return lam

    @staticmethod
    def restrain_lam(matrix, mean, std):
        """
        将用户订单量矩阵(id * time)缩放到指定的分布config上。
        :param matrix:
        :param mean:
        :param std:
        :return:
        """
        me = matrix.mean().mean()  # 计算整个矩阵的均值
        st = np.sqrt(((matrix - me) * (matrix - me)).mean().mean())  # 计算整个矩阵的标准差
        # print(me, st)
        matrix = (matrix - me) / st  # 缩放为标准正态分布
        # 先将matrix缩放到（0，1）正态分布，再将其缩放到符合我们要求的分布，参考标准差和均值的性质
        # 在实验时间内，人均的打车单量
        # print(matrix.mean().mean(), np.sqrt(((matrix - me) * (matrix - me)).mean().mean()))
        matrix = matrix * (std / len(matrix.columns)) + mean / len(matrix.columns)
        matrix = matrix.where(matrix > 0, 0)  # 如果不满足>0，matrix = 0
        return matrix

    def get_base_lams(self, environments, pas, features):
        """
        分别计算每个用户在每种environment(根据时间变化的几个特征)情况下，的打车次数
        :param environments:
        :param pas:
        :param features: 起作用的特征
        :return:
        """
        lam = self.get_base_lam(environments.iloc[0, :], pas, features)
        columns = []
        for i in range(1, len(environments)):
            columns.append(str(environments.iloc[i, :]['dummy_date']) + ' ' + str(environments.iloc[i, :]['dummy_time']))
        lams = pd.DataFrame([lam] * len(columns), index=columns)
        lams = lams.T
        lams = self.restrain_lam(lams, self.config.mean_lam, self.config.std_lam)
        return lams

    @staticmethod
    def get_delta_lam(pas, features, func_mode):
        """
        delta_lambda函数：生成所有用户所有时刻提升的打车次数
        有8个特征，2个cat，6个num
        :param env:
        :param pas:
        :param features: 对uplift有作用的feature：uplift + mix
        :return:
        """

        cat_f = features[0] + features[2]
        num_f = features[1] + features[3]
        pas_cat_data = pas.loc[:, cat_f]
        pas_cat_data = pd.get_dummies(pas_cat_data)  # 将选出的特征进行onehot表征
        print('对uplift有作用的特征:')
        print(cat_f + num_f)

        # 对离散值进行线性组合后，经过非线性函数(x - 0.5)^2
        discrete_num = np.zeros(len(pas_cat_data))
        importants = list(np.random.choice(list(range(-50, 50)), len(pas_cat_data.columns)))
        for i, j in zip(list(pas_cat_data.columns)[1:], importants):
            discrete_num += pas_cat_data[i] * j  # 每个特征的取值 * 该特征的重要性
        discrete_num = (discrete_num - discrete_num.mean()) / discrete_num.std()  # normalization

        if func_mode == 0:
            X = pas.loc[:, num_f]
            tau = 2 * np.log1p(np.exp(X.iloc[:, 0] + X.iloc[:, 1])) \
                   + np.maximum(np.zeros(len(pas)), X.iloc[:, 3] + X.iloc[:, 4])\
                   + (discrete_num - 0.5) ** 2 + 2 * X.iloc[:, 5] + X.iloc[:, 2]
        return tau

    @staticmethod
    def restrain_delta_lam(matrix, mean, std, lift_threshold):
        me = matrix.mean().mean()
        st = np.sqrt(((matrix - me) * (matrix - me)).mean().mean())
        matrix = (matrix - me) / st  # 缩放为标准正态分布
        matrix = matrix * (std / len(matrix.columns)) + mean / len(matrix.columns)
        matrix = matrix.where(matrix > lift_threshold, 0)
        return matrix

    def get_delta_lams(self, environments, pas, features, uplift_threshold, func_mode):
        """
        delta_lambda函数：生成所有用户所有时刻提升打车次数的函数值
        :param environments:
        :param pas:
        :param importants:
        :return:
        """
        delta_lam = self.get_delta_lam(pas, features, func_mode)
        columns = []
        for i in range(1, len(environments)):
            columns.append(str(environments.iloc[i, :]['dummy_date']) + ' ' +
                           str(environments.iloc[i, :]['dummy_time']))
        delta_lams = pd.DataFrame([delta_lam] * len(columns), index=columns)
        delta_lams = delta_lams.T
        delta_lams = self.restrain_delta_lam(delta_lams, self.config.mean_delta_lam, self.config.std_delta_lam,
                                             uplift_threshold)
        return delta_lams

    @staticmethod
    def analyze_data(passengers, hour_orders, data, config):
        """
        分析实验数据
        :param passengers:
        :param hour_orders:
        :param data:
        :param config:
        :return:
        """
        simulation_data_info = pd.DataFrame([config.hive_path], columns=['table_address'])
        simulation_data_info['create_name'] = config.user_name
        simulation_data_info['create_date'] = config.start_date
        simulation_data_info['passengers_num'] = config.p_num
        simulation_data_info['features_num'] = len(data.columns)
        simulation_data_info['mean_orders'] = hour_orders.sum(axis=1).mean()

        T_orders_mean = hour_orders[passengers.group_type == 0].sum(axis=1).mean()
        C_orders_mean = hour_orders[passengers.group_type == 1].sum(axis=1).mean()
        simulation_data_info[r'mean_orders_T-C'] = T_orders_mean-C_orders_mean

        T_gmv_mean = passengers[passengers['group_type'] == 0].gmv.mean()
        C_gmv_mean = passengers[passengers['group_type'] == 1].gmv.mean()
        simulation_data_info['T_gmv_mean'] = T_gmv_mean
        simulation_data_info['C_gmv_mean'] = C_gmv_mean
        simulation_data_info[r'gmv_mean_T/C'] = str(round((T_gmv_mean/C_gmv_mean-1)*100, 3)) + '%'

        print(simulation_data_info.T)

    def get_infos(self, hour_orders, passengers, environments):
        """
        # 生成原始信息，此信息只包所有订单不为0的乘客信息，未复制多次打车的乘客信息
        将所有用户的订单信息集中起来。
        :param hour_orders:
        :param passengers:
        :param environments:
        :return:
        """

        data = pd.DataFrame()
        for time in hour_orders.columns:
            hour_data = passengers.iloc[:, :3].copy()
            hour_data['hour_orders'] = np.array(hour_orders[time])
            for e_feathers in environments.columns:
                hour_data[e_feathers] = np.array(environments.loc[time][e_feathers])[0]
            hour_data = hour_data[hour_data.hour_orders != 0]  # 删除订单为0的用户
            # 将每天每个小时的有订单的用户连接起来
            data = pd.concat([data, hour_data], ignore_index=True)
        return data

    @staticmethod
    def duplicate_orders(data):
        """
        复制多次打车的乘客的信息,为了给每个乘客的每个订单加一些其他的信息，方便后续累加生成乘客粒度的信息
        :param data:
        :return:
        """
        copys = []
        for i in range(1, 10):
            copy = data.copy()
            tmp = copy[copy.hour_orders >= i]
            copys.append(tmp)
        res = pd.concat(copys, ignore_index=True)
        return res

    def get_data(self, orders, passengers, environments, o_info):
        environments.index = [environments['dummy_date']+' '+environments['dummy_time']]
        order_infos = self.get_infos(orders, passengers, environments)
        origin_data = self.duplicate_orders(order_infos)
        # 为每个订单附加一些独立于其他信息的订单信息。
        final_data = self.base.add_columns_obey_prob(origin_data, o_info[0], o_info[1], o_info[2])
        return final_data

    @staticmethod
    def generate_passengers(passengers, orders_data):
        """
        # ## 生成以用户为粒度的数据
        :param passengers:
        :param orders_data:
        :return:
        """
        passengers_data = passengers.copy()
        passengers_data['gmv'] = orders_data.groupby(by=["pid"])['order_cost'].sum()
        passengers_data = passengers_data.fillna(0)
        return passengers_data


class Validation:
    """
    主要进行模型效果的检验：
    1. uplift curve
    2. auuc
    """

    def __init__(self, passengers, passengers_data1, passengers_data2, hour_orders1, hour_orders2):
        """
        这里面应该有信息的冗余
        :param passengers:
        :param passengers_data1:
        :param passengers_data2:
        :param hour_orders1:
        :param hour_orders2:
        """
        """
        Input：每条数据的分组信息、预测uplift值、真实uplift值、gmv、cost
        Output：
        """
        self.main(passengers_data1)

    def main(self, passengers_data1):
        tmp = self.add_rand(passengers_data1)
        raw, auc = self.merge_compute_auuc(tmp, 10, 'ground_truth')
        raw.columns = ['ii', 'acc_lift', 'acc_cost']
        raw['i'] = np.array(range(1, len(raw) + 1)) / len(raw)
        self.plot_curve(raw, auc, 'ss')
        print(raw)

    def merge_compute_auuc(self, data, bins, s):
        t, c = self.split(data, bins, s)
        lift = self.compute_lift(t, c)
        curve = self.compute_curve(lift)
        auuc = self.compute_auuc(curve)
        return curve, auuc

    @staticmethod
    def split(data, bins, s):
        # print(list(data.keys()))  # todo: changed
        T = data[data['group_type'] == 0].sort_values(by=[s], ascending=False)
        C = data[data['group_type'] == 1].sort_values(by=[s], ascending=False)

        n1, n2 = len(T)//bins, len(C)//bins
        t_num, c_num, t_gmv, c_gmv, t_lift, c_lift, t_cost, c_cost = [], [], [], [], [], [], [], []
        for i in range(bins):
            t_num.append(n1)
            c_num.append(n2)
            t_gmv.append(T['gmv'][(i)*n1:(i+1)*n1].sum())
            c_gmv.append(C['gmv'][(i)*n2:(i+1)*n2].sum())
            t_lift.append(T[s][(i)*n1:(i+1)*n1].sum())
            c_lift.append(C[s][(i)*n1:(i+1)*n1].sum())
            t_cost.append(np.float(T['spend_cost'][(i)*n1:(i+1)*n1].sum()))
            c_cost.append(np.float(C['spend_cost'][(i)*n2:(i+1)*n2].sum()))
        T_buckets = pd.DataFrame([t_num, t_gmv, t_lift, t_cost]).T
        C_buckets = pd.DataFrame([c_num, c_gmv, c_lift, c_cost]).T
        return T_buckets, C_buckets

    @staticmethod
    def compute_lift(t, c):
        lift = pd.DataFrame()
        lift['id'] = pd.Series(list(range(len(t))))
        # todo: 这个lift计算应该有问题。truth用的是gmv的差，pred用的是真实的函数生成的lift
        # truth_lift = t_gmv / t_num * c_num - c_gmv
        lift['truth_lift'] = t.iloc[:, 1] / t.iloc[:, 0] * c.iloc[:, 0] - c.iloc[:, 1]
        # pred_lift = t_lift / t_num * c_num
        lift['prediction_lift'] = t.iloc[:, 2] / t.iloc[:, 0] * c.iloc[:, 0]
        # total_cost = t_cost. 此时c_cost为0
        lift['total_cost'] = t.iloc[:, 3]
        # num_treatment = t_num
        lift['num_treatment'] = t.iloc[:, 0]
        # num_control = c_num
        lift['num_control'] = c.iloc[:, 0]
        return lift

    @staticmethod
    def compute_curve(lift):
        totalLift = lift['truth_lift'].sum()
        totalCost = lift['total_cost'].sum()
        accLift, accCost, ratio = 0, 0, 0
        ress = []
        for i in lift.index:
            # todo: 这个lift计算有问题：
            accLift += lift.iloc[i]['truth_lift']
            accCost += lift.iloc[i]['total_cost']
            ratio += 1
            res = [ratio / 100, accLift / totalLift, accCost / totalCost]
            ress.append(res)
        curve = pd.DataFrame(ress, columns=['分位数', '累计增益', '累计成本'])
        return curve

    @staticmethod
    def compute_auuc(curve):
        acc = 0
        for i in curve.index:
            if i == 0:
                acc += curve.iloc[i, 1]*curve.iloc[i, 2]
            else:
                acc += curve.iloc[i, 1]*(curve.iloc[i, 2]-curve.iloc[i-1, 2])
        return acc

    @staticmethod
    def add_rand(select_data):
        select_data['rand'] = np.random.normal(select_data.ground_truth.mean(),
                                               select_data.ground_truth.std(), len(select_data))
        select_data['X_ground_truth'] = 0.5*select_data.rand+0.5*select_data.ground_truth
        return select_data

    @staticmethod
    def plot_curve(raw, auc, title):
        # curve = pd.DataFrame([['0', '0', '0']] +list(map(lambda x: x.split("\t"), raw.split("\n"))),
        # columns = ['i', 'acc_cost', 'acc_lift']).astype("float")
        curve = raw
        x = curve['i']
        acc_cost = curve['acc_cost']
        acc_lift = curve['acc_lift']
        l_cost, = plt.plot([0]+list(x), [0]+list(acc_cost), label = 'acc_cost')
        l_lift, = plt.plot([0]+list(x), [0]+list(acc_lift), label = 'acc_lift')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
        plt.legend(handles=[l_cost, l_lift], labels=['acc_cost', 'acc_lift'])
        plt.text(0.82, 0.1, 'auc = %.2f' % auc)
        # plt.savefig("./auc/%s" % title)
        plt.grid()
        plt.show()


def get_regression_data(n, uplift_threshold=(-0.0002), func_mode=0, is_onehot=True):
    """

    :param n:  number of examples
    :param uplift_threshold:    uplift单量的最小值，当<0时，说明存在'sleeping dog'类用户
    :param func_mode:
    :return:
    """
    config = Config(p_num=n, start_data='2080-01-01', test_days=7, mean_lam=2, std_lam=5,
                    mean_delta_lam=0.01, std_delta_lam=0.2)
    base_info = BaseInfo(config)
    order_info = OrderInfor(config)
    s = datetime.datetime.now()
    passengers, X, environments = base_info.main_base_info()
    y, w, tau, b = order_info.main_order_info(passengers, X, environments, uplift_threshold, func_mode)
    e = datetime.datetime.now()
    print('生成数据用时', e - s)
    if is_onehot:
        return y, X, w, tau, b
    else:
        del passengers['pid']
        del passengers['train_or_test']
        del passengers['group_type']
        return y, passengers, w, tau, b


def get_classification_data(n, cost_ratio, is_onehot):
    """

    :param n:
    :param cost_ratio:
    :return: user_df, x_names
    """
    y, X, w, tau, b = get_regression_data(n, is_onehot=is_onehot)
    x_names = list(X.columns)
    X['gmv'] = y
    X['cost'] = cost_ratio * y + np.random.randn(len(y))
    X.loc[X.cost < 0, 'cost'] = 0
    X.loc[X.gmv < X.cost, 'cost'] = X.gmv * cost_ratio

    X['group'] = w
    X.loc[X['group'] == 1, 'group'] = 'control'
    X.loc[X['group'] == 0, 'group'] = 'treatment'
    return X, x_names




if __name__ == '__main__':
    # get_regression_data(n=10000, func_mode=0)
    get_classification_data(1000, 0.05)

# ## 落数据
# drop1 = """alter table gulfstream.car_bonus_uplift_model_feature_table_gyc
# drop partition (year = '2020', month = '07', day = '08')"""
# drop2 = """alter table gulfstream.car_bonus_uplift_model_feature_table_gyc_info
# drop partition (year = '2020', month = '07', day = '08')"""
# spark.sql(drop1)
# spark.sql(drop2)

# drop_query = '''
# drop table gulfstream.car_bonus_simuliation_data_gyc
# '''
# spark.sql(drop_query)
# drop_query = '''
# drop table gulfstream.car_bonus_simuliation_data_gyc_info
# '''
# spark.sql(drop_query)

#
# def get_create_query(p_data, path):
#     query = ''
#     for i, j in zip(p_data.dtypes.index, p_data.dtypes):
#         if j == 'object':
#             query += str(i)+' string, \n'
#         elif j == 'int64':
#             query += str(i)+' bigint, \n'
#         elif j == 'float64':
#             query += str(i)+' double, \n'
#         else:
#             print(i, j, "出错")
#     query = query.rstrip(', \n')
#     create_query = """
# create table if not exists {}(
# {}
# )
# COMMENT 'Simuliation data'
# partitioned by (year string, month string, day string, user_name string)
# ROW FORMAT DELIMITED
# FIELDS TERMINATED BY '|'
# STORED AS TEXTFILE
#     """.format(path, query)
#     return create_query
#
# print(get_create_query(passengers_data1, config.hive_path))





# import os
# def save_data(data, config, experience = False):
#     path = config.hive_path
#     create_query = get_create_query(data, path)
#     spark.sql(create_query)
#     try:
#         os.mkdir('Simulation_data')
#     except:
#         pass
#     print('将乘客数据写入hive表:', config.hive_path)
#     data.to_csv('Simulation_data/data.csv', header = False, index = False, sep = '|')
#
#     start_date = datetime.datetime.strptime(config.start_date, '%Y-%m-%d')
#     if experience:
#         delta_days = datetime.timedelta(days = config.test_days)
#         end_date = start_date+delta_days
#         end_date = str(end_date).split()[0]
#         year, month, day = end_date.split('-')[0], end_date.split('-')[1], end_date.split('-')[2]
#     else:
#         start_date = str(start_date).split()[0]
#         year, month, day = start_date.split('-')[0], start_date.split('-')[1], start_date.split('-')[2]
#     load_query = '''
#     load data local inpath './Simulation_data/data.csv' into table {}
#     partition (year = '{}', month = '{}', day = '{}', user_name = '{}')
#     '''.format(path, year, month, day, config.user_name)
#     print(load_query)
#     spark.sql(load_query)
# save_data(passengers_data1, config)
# save_data(passengers_data2, config, experience = True)
#
#
# def save_info(data_info, config, experience = False):
#     data_info.to_csv('./Simulation_data/data_infos.csv', mode = 'w+')
#     path = config.hive_path+'_info'
#     create_query = get_create_query(data_info, path)
#     spark.sql(create_query)
#     try:
#         os.mkdir('Simulation_data')
#     except:
#         pass
#     print('将乘客信息写入hive表:', path)
#     data_info.to_csv('Simulation_data/data_info.csv', header = False, index = False, sep = '|')
#
#     start_date = datetime.datetime.strptime(config.start_date, '%Y-%m-%d')
#     if experience:
#         delta_days = datetime.timedelta(days = config.test_days)
#         end_date = start_date+delta_days
#         end_date = str(end_date).split()[0]
#         year, month, day = end_date.split('-')[0], end_date.split('-')[1], end_date.split('-')[2]
#     else:
#         start_date = str(start_date).split()[0]
#         year, month, day = start_date.split('-')[0], start_date.split('-')[1], start_date.split('-')[2]
#     load_query = '''
#     load data local inpath './Simulation_data/data_info.csv' into table {}
#     partition (year = '{}', month = '{}', day = '{}', user_name = '{}')
#     '''.format(path, year, month, day, config.user_name)
#     print(load_query)
#     spark.sql(load_query)
# save_info(data_info1, config)
# save_info(data_info2, config, True)
#
#
# select_query = '''
# select * from gulfstream.car_bonus_simulation_data_gyc_info order by day, year, month
# '''.format(config.user_name)
# select_data = spark.sql(select_query).toPandas()
# print(len(select_data))
# select_data
#
#
# select_query = '''
# select * from gulfstream.car_bonus_simulation_data_gyc where concat(year, month, day) = {} limit 10
# '''.format(''.join(config.start_date.split('-')))
# print(select_query)
# select_data = spark.sql(select_query).toPandas()
# select_data
#
# exit()
#
#
# def rmse(a, b):
#     return np.sqrt(((a-b)**2).mean())
#
#
#
#
#
# def mape(y_true, y_pred):
#     """
#     参数:
#     y_true -- 测试集目标真实值
#     y_pred -- 测试集目标预测值
#
#     返回:
#     mape -- MAPE 评价指标
#     """
#
#     n = len(y_true)
#     mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
#     return mape




# 计算每一项数据与均值的差

'''
def de_mean(x):
    x_bar = np.mean(x)
    return [x_i - x_bar for x_i in x]
# 辅助计算函数 dot product 、sum_of_squares
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    return dot(v, v)


# 方差
def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)


# 标准差
def standard_deviation(x):
    return math.sqrt(variance(x))


def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n -1)


def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0

'''