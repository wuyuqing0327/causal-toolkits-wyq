

import pandas as pd
import numpy as np

# 特征列表
'''
name-特征名字           中英文均可
type-特征类型           选择范围{int, float, class}
distribution-分布类型   选择范围{正态分布，泊松分布，均匀分布，指数分布，等概率抽样，不等概率抽样}
parameter-根据不同分布选择不同的参数
    正态分布：[mu, sigma]
    泊松分布：[lambda]
    均匀分布：[min，max]
    指数分布：[lambda]
    等概率抽样：[[被抽取的样本]]
    不等概率抽样：[[被抽取的样本]，[每个样本被抽取到的概率, 此处概率总和超过1也可以，后期会进行概率归一化]]
'''

# ### 用户属性特征
passengers_features = {
    'p_var00_name': 'd_pas_app_type',
    'p_var00_type': 'string',
    'p_var00_distribution': '不等概率抽样',
    'p_var00_parameter': [['1', '101'], [14, 14]],

    'p_var01_name': 'd_phone_model',
    'p_var01_type': 'string',
    'p_var01_distribution': '不等概率抽样',
    'p_var01_parameter': [['iPhone', 'iPhone10, 3', 'iPhone9, 2', 'iPhone9, 1', 'iPhone10, 2', 'iPhone8, 2', \
                           'iPhone8, 1', 'iPhone11, 6', 'iPhone10, 1', 'iPhone7, 2', 'iPhone11, 8', 'iPhone6', \
                           'EML-AL00', 'OPPOR9s', 'iPhone7, 1', 'OPPOR11', 'MHA-AL00', 'vivoX9', 'PACM00', 'MI8'], \
                          [13, 19, 7, 18, 1, 12, 10, 17, 15, 19, 7, 1, 1, 0, 9, 8, 11, 6, 13, 15]],

    'p_var02_name': 'g_pas_sex',
    'p_var02_type': 'string',
    'p_var02_distribution': '不等概率抽样',
    'p_var02_parameter': [['0', '1', '2'], [11, 19, 6]],

    'p_var03_name': 'g_pas_constellation',
    'p_var03_type': 'string',
    'p_var03_distribution': '不等概率抽样',
    'p_var03_parameter': [['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                          [17, 1, 12, 16, 3, 1, 7, 18, 14, 3, 7, 9]],

    'p_var04_name': 'g_pas_trade_id',
    'p_var04_type': 'string',
    'p_var04_distribution': '不等概率抽样',
    'p_var04_parameter': [['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'],
                          [13, 16, 0, 7, 4, 12, 2, 19, 0, 2, 3, 6, 6, 3, 0]],

    'p_var05_name': 'g_bd_passenger_lifecycle_type',
    'p_var05_type': 'string',
    'p_var05_distribution': '不等概率抽样',
    'p_var05_parameter': [['成熟期', '衰退期', '再激活期', '流失期', '沉默期', '新生期'], [1, 2, 1, 4, 10, 19]],

    'p_var06_name': 'g_bd_passenger_rfm_type',
    'p_var06_type': 'string',
    'p_var06_distribution': '不等概率抽样',
    'p_var06_parameter': [['低价值', '发展', '潜力', '高价值', '流失用户', '维系', '新用户'], [6, 7, 14, 18, 16, 8, 18]],

    'p_var07_name': 'g_bd_passenger_profile_occupation',
    'p_var07_type': 'string',
    'p_var07_distribution': '不等概率抽样',
    'p_var07_parameter': [['0', '1'], [2, 3]],

    'p_var08_name': 'g_pas_edu_type',
    'p_var08_type': 'string',
    'p_var08_distribution': '不等概率抽样',
    'p_var08_parameter': [['1', '2', '3', '4'], [19, 6, 5, 5]],

    'p_var09_name': 'g_pas_auth_state',
    'p_var09_type': 'string',
    'p_var09_distribution': '不等概率抽样',
    'p_var09_parameter': [['0', '1', '2', '3', '4', '5', '6', '7', '8'], [17, 19, 3, 15, 4, 10, 1, 4, 13]],

    'p_var10_name': 'g_bd_level_id',
    'p_var10_type': 'string',
    'p_var10_distribution': '不等概率抽样',
    'p_var10_parameter': [['0', '1', '2', '3', '4', '5', '6'], [6, 3, 17, 18, 18, 16, 1]],

    'p_var11_name': 'g_bd_age_level',
    'p_var11_type': 'string',
    'p_var11_distribution': '不等概率抽样',
    'p_var11_parameter': [['0', '1', '2', '3', '4', '5', '6', '7'], [3, 3, 3, 17, 11, 4, 14, 12]],

    'p_var12_name': 'city_id',
    'p_var12_type': 'string',
    'p_var12_distribution': '不等概率抽样',
    'p_var12_parameter': [
        ['154', '286', '118', '93', '25', '106', '39', '161', '301', '155', '135', '92', '58', '170', '105', '157', \
         '45', '188', '147', '79', '253', '83', '160', '16', '17', '156', '8', '90', '19', '95', '44', '117', '10',
         '255', '137'],
        [19, 15, 3, 0, 3, 11, 0, 17, 15, 4, 6, 12, 6, 1, 17, 18, 16, 1, 8, 12, 10, 3, 6, 13, 11, 12, 3, 6, 1, 8, 12, 2,
         10, 19, 3]],

    'p_var13_name': 'f_coupon_rely_level',
    'p_var13_type': 'string',
    'p_var13_distribution': '不等概率抽样',
    'p_var13_parameter': [['A', 'B', 'C', 'D'], [10, 9, 14, 4]],

    'p_var14_name': 'f_order_subsidy_sensitive_level',
    'p_var14_type': 'string',
    'p_var14_distribution': '不等概率抽样',
    'p_var14_parameter': [['A', 'B', 'C', 'D'], [19, 10, 13, 7]],

    'p_var15_name': 'f_subsidy_sensitive',
    'p_var15_type': 'string',
    'p_var15_distribution': '不等概率抽样',
    'p_var15_parameter': [['A', 'B', 'C'], [13, 13, 4]],

    'p_var16_name': 'moon_income',
    'p_var16_type': 'float',
    'p_var16_distribution': '正态分布',
    'p_var16_parameter': [4000, 1500],

    'p_var17_name': 'g_pas_age',
    'p_var17_type': 'float',
    'p_var17_distribution': '正态分布',
    'p_var17_parameter': [40, 10],

    'p_var18_name': 'o_finish_order_cnt_30day',
    'p_var18_type': 'float',
    'p_var18_distribution': '正态分布',
    'p_var18_parameter': [40, 10],

    # 'p_var19_name': 's_f_0',  # simulation_float_num
    # 'p_var19_type': 'float',
    # 'p_var19_distribution': '正态分布',
    # 'p_var19_parameter': [4000, 1500],
}

# ### 环境特征
environments_features = {
    'e_var01_name': 'weather',
    'e_var01_type': 'class',
    'e_var01_distribution': '不等概率抽样',
    'e_var01_parameter': [[0, 1], [0.8, 0.2]],  # 0：晴   1：雨

    'e_var02_name': 'wait_time',
    'e_var02_type': 'float',
    'e_var02_distribution': '指数分布',
    'e_var02_parameter': [1],

    'e_var03_name': 'response_rate',
    'e_var03_type': 'float',
    'e_var03_distribution': '指数分布',
    'e_var03_parameter': [1],
}

# ### 订单特征
order_features = {
    'o_var01_name': 'order_cost',
    'o_var01_type': 'float',
    'o_var01_distribution': '正态分布',
    'o_var01_parameter': [20, 5],  # 单位分钟 # 后面累加这个值作为某个用户的gmv。单位应该是rmb/h

    # 'o_var02_name' : '路途长度',
    # 'o_var02_type' : 'float',
    # 'o_var02_distribution' : '正态分布',
    # 'o_var02_parameter' : [1],

    # 'o_var03_name' : '订单价格',
    # 'o_var03_type' : 'float',
    # 'o_var03_distribution' : '正态分布',
    # 'o_var03_parameter : [1, 1],

    'o_var04_name': 'other_subsidy',
    'o_var04_type': 'float',
    'o_var04_distribution': '指数分布',
    'o_var04_parameter': [1],
}


def extend_passenger_features(num):
    """
    将用户特征从20个增加到20+num个，增加的都是正态分布的连续特征
    :param num:
    :return:
    """
    for i in range(num):
        passengers_features['p_var%s_name' % (i + 19)] = 's_f_%s' % i
        passengers_features['p_var%s_type' % (i + 19)] = 'float'
        passengers_features['p_var%s_distribution' % (i + 19)] = '正态分布'
        passengers_features['p_var%s_parameter' % (i + 19)] = [0, 1]

    # print(passengers_features.keys())
    # print(len(passengers_features) / 4)
    return passengers_features


# 类别型特征、连续型特征名称
cat_features_names = [
    'd_pas_app_type',
    'd_phone_model',
    'g_pas_sex',
    'g_pas_constellation',
    'g_pas_trade_id',
    'g_bd_passenger_lifecycle_type',
    'g_bd_passenger_rfm_type',
    'g_bd_passenger_profile_occupation',
    'g_pas_edu_type',
    'g_pas_auth_state',
    'g_bd_level_id',
    'g_bd_age_level',
    'city_id',
    'f_coupon_rely_level',
    'f_order_subsidy_sensitive_level',
    'f_subsidy_sensitive'
]
num_features_names = [
    'moon_income',
    'g_pas_age',
    'o_finish_order_cnt_30day',
    # 'fake_receive_cost'
]


def extend_num_features_names(num):
    for i in range(num):
        num_features_names.append('s_f_%s' % i)
    return num_features_names
