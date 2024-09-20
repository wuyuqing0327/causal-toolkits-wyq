import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_auuc(cate=None, control_name='control', title='multi-treatment uplift curve', is_plot=True):
    """
    cate(array): x在每个treatment下的treatment effect的预测值
    1. 根据pred_max_lift排序，并分桶
    2. 将每个桶中，
        实验组: 筛选出pred_max_treatment == actual_treatment的user
        对照组：可以使用全部的user；也可以根据第一步得到的实验组user在每个treatment上的数量来选择对照组user，在代码中，使用C_filt=True启用
    3. 计算每个组中的ATE, 之后思路和spark版本评估方式相同

    :param is_plot:
    :param control_name:
    :param title:
    :param cate: ndarray user在每个treatment上lift的预测值
    :param t_groups: array unique treatment name
    :param treatment: array
    :param y: gmv
    :param cost: 每种treatment的cost
    :return:

    """
    cate['pred_max_treat'] = cate['group']
    cate['pred_max_lift'] = cate['ite']
    cate['actual_treat'] = cate['group']
    cate['y'] = cate['label']

    T_buckets, C_buckets = split(cate, s='pred_max_lift', control_name=control_name, C_filt=False)
    lift = compute_lift(T_buckets, C_buckets)
    curve = compute_curve(lift)
    aucc, auuc = compute_auuc(curve)
    if is_plot:
        show_plot(curve, aucc, title)
    return aucc, auuc


def split(data, score, bins=10, control_name="control", C_filt=False):
    """
    分桶，并对实验组和对照组用户进行筛选
    :param control_name:
    :param data:
    :param bins:
    :param s:
    :param C_filt:
    :return:
    """
    T = data[data['actual_treat'] != control_name].sort_values(by=[score], ascending=False)
    C = data[data['actual_treat'] == control_name].sort_values(by=[score], ascending=False)

    n1, n2 = len(T) // bins, len(C) // bins  # 实验组/对照组每个桶的大小
    t_num, c_num, t_gmv, c_gmv, t_lift, c_lift, t_cost, c_cost = [], [], [], [], [], [], [], []
    for i in range(bins):
        # 多加了一个筛选条件，此条件对单treatment不起作用
        treat_bin = T[i * n1:(i + 1) * n1][T["pred_max_treat"] == T["actual_treat"]]
        control_bin = C[i * n2:(i + 1) * n2]
        if C_filt:  # 从control组中随机抽取与实验组pred_max_treat相同的行数
            filt_df = pd.DataFrame(columns=control_bin.columns)
            group_num = treat_bin['pred_max_treat'].value_counts().to_dict()
            for group, num in group_num.items():
                tmp = control_bin[control_bin['pred_max_treat'] == group]
                length = len(tmp.index)
                if length != 0:  # 如果控制组的行数 < 实验组的行数，则采用全部控制组样本，else不放回采样
                    if length < num:
                        filt_df = filt_df.append(tmp)
                    else:
                        filt_df = filt_df.append(tmp.sample(num))
            control_bin = filt_df

        t_num.append(len(treat_bin.index))
        c_num.append(len(control_bin.index))
        t_gmv.append(treat_bin['y'].sum())
        c_gmv.append(control_bin['y'].sum())
        t_lift.append(treat_bin[score].sum())
        c_lift.append(control_bin[score].sum())
        t_cost.append(np.float(treat_bin['cost'].sum()))
        c_cost.append(np.float(control_bin['cost'].sum()))

    T_buckets = pd.DataFrame([t_num, t_gmv, t_lift, t_cost], index=['t_num', 't_gmv', 't_lift', 't_cost']).T
    C_buckets = pd.DataFrame([c_num, c_gmv, c_lift, c_cost], index=['c_num', 'c_gmv', 'c_lift', 'c_cost']).T
    return T_buckets, C_buckets


def compute_lift(t, c):
    lift = pd.DataFrame()
    lift['id'] = pd.Series(list(range(len(t))))
    # truth_lift = t_gmv / t_num * c_num - c_gmv
    lift['truth_lift'] = c['c_num'] * t['t_gmv'] / t['t_num'] - c['c_gmv']
    # pred_lift = t_lift / t_num * c_num
    lift['prediction_lift'] = c['c_num'] * t['t_lift'] / t['t_num']
    # total_cost = t_cost. 此时c_cost为0
    lift['total_cost'] = t['t_cost']
    # num_treatment = t_num
    lift['num_treatment'] = t['t_num']
    # num_control = c_num
    lift['num_control'] = c['c_num']
    return lift


def compute_curve(lift):
    totalLift = lift['truth_lift'].sum()
    totalCost = lift['total_cost'].sum()
    accLift, accCost, ratio = 0, 0, 0
    ress = []
    for i in lift.index:
        accLift += lift.iloc[i]['truth_lift']
        accCost += lift.iloc[i]['total_cost']
        ratio += 1
        res = [ratio / 10, accLift / totalLift, accCost / totalCost]
        ress.append(res)
    curve = pd.DataFrame(ress, columns=['分位数', '累计增益', '累计成本'])
    return curve


def compute_auuc(curve):
    """
    计算逻辑: 利用微积分思想进行近似计算，每次计算一小块长方形的面积
    :param curve: (ratio, accLift, accCost)
    :return: 返回aucc和auuc值
    """
    aucc = 0
    auuc = 0

    for i in curve.index:
        if i == 0:
            aucc += curve.iloc[i, 1] * curve.iloc[i, 2]
            auuc += curve.iloc[i, 1] * curve.iloc[i, 0]
        else:
            aucc += curve.iloc[i, 1] * (curve.iloc[i, 2] - curve.iloc[i - 1, 2])
            auuc += curve.iloc[i, 1] * (curve.iloc[i, 0] - curve.iloc[i - 1, 0])
    return aucc, auuc


def show_plot(curve, aucc, title):
    plt.cla()
    x = curve['分位数']
    acc_cost = curve['累计成本']
    acc_lift = curve['累计增益']
    l_cost, = plt.plot([0] + list(x), [0] + list(acc_cost), label='acc_cost', color='r')
    l_lift, = plt.plot([0] + list(x), [0] + list(acc_lift), label='acc_lift', color='g')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    plt.legend(handles=[l_lift, l_cost], labels=['acc_lift', 'acc_cost'])
    plt.text(0.82, 0.1, 'aucc = %.2f' % aucc)
    plt.title(title)
    plt.grid(True)
    plt.savefig("aucc_curve.png")
    plt.show()
