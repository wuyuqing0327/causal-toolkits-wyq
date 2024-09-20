import copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


plt.style.use('fivethirtyeight')
RANDOM_COL = 'Random'
TRT_RATIO = 'trt_ratio'

def plot_auuc(path, cate=None, control_name='control', title='multi-treatment uplift curve', is_plot=True):
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
        show_plot(curve, aucc, title, path)
    return aucc, auuc

def split(data, s, bins=10, control_name="control", C_filt=False):
    """
    分桶，并对实验组和对照组用户进行筛选
    :param control_name:
    :param data:
    :param bins:
    :param s:
    :param C_filt:
    :return:
    """
    T = data[data['actual_treat'] != control_name].sort_values(by=[s], ascending=False)
    C = data[data['actual_treat'] == control_name].sort_values(by=[s], ascending=False)

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
        t_lift.append(treat_bin[s].sum())
        c_lift.append(control_bin[s].sum())
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


def show_plot(curve, aucc, title, path):
    plt.cla()
    x = curve['分位数']
    acc_cost = curve['累计成本']
    acc_lift = curve['累计增益']
    l_cost, = plt.plot([0] + list(x), [0] + list(acc_cost), label='acc_cost', color='r')
    l_lift, = plt.plot([0] + list(x), [0] + list(acc_lift), label='acc_lift', color='g')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    plt.legend(handles=[l_lift, l_cost], labels=['acc_lift', 'acc_cost'], loc='upper right')
    plt.text(0.82, 0.1, 'aucc = %.2f' % aucc)
    plt.title(title + ' ' + 'aucc = %.2f' % aucc)
    plt.grid(True)
    plt.savefig(path + title + "_" + "aucc_curve.png", bbox_inches="tight")
    # plt.show()

def plot_all(path, cate=None, treatment_groups=None, treatment_test=None, y_test=None, cost_test=None, control_name='control', title='multi-treatment uplift curve', select_treatment_group=None,
             plot_qini_lift=1, is_find_best_parameters=0):
    """
    把每个Treatment组拆开和Control组进行比较
    :param is_find_best_parameters:
    :param plot_qini_lift:
    :param select_treatment_group:
    :param cate:
    :param treatment_groups:
    :param treatment_test:
    :param y_test:
    :param cost_test:
    :param control_name:
    :param title:
    :return:
    """
    result = pd.DataFrame(cate, columns=treatment_groups)
    result['group'] = treatment_test
    result['label'] = y_test
    result['cost'] = cost_test
    result['is_treated'] = result['group'].apply(lambda x: 0 if x == control_name else 1)
    result = result[(result['group'] == select_treatment_group) | (result['group'] == control_name)]
    t_groups_copy = copy.deepcopy(treatment_groups)
    t_groups_copy.remove(select_treatment_group)
    result = result.drop(columns=t_groups_copy)
    result.rename(columns={select_treatment_group: "ite"}, inplace=True)
    if is_find_best_parameters:
        return plot_auuc(path, cate=result, control_name=0, title=title, is_plot=False), qini_score(result, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite', flag=0)
    else:
        plot_qini(result, path, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite', title=title)
        print("\n############################")
        print("auuc score is: {}".format(auuc_score(result, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite')))
        print("qini score is: {}".format(qini_score(result, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite')))
        print("############################\n")
        if plot_qini_lift == 0:
            plot_gain(result, path, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite', title=title)
            plot_lift(result, path, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite', title=title)
        plot_auuc(path, cate=result, control_name=control_name, title=title)


def plot(df, path, kind='gain', n=100, figsize=(8, 8), title=None, *args, **kwarg):
    """Plot one of the lift/gain/Qini charts of model estimates.

    A factory method for `plot_lift()`, `plot_gain()` and `plot_qini()`. For details, pleas see docstrings of each
    function.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns.
        kind (str, optional): the kind of plot to draw. 'lift', 'gain', and 'qini' are supported.
        n (int, optional): the number of samples to be used for plotting.
    """
    catalog = {'lift': get_cumlift,
               'gain': get_cumgain,
               'qini': get_qini}

    assert kind in catalog.keys(), '{} plot is not implemented. Select one of {}'.format(kind, catalog.keys())

    df = catalog[kind](df, *args, **kwarg)
    df = df.rename(columns={'tau': 'ite'})
    score = None
    if kind == "qini":
        df_copy = df.div(df.iloc[-1, :], axis=1)
        score = (df_copy.sum(axis=0) - df_copy[RANDOM_COL].sum()) / df_copy.shape[0]
        score = score.to_dict()['ite']
        score = 'qini score: %.2f' % score
    elif kind == "gain":
        df_copy = df.div(np.abs(df.iloc[-1, :]), axis=1)
        score = df_copy.sum() / df_copy.shape[0]
        score = score.to_dict()['ite']
        score = 'auuc score: %.2f' % score

    if (n is not None) and (n < df.shape[0]):
        df = df.iloc[np.linspace(0, df.index[-1], n, endpoint=True)]

    df.plot(figsize=figsize, secondary_y=TRT_RATIO)
    if score is not None:
        plt.title(title+' '+score)
    else:
        plt.title('%s_Cumulative Population Lift' %title)
    plt.xlabel('Population')
    plt.ylabel('{}'.format(kind.title()))
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(path + title + kind + "_curve.png", bbox_inches="tight")
    # plt.show()


def get_cumlift(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                random_seed=42):
    """Get average uplifts of model estimates in cumulative population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the mean of the true treatment effect in each of cumulative population.
    Otherwise, it's calculated as the difference between the mean outcomes of the
    treatment and control groups in each of cumulative population.

    For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
    and Uplift Modeling: A review of the literature`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): average uplifts of model estimates in cumulative population
    """

    assert ((outcome_col in df.columns) and (treatment_col in df.columns) or
            treatment_effect_col in df.columns)
    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = '__random_{}__'.format(i)
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    model_names = [treatment_effect_col]
    model_names.extend(random_cols)

    lift = []
    for i, col in enumerate(model_names):
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        # to calculate the average treatment_effects of cumulative population.
        df['cumsum_tr'] = df[treatment_col].cumsum()
        df['cumsum_ct'] = df.index.values - df['cumsum_tr']
        df['cumsum_y_tr'] = (df[outcome_col] * df[treatment_col]).cumsum()
        df['cumsum_y_ct'] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()

        lift.append(df['cumsum_y_tr'] / df['cumsum_tr'] - df['cumsum_y_ct'] / df['cumsum_ct'])

    lift = pd.concat(lift, join='inner', axis=1)
    lift.loc[0] = np.zeros((lift.shape[1],))
    lift = lift.sort_index().interpolate()

    lift.columns = model_names
    lift[RANDOM_COL] = lift[random_cols].mean(axis=1)
    lift.drop(random_cols, axis=1, inplace=True)

    return lift


def get_cumgain(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                normalize=False, random_seed=42):
    """Get cumulative gains of model estimates in population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
    and Uplift Modeling: A review of the literature`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in population
    """

    lift = get_cumlift(df, outcome_col, treatment_col, treatment_effect_col, random_seed)

    # cumulative gain = cumulative lift x (# of population)
    gain = lift.mul(lift.index.values, axis=0)

    if normalize:
        gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)
    return gain


def get_qini(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
             normalize=False, random_seed=42):
    """Get Qini of model estimates in population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in population
    """
    assert ((outcome_col in df.columns) and (treatment_col in df.columns) or
            treatment_effect_col in df.columns)

    df = df.copy()
    random_cols = []
    np.random.seed(random_seed)
    for i in range(10):
        random_col = '__random_{}__'.format(i)
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    model_names = [treatment_effect_col]
    model_names.extend(random_cols)

    qini = []
    for i, col in enumerate(model_names):
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df['cumsum_tr'] = df[treatment_col].cumsum()
        df['cumsum_ct'] = df.index.values - df['cumsum_tr']
        df['cumsum_y_tr'] = (df[outcome_col] * df[treatment_col]).cumsum()
        df['cumsum_y_ct'] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()

        l = df['cumsum_y_tr'] - df['cumsum_y_ct'] * \
            df['cumsum_tr'] / df['cumsum_ct']

        if col == treatment_effect_col:
            trt_rate = df['cumsum_tr'] / df.index
            df[treatment_effect_col].cumsum()
        qini.append(l)

    qini.append(trt_rate)
    qini = pd.concat(qini, join='inner', axis=1)
    qini.loc[0] = np.zeros((qini.shape[1],))
    qini = qini.sort_index().interpolate()

    qini.columns = model_names + [TRT_RATIO]
    qini[RANDOM_COL] = qini[random_cols].mean(axis=1)
    qini.drop(random_cols, axis=1, inplace=True)

    if normalize:
        qini = qini.div(qini.iloc[-1, :], axis=1)

    return qini


def plot_gain(df, path, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              normalize=False, random_seed=42, n=100, figsize=(8, 8), title='gain'):
    """Plot the cumulative gain chart (or uplift curve) of model estimates.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
    and Uplift Modeling: A review of the literature`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()
        n (int, optional): the number of samples to be used for plotting
    """

    plot(df, path, kind='gain', n=n, figsize=figsize, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, normalize=normalize, random_seed=random_seed, title=title)


def plot_lift(df, path, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              random_seed=42, n=100, figsize=(8, 8), title='lift'):
    """Plot the lift chart of model estimates in cumulative population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the mean of the true treatment effect in each of cumulative population.
    Otherwise, it's calculated as the difference between the mean outcomes of the
    treatment and control groups in each of cumulative population.

    For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
    and Uplift Modeling: A review of the literature`.

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        random_seed (int, optional): random seed for numpy.random.rand()
        n (int, optional): the number of samples to be used for plotting
    """

    plot(df, path, kind='lift', n=n, figsize=figsize, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, random_seed=random_seed, title=title)


def plot_qini(df, path, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              normalize=False, random_seed=42, n=100, figsize=(8, 8), title='qini'):
    """Plot the Qini chart (or uplift curve) of model estimates.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()
        n (int, optional): the number of samples to be used for plotting
    """

    plot(df, path, kind='qini', n=n, figsize=figsize, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, normalize=normalize, random_seed=random_seed, title=title)


def auuc_score(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau', normalize=True):
    """Calculate the AUUC (Area Under the Uplift Curve) score.

     Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not

    Returns:
        (float): the AUUC score
    """
    # print("ooooookkkkkk")
    cumgain = get_cumgain(df, outcome_col, treatment_col, treatment_effect_col, normalize)
    # print("kkkkkkkkooooooo")
    return cumgain.sum() / cumgain.shape[0]


def qini_score(df, outcome_col='y', treatment_col='w', treatment_effect_col='ite', normalize=True, flag=1):
    """Calculate the Qini score: the area between the Qini curves of a model and random.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

     Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not

    Returns:
        (float): the Qini score
    """

    qini = get_qini(df, outcome_col, treatment_col, treatment_effect_col, normalize)
    if flag:
        print(qini.sum(axis=0), qini[RANDOM_COL].sum(), qini.shape[0])
    qini = (qini.sum(axis=0) - qini[RANDOM_COL].sum()) / qini.shape[0]
    return qini[treatment_effect_col]