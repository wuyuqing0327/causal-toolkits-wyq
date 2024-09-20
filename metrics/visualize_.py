import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mul_treat_eva import plot_auuc

plt.style.use('fivethirtyeight')
RANDOM_COL = 'Random'
TRT_RATIO = 'trt_ratio'


def plot_all(cate=None, treatment_groups=None, treatment_test=None, y_test=None, cost_test=None, control_name='control', title='multi-treatment uplift curve', select_treatment_group=None,
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
        return plot_auuc(result, control_name='control', title=title, is_plot=False), qini_score(result, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite', flag=0)
    else:
        plot_qini(result, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite')
        print("\n############################")
        print("qini score is: {}".format(qini_score(result, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite')))
        print("############################\n")
        if plot_qini_lift == 0:
            plot_gain(result, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite')
            plot_lift(result, outcome_col='label', treatment_col='is_treated', treatment_effect_col='ite')
        plot_auuc(result, control_name='control', title=title)


def plot(df, kind='gain', n=100, figsize=(8, 8), *args, **kwarg):
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
        score = 'aucc score: %.2f' % score

    if (n is not None) and (n < df.shape[0]):
        df = df.iloc[np.linspace(0, df.index[-1], n, endpoint=True)]

    df.plot(figsize=figsize, secondary_y=TRT_RATIO)
    if score is not None:
        plt.title(score)
    plt.xlabel('Population')
    plt.ylabel('{}'.format(kind.title()))
    plt.grid(True)
    plt.legend()
    plt.savefig(kind + "_curve.png")
    plt.show()


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


def plot_gain(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              normalize=False, random_seed=42, n=100, figsize=(8, 8)):
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

    plot(df, kind='gain', n=n, figsize=figsize, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, normalize=normalize, random_seed=random_seed)


def plot_lift(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              random_seed=42, n=100, figsize=(8, 8)):
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

    plot(df, kind='lift', n=n, figsize=figsize, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, random_seed=random_seed)


def plot_qini(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              normalize=False, random_seed=42, n=100, figsize=(8, 8)):
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

    plot(df, kind='qini', n=n, figsize=figsize, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, normalize=normalize, random_seed=random_seed)


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
    cumgain = get_cumgain(df, outcome_col, treatment_col, treatment_effect_col, normalize)
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
