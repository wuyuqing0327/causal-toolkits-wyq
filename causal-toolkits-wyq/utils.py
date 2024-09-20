import numpy as np
import pandas as pd


def increasing_constrain(currentNodeSummary):
    """
    强制单调性约束函数
    :param currentNodeSummary: 节点的各个Treatment的信息
    :return: int (1单调递增，-1单调递减)
    """
    keys = list(currentNodeSummary.keys())
    keys.remove('control')
    x = sorted(keys)
    x.append('control')
    y = []
    for i in x:
        # print(isinstance(currentNodeSummary[i], list))
        if isinstance(currentNodeSummary[i], list):
            y.append(currentNodeSummary[i][2])
        else:
            y.append(currentNodeSummary[i])
    return is_decreasing_list(y)


def is_increasing_list(y):
    for i in range(len(y) - 1):
        if y[i] > y[i + 1]:
            return 0
    return 1


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


def read_csv(data_dir, verbose=True):
    """
    pandas 内存优化类
    :param verbose:
    :param data_dir: 文件目录
    :return: dataframe
    """
    df = pd.read_csv(data_dir, parse_dates=True, keep_date_col=True)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
