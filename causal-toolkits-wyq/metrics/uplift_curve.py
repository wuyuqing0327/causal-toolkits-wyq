from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot(df):
    bucket_num = 10
    t = df.loc[df['group'] == 'treatment']
    c = df.loc[df['group'] == 'control']

    print("\t========= t count: ", len(t))
    print("\t========= c count: ", len(c))

    bucket_t = format_bucket(t, bucket_num)
    bucket_c = format_bucket(c, bucket_num)

    print("bucket_t.columns: ", bucket_t.columns)
    print(bucket_t)
    print(bucket_c)

    bucket_all = bucket_t.merge(bucket_c, left_index=True, right_index=True, how='inner').reset_index()
    bucket_all['true_lift'] = bucket_all['gmv_x'] / bucket_all['pid_x'] * bucket_all['pid_y'] - bucket_all['gmv_y']
    bucket_all['subsidy_rate'] = bucket_all['cost_x'] / bucket_all['gmv_x']
    bucket_all['avg_gmv'] = bucket_all['gmv_x'] / bucket_all['pid_x']
    print("bucket_all.columns: ", bucket_all.columns)
    print(bucket_all)

    acc_all = bucket_all[['true_lift', 'cost_x']].cumsum()
    acc_all = acc_all / acc_all.iloc[-1]
    print("acc_all.columns: ", acc_all.columns)
    print(acc_all)
    auuc = compute_auuc(acc_all)
    plot_curve(bucket_all, acc_all, bucket_num, auuc)


def plot_curve(bucket, acc, bucket_num, auuc):
    x = np.arange(0, 1.01, 1 / bucket_num)

    # fig, (ax2, ax1) = plt.subplots(2)
    ax2 = plt.axes([0.15, 0.5, 0.75, 0.5])
    ax1 = plt.axes([0.15, 0.0, 0.75, 0.5])
    acc_lift = [0.0] + list(acc['true_lift'].values)
    acc_cost = [0.0] + list(acc['cost_x'].values)
    l_cost, = ax1.plot(x, acc_cost)
    l_lift, = ax1.plot(x, acc_lift)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    ax1.legend(handles=[l_cost, l_lift], labels=['acc_cost', 'acc_lift'])
    ax1.text(0.8, 0.2, 'auc=%.2f' % auuc)

    avg_gmv = [0] + list(bucket['avg_gmv'].values)
    subsidy_rate = [None] + list(bucket['subsidy_rate'].values)
    # l_gmv, = plt.plot(x, avg_gmv, label='avg_gmv')
    # l_subsidy, = plt.plot(x, subsidy_rate, label='subsidy_rate')
    # plt.legend(handles=[l_gmv, l_subsidy], labels=['avg_gmv', 'subsidy_rate'])

    ax2_twin = ax2.twinx()
    l_gmv = ax2.bar(x, avg_gmv, 0.05)
    l_subsidy, = ax2_twin.plot(x, subsidy_rate, 'r--')
    ax2.legend(handles=[l_gmv, l_subsidy], labels=['avg_gmv', 'subsidy_rate'])
    plt.show()


def format_bucket(df, bucket_num):
    num_per_bucket = len(df) / bucket_num

    sort_df = df.sort_values(by='prediction', ascending=False).reset_index(drop=True)
    sort_df['bucket_id'] = (sort_df.index / num_per_bucket).astype('int')

    bucket_df = sort_df.groupby(by='bucket_id').agg({'cost': 'sum', 'gmv': 'sum', 'pid': 'count'})  # .reset_index()

    return bucket_df


def compute_auuc(acc):
    acc_lift = list(acc['true_lift'].values)
    acc_cost = list(acc['cost_x'].values)

    res = 0.0
    for i in range(len(acc_lift)):
        if i == 0:
            res += acc_lift[i] * acc_cost[i]
        else:
            res += acc_lift[i] * (acc_cost[i] - acc_cost[i - 1])
    return res


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    data_size = 200
    data = {
        'pid': np.arange(0, data_size),
        'gmv': np.random.rand(data_size) * 100,
        'cost': np.random.rand(data_size) * 5,
        'group': list(map(lambda x: "treatment" if x > 2 else "control", np.random.randint(0, 10, data_size))),
        'prediction': np.random.rand(data_size)
    }

    df = pd.DataFrame(data)
    df.loc[df['group'] == 'control', 'cost'] = 0
    # print(df)
    plot(df)
