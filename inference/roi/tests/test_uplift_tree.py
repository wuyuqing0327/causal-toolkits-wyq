from causalml.inference.roi import UpliftTreeRegressor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .const import RANDOM_SEED, N_SAMPLE, CONTROL_NAME, TREATMENT_NAMES, CONVERSION, COST
from causalml.simutation.regression_data import get_classification_data
from causalml.metrics.uplift_curve import plot as plot_uplift_curve


def test_data_make(generate_regression_data):
    # df, x_names = generate_regression_data()
    # print(df)
    # print(x_names)
    a = [1, 2, 3, 4, 5, 6, 7, -10]
    b = 100
    c = list(map(lambda x: x / b, a))
    print(c)


def test_percentile(generate_regression_data):
    df, x_names = generate_regression_data()

    print("hehhe")
    X = df[x_names].values
    treatment = df['treatment_group_key'].values
    y = df[CONVERSION].values
    c = df[COST].values
    rows = [list(X[i]) + [treatment[i]] + [y[i]] + [c[i]] for i in range(len(X))]

    columnValues = [row[-2] for row in rows]
    lspercentile = np.percentile(columnValues, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    lsUnique = list(set(lspercentile))
    lsUnique.sort()
    print(lsUnique)
    if lsUnique[0] == 0:
        lsUnique[0] = 0.00001
    print(lsUnique)
    print(max(df['gmv']))

    results = {}
    for row in rows:
        # treatment group in the 2nd last column
        r = row[-3]
        y = row[-2]
        c = row[-1]
        if r not in results:
            results[r] = [0, 0, 0, [0] * (len(lsUnique) + 1)]

        results[r][0] += 1
        results[r][1] += y
        results[r][2] += c

        bucket_index = len(lsUnique)
        for i, val in enumerate(lsUnique):
            if y < val:
                bucket_index = i
                break
        results[r][3][bucket_index] += 1
    print(results)


def test_get_uplift():
    data = {

        'is_treated': np.random.randint(0, 2, 100),
        'gmv': np.random.rand(100) * 100,
        'predict': np.random.rand(100),
        'cost': np.random.rand(100) * 10
    }
    df = pd.DataFrame(data, columns=['is_treated', 'gmv', 'predict', 'cost'])
    df.info()
    print(df.head(100))
    print("hehe")
    print(df.pivot_table(values='gmv',
                         index='is_treated',
                         aggfunc=[np.mean, np.size, np.sum],
                         margins=True))
    print(df.pivot_table(values='cost',
                         index='is_treated',
                         aggfunc=[np.mean, np.size, np.sum],
                         margins=True))
    # plot_uplift(df, normalize=True)
    # plt.show()


def test_UpliftTreeRegressor():
    # df, x_names = generate_regression_data()

    df, x_names = get_classification_data(n=1000, cost_ratio=0.05, is_onehot=False)
    print(df.shape)
    df_train, df_test = train_test_split(df,
                                         test_size=0.2,
                                         random_state=RANDOM_SEED)

    # print("\ndf:")
    # print(df.pivot_table(values='gmv',
    #                      index='group',
    #                      aggfunc=[np.mean, np.size],
    #                      margins=True))
    # print("\ndf_train:")
    # print(df_train.pivot_table(values='gmv',
    #                            index='group',
    #                            aggfunc=[np.mean, np.size, np.sum],
    #                            margins=True))
    # print("\ndf_test:")
    # print(df_test.pivot_table(values='gmv',
    #                           index='group',
    #                           aggfunc=[np.mean, np.size, np.sum],
    #                           margins=True))
    # print(df_test.pivot_table(values='cost',
    #                           index='group',
    #                           aggfunc=[np.mean, np.size, np.sum],
    #                           margins=True))

    # df_test = df_train
    # Train the UpLift Random Forest classifer
    uplift_model = UpliftTreeRegressor(control_name=TREATMENT_NAMES[0], evaluationFunction='Roi', max_depth=10,
                                       max_features=5, x_names=x_names, min_samples_leaf=100)

    uplift_model.fit(X=df_train[x_names].values,
                     treatment=df_train['group'].values,
                     y=df_train[CONVERSION].values,
                     c=df_train[COST].values)

    _, _, _, y_pred = uplift_model.predict(df_test[x_names].values,
                                           full_output=True)

    df_test.reset_index()
    result_data = {
        "pid": np.arange(0, len(df_test)),
        "gmv": df_test['gmv'],
        "cost": df_test['cost'],
        "group": df_test['group'],
        "prediction": y_pred['treatment']
    }
    result = pd.DataFrame(result_data)
    print(result)
    plot_uplift_curve(result)

    # result = pd.DataFrame(y_pred)
    # print("\nresult: ", result.info())
    # print(result)
    # result.drop(CONTROL_NAME, axis=1, inplace=True)
    #
    # best_treatment = np.where((result < 0).all(axis=1),
    #                           CONTROL_NAME,
    #                           result.idxmax(axis=1))
    #
    # # Create a synthetic population:
    #
    # # Create indicator variables for whether a unit happened to have the
    # # recommended treatment or was in the control group
    # actual_is_best = np.where(
    #     df_test['group'] == "treatment", 1, 0
    # )
    # actual_is_control = np.where(
    #     df_test['group'] == CONTROL_NAME, 1, 0
    # )
    #
    # synthetic = (actual_is_best == 1) | (actual_is_control == 1)
    # synth = result[synthetic]
    # print("\nsynth: ", synth.info())
    # print(synth)
    #
    # auuc_metrics = synth.assign(
    #     is_treated=1 - actual_is_control[synthetic],
    #     gmv=df_test.loc[synthetic, CONVERSION].values,
    #     cost=df_test.loc[synthetic, COST].values,
    #     predict=synth.max(axis=1)
    # ).drop(columns=result.columns)
    #
    # print("\nauuc_metrics: ", auuc_metrics.info())
    # print(auuc_metrics)
    # plot_uplift(auuc_metrics, outcome_col='gmv', treatment_col='is_treated', predict_col="predict", cost_col="cost",
    #             normalize=True)
    #
    # plt.show()

    # graph = uplift_tree_plot(uplift_model.fitted_uplift_tree, x_names)
    # image = Image(graph.create_png())
    # im_array = np.array(image)
    # plt.imshow(im_array)
    # plt.show()

    # cumgain = get_cumgain(auuc_metrics,
    #                       outcome_col=CONVERSION,
    #                       treatment_col='is_treated')

    # Check if the cumulative gain of UpLift Random Forest is higher than
    # random
    # assert cumgain['uplift_tree'].sum() > cumgain['Random'].sum()

    # plt.show()
