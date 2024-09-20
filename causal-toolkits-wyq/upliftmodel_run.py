import numpy as np
import pandas as pd
import os
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pylift.eval import UpliftEval
from metrics.visualize import qini_score as QINI
from metrics.visualize import auuc_score as AUUC
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier, BaseRClassifier
from inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from inference.tree.models_ import UpliftRandomForestClassifier
from common import *
from config import *

# from inference.nn import CEVAE
from inference.nn import DragonNet
from propensity import ElasticNetPropensityModel

# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.neural_network import MLPClassifier, MLPRegressor

# from inference.nn import CEVAE
# from inference.tf import DragonNet
from propensity import ElasticNetPropensityModel

from plot_causal import plot_all, plot, auuc_score as Auuc_score, qini_score as Qini_score


from sklearn.preprocessing import OneHotEncoder
from feature_selection.preprocessor import LabelEncoder
from feature_selection.selector import Selector
from feature_selection.filters import FilterSelect
from feature_selection.filter_uplift_tree import *
from config import *
from datetime import datetime
from tensorflow import keras

warnings.filterwarnings("ignore")
logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)



channel_type = 'churn'
if channel_type == 'activate':
    input_sample_path = ACTIVATE_CHANNEL_PATH
    refactor_sample_path = ACTIVATE_CHANNEL_REFACTOR_PATH
    cleaned_sample_path = ACTIVATE_CHANNEL_CLEANED_PATH
elif channel_type == 'churn':
    input_sample_path = CHURN_CHANNEL_PATH
    refactor_sample_path = CHURN_CHANNEL_REFACTOR_PATH
    cleaned_sample_path = CHURN_CHANNEL_CLEANED_PATH
elif channel_type == 'cross_churn':
    input_sample_path = CROSS_CHURN_PATH
    refactor_sample_path = CROSS_CHURN_REFACTOR_PATH
    cleaned_sample_path = CROSS_CHURN_CLEANED_PATH
print(input_sample_path)


data = load_df(input_sample_path)
print(data.shape)

rename = {}
for i in list(data.columns):
    rename.update({i: i.split('.')[1]})
data.rename(columns= rename, inplace=True)

data['get_coupon_amount'].value_counts()

data['treatment_group']= np.where(data['get_coupon_amount']==0, 'control',
                          np.where(data['get_coupon_amount']==6, 'treatment_06',
                            np.where(data['get_coupon_amount']==12, 'treatment_12',
                             np.where(data['get_coupon_amount']==15, 'treatment_15',
                             np.where(data['get_coupon_amount']==20, 'treatment_20', np.nan)))))


data['treatment_group'].value_counts()
data['is_create']=np.where(pd.isnull(data['route_id']),0,1)
data['is_finish']=np.where(pd.isnull(data['order_id']),0,1)
print(data['is_create'].mean())
print(data['is_finish'].mean())
data.groupby(['treatment_group']).agg({'is_create':'mean','is_finish':'mean','get_coupon_amount':'mean'})

### split data 
def data_split(data):

    fea = list(data.columns)
#     fea = [x for x in data.columns if x not in [LABEL_COL, TREATMENT_COL]]
    y = data[LABEL_COL]
    X = data[fea]

    fea_train, fea_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1221)

    X_train = fea_train[fea]
    X_test = fea_test[fea]
    treat_train = fea_train[TREATMENT_COL]
    treat_test = fea_test[TREATMENT_COL]

    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test
    treat_train = treat_train
    treat_test = treat_test
    return X_train, X_test, y_train, y_test, treat_train, treat_test 
X_train_data, X_test_data, y_train, y_test, treat_train, treat_test = data_split(data)
print(X_train_data.shape)
print(X_test_data.shape)


# preprocessing
def preprocessing(df):
    # 空值替换
    for i in EXT_ONEHOT_COLS:
        df[i] = np.where(pd.isnull(df[i]), 'missing', df[i])
        df[i] = np.where(df[i] == '0', 'missing', df[i])
    for i in LEVEL_COLS:
        df[i] = np.where(pd.isnull(df[i]), 'missing', df[i])
        df[i] = np.where(df[i] == '0', 'missing', df[i])
    df_new = df.replace([np.nan, -999999, 999999, -1, -99999, 99999, -9999, 9999], 0)
    data_y = df_new[LABEL_COL]
    data_treat = df_new[TREATMENT_COL]
    df_new.drop(DROP_COLS, axis=1, inplace=True, errors='ignore')

    ohe = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
    lbe = LabelEncoder() 
    X_ohe = ohe.fit_transform(df_new[EXT_ONEHOT_COLS])  
    df_X_ohe = pd.DataFrame(X_ohe, columns=ohe.get_feature_names(), index=df.index)
    for i in LEVEL_COLS:
        df_new = df_new.sort_values([i])
    df_new[LEVEL_COLS] = lbe.fit_transform(df_new[LEVEL_COLS])

    df_new.drop(EXT_ONEHOT_COLS, axis=1, inplace=True)
    data_X = pd.concat([df_new, df_X_ohe], axis=1)

    # selector剔除唯一值
    slc = Selector(data=data_X, labels=data_y)
    # slc.identify_missing(missing_threshold=0.6)
    slc.identify_single_unique()
    data_X = slc.remove(methods=['single_unique'])
    

    fs = FilterSelect()
#     ## 特征间相关性
#     rest_vars = fs.filter_P(pd.concat([data_X,data_y],axis=1), list(data_X.columns), LABEL_COL, k=500)
    
#     print("pearson_after_final_vars:", len(rest_vars))
    
    ## 特征的overlap差异
    feat_imp = fs.get_importance(pd.concat([data_X,data_treat,data_y],axis=1), list(data_X.columns)[::-1], y_name=LABEL_COL, method='KL',
                                     experiment_group_column=TREATMENT_COL,
                                     control_group='control')
    top_n_features_df = feat_imp[['feature','rank','score','misc','method']]
    top_n_features_df = top_n_features_df[top_n_features_df['score']>0]
    print("top_n_features:", top_n_features_df.shape)
    top_n_features_df.to_csv('best_model/top_n_features.csv', index=False)
    top_n_features = top_n_features_df['feature'].tolist()
        
    
    if COST_COL not in list(data_X[top_n_features].columns):
        final_df = pd.concat([data_X[top_n_features], df[COST_COL], data_treat, data_y, 
                              df[['route_from_grid_id_12','open_app_time','open_app_hour']]], axis=1)
        top_n_features.append(COST_COL)
    else:
        final_df = pd.concat([data_X[top_n_features], data_treat, data_y, 
                              df[['route_from_grid_id_12','open_app_time','open_app_hour']]], axis=1)
    print("feature preprocess end, data_no_missing.shape is: ", final_df.shape)
    pickle.dump(ohe, open('best_model/ohe.pkl', 'wb'))
    pickle.dump(lbe, open('best_model/lbe.pkl', 'wb'))
    pickle.dump(top_n_features, open('best_model/features.pkl', 'wb'))
    return final_df,top_n_features


def process_pred(df, process_path='best_model/'):
    """
        Desc: when prediction, The feature processing is the same as the prediction
        Return:
            input_path: raw data from hive sql
            output_path: cleaned data for model prediction
    """
    if EXT_ONEHOT_COLS:
        for i in EXT_ONEHOT_COLS:
            df[i] = np.where(pd.isnull(df[i]), 'missing', df[i])
            df[i] = np.where(df[i] == '0', 'missing', df[i])
    if LEVEL_COLS:
        for i in LEVEL_COLS:
            df[i] = np.where(pd.isnull(df[i]), 'missing', df[i])
            df[i] = np.where(df[i] == '0', 'missing', df[i])
    df = df.replace([np.nan, -999999, 999999, -1, -99999, 99999, -9999, 9999], 0)
    print("prediction: process_feature begins:", df.shape)
    ohe = pickle.load(open(process_path + "ohe.pkl", "rb+"))
    lbe = pickle.load(open(process_path + "lbe.pkl", "rb+"))
    X_ohe = ohe.transform(df[EXT_ONEHOT_COLS].astype(str))
    df_X_ohe = pd.DataFrame(X_ohe, columns=ohe.get_feature_names(), index=df.index)
    df[LEVEL_COLS] = lbe.transform(df[LEVEL_COLS])
    new_df = pd.concat([df, df_X_ohe], axis=1)
    features = pickle.load(open(process_path + "features.pkl", "rb+"))
    final_df = new_df[features + [TREATMENT_COL, LABEL_COL] + ['route_from_grid_id_12','open_app_time','open_app_hour']]
    print("prediction: predict data process finish:", final_df.shape)
    return final_df

X_train,features = preprocessing(X_train_data)
print(X_train.shape)
X_test = process_pred(X_test_data)
print(X_test.shape)

X_train.to_csv('best_model/X_train.csv', index=False)
X_test.to_csv('best_model/X_test.csv', index=False)

### modelling
X_train = pd.read_csv('best_model/X_train.csv')
X_test = pd.read_csv('best_model/X_test.csv')
feature = pickle.load(open("best_model/" + "features.pkl", "rb+"))
print(X_train.shape)
print(X_test.shape)
print(len(feature))

def auuc(treat_train, data, model, model_name, control_name='control', path='plot/'):
    t_groups = np.unique(treat_train[treat_train != control_name])
    t_groups.sort()
    t_groups_names = []
    for group in t_groups:
        t_groups_names.append('delta_{}'.format(group))
    X_data = {}
    y_data = {}
    treat_data = {}
    cost_data = {}
    w = {}
    tau_data = {}
    upev_data = {}
    auuc_score = []
    causlmlauuc = []
    causlmlqini = []
    treatment_groups = []

    fea = pickle.load(open("best_model/features.pkl", "rb+"))
    for group in t_groups:
        group_name = t_groups_names[list(t_groups).index(group)]
        X_data[group] = data[(data[TREATMENT_COL] == control_name) | (data[TREATMENT_COL] == group)][
            fea]
        y_data[group] = data[(data[TREATMENT_COL] == control_name) | (data[TREATMENT_COL] == group)][
            LABEL_COL]
        treat_data[group] = \
            data[(data[TREATMENT_COL] == control_name) | (data[TREATMENT_COL] == group)][
                TREATMENT_COL]
        cost_data[group] = \
            data[(data[TREATMENT_COL] == control_name) | (data[TREATMENT_COL] == group)][
                COST_COL]
        w[group] = (treat_data[group] == group).astype(int)

        if model_name == 'uplift_tree':
            tau_data[group] = pd.DataFrame(model.predict(X_data[group].values), columns=list([group_name]))

#         elif model_name == 'DragonNet':
#             X_data_scaled = scale_x.transform(X_data[group].values)
#             tau_data[group] = pd.DataFrame(model.predict_tau(X_data_scaled), columns=list([group_name]))

        else:
            tau_data[group] = pd.DataFrame(model.predict(X_data[group]))
            tau_data[group].columns = list(t_groups_names)
            tau_data[group] = pd.DataFrame(tau_data[group][group_name], columns=[group_name])

        upev_data[group] = UpliftEval(w[group], y_data[group], tau_data[group]['delta_{}'.format(group)],
                                      n_bins=100)
        auuc_score_i = upev_data[group].Q_cgains

        w_group = pd.concat([w[group].reset_index(drop=True), y_data[group].reset_index(drop=True),
                             tau_data[group].reset_index(drop=True)], axis=1)
        w_group.columns = [TREATMENT_COL, LABEL_COL, 'ite']
        predict = pd.concat([tau_data[group].reset_index(drop=True), y_data[group].reset_index(drop=True), treat_data[group].reset_index(drop=True)], axis=1)
        predict.columns = ['tau', 'y', 'w']
        predict['w'] = np.where(predict['w'] == 'control', 0, 1)

        causlmlauuc_i = Auuc_score(w_group, outcome_col=LABEL_COL, treatment_col=TREATMENT_COL, treatment_effect_col='ite')
        causlmlqini_i = Qini_score(w_group, outcome_col=LABEL_COL, treatment_col=TREATMENT_COL, treatment_effect_col='ite')

        auuc_score.append(auuc_score_i)
        causlmlauuc.append(causlmlauuc_i)
        causlmlqini.append(causlmlqini_i)
        treatment_groups.append(group)
        plot_all(path=path, cate=tau_data[group].values, treatment_groups=list([group]),
                 treatment_test=treat_data[group].values,
                 y_test=y_data[group].values, cost_test=cost_data[group].values,
                 title="%s_Model %s" % (model_name, group),
                 select_treatment_group=group, control_name=control_name, plot_qini_lift=1)

        plot(df=predict, path=path, kind='qini', n=100, figsize=(8, 8), title='get_qini')
        plot(df=predict, path=path, kind='gain', n=100, figsize=(8, 8), title='get_cumgain')
        plot(df=predict, path=path, kind='lift', n=100, figsize=(8, 8), title='get_cumlift')

    return auuc_score, causlmlauuc, causlmlqini, treatment_groups

LGBM_Classifier=LGBMClassifier(
        n_estimators=300,  
        max_depth=5,  
        learning_rate=0.03,
        random_state=1023,  
        verbose=-1,
        subsample=0.75,
        is_unbalance=True,
        num_leaves=3,
        colsample_bytree=0.75,
        min_data_in_leaf=30
    )
def tLearner(X_train, y_train, treat_train, solver='classification'):

    print("t_learner_start\n")
    if solver == 'classification':
        t_learner = BaseTClassifier(learner=LGBM_Classifier, control_name='control')
    else:
        t_learner = BaseTRegressor(learner=LGBM_Classifier, control_name='control')
    t_learner.fit(X_train, treat_train, y_train)
    pickle.dump(t_learner, open('best_model/t_learner.pkl', 'wb'))
    return t_learner
multi_treat_t = tLearner(X_train[feature], X_train[LABEL_COL], X_train[TREATMENT_COL])
train_auuc_score, train_causlmlauuc, train_causlmlqini, train_treatment_groups = auuc(X_train[TREATMENT_COL], X_train, multi_treat_t, 't_learner')
test_auuc_score, test_causlmlauuc, test_causlmlqini, test_treatment_groups = auuc(X_test[TREATMENT_COL], X_test, multi_treat_t, 't_learner')
print(train_auuc_score,train_treatment_groups)
print(test_auuc_score,test_treatment_groups)

# multi-treat-tree-model
learner_rf = UpliftRandomForestClassifier(
    n_estimators=10,
    max_depth=8,
    max_features=0.5,
    min_samples_leaf=50,
    min_samples_treatment=10,
    n_reg=10,
    control_name='control',
    evaluationFunction='CTS',
    is_constrained=True
)
learner_rf.fit(X_train[feature], X_train[TREATMENT_COL], X_train[LABEL_COL])
cate_rf_no_p_test = learner_rf.predict(X_test[feature], is_generate_json=False)
pickle.dump(learner_rf, open('learner_rf.pkl', 'wb'))


# nn-model
# dragonnet
def dragon(self, X_train, y_train, treat_train, DragonNet_param):
    print("dragonnet_start\n")
    if self.solver == 'classification':
        return ''
    if isinstance(treat_train.values[0], str):
        treat_names = np.unique(treat_train.values)
        treat_names = np.array(list(set(treat_names) - set([control_name])))
        treat_names = np.sort(treat_names)
        treat_names = np.append(control_name, treat_names)
        treat_name_replace = list(range(0, len(treat_names) + 1, 1))
        treatment_dict = dict(zip(treat_names, treat_name_replace))
        treat_train = [treatment_dict[i] if i in treatment_dict else i for i in treat_train.values]
        if len(treat_names) >= 2:
            treat_onehot = OneHotEncoder()
            treat_train = treat_onehot.fit_transform(pd.DataFrame(treat_train)).toarray()
            self.treat_onehot = treat_onehot
            pickle.dump(self.treat_onehot, open('best_model/treat_onehot.pkl', 'wb'))

    scale_x = StandardScaler()
    scale_y = StandardScaler()
    scale_treat = StandardScaler()
    X_train_scaled = scale_x.fit_transform(X_train.astype('float').values)
    y_train_scaled = scale_y.fit_transform(y_train.values.reshape(-1, 1))
    treat_train_scaled = scale_treat.fit_transform(treat_train)
    self.scale_x = scale_x
    self.scale_y = scale_y
    self.scale_treat = scale_treat

    dragonnet_learner = DragonNet(**DragonNet_param)
    dragonnet_learner.fit(X_train_scaled, treatment=treat_train_scaled, y=y_train_scaled.reshape(-1))
    dragonnet_learner.save('best_model/dragonnet.h5')
    return dragonnet_learner

DragonNet_param = {'neurons_per_layer': 10,
                       'targeted_reg': False,
                       'batch_size': 512,
                       'epochs': 40,
                       'learning_rate': 1e-5,
                       'reg_l2': 30,
                       'verbose': True,
                       }
dn_model = dragon(X_train, y_train, treat_train, DragonNet_param)


##### evaluate #####
treat_test = pd.DataFrame(treatment_test)
treat_test = list(np.where(treat_test['treatment_group'] == 'control', False, True))
test_score = pd.DataFrame(cate_rf_no_p_test, columns=['uplift_score'])
upev_data = UpliftEval(treat_test, np.array(y_test), np.array(test_score), n_bins=100)
auuc_score = upev_data.Q_cgains
cumulative_gain = upev_data.cgains

w_group = pd.concat([pd.DataFrame(treat_test), y_test.reset_index(drop=True), test_score.reset_index(drop=True)], axis=1)
w_group.columns = [treatment_col, label_col, 'uplift_score']
causlmlauuc = AUUC(w_group, outcome_col=label_col, treatment_col=treatment_col, treatment_effect_col='treatment')[0]
causlmlqini = QINI(w_group, outcome_col=label_col, treatment_col=treatment_col, treatment_effect_col='treatment')[0]

# 画图
for group in t_groups:
    plot_all(cate=cate_rf_no_p_test, treatment_groups=t_groups, treatment_test=treatment_test.values,
             y_test=y_test.values, cost_test=cost_test, title="rf-learner-test multi-treatment uplift curve",
             select_treatment_group=group)





