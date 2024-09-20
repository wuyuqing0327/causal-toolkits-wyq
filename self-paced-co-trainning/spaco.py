import numpy as np
import pandas as pd
 
 
def train_predict(model,X_data,y_data,unlabel_data):
    model.fit(X_data,y_data)
    pred_prob = model.predict_proba(unlabel_data)
    # print(pred_prob)
    return pred_prob
 
 
def sel_idx(score,y_data,λ=0.1):
    y = y_data
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    count_per_class = [len(y_data[y_data == c]) for c in clss]
    print('count_per_class:', count_per_class)
    pred_y = np.argmax(score, axis=1)  # 对unlabel数据的预测值,eg：对于每一组预测概率如果0列值>1列的值则此样本预测值为1
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]  # 对应预测label为每一类 的索引
        # print('indi:',indices.shape)
        cls_score = score[indices,cls]
        idx_sort = np.argsort(cls_score)  # 每一类分别预测概率值排序（升序）并取索引
        add_num = min(int(np.ceil(count_per_class[cls] * λ)),indices.shape[0])  # 每次添加预测样本数,count_per_class[cls] * λ
        add_indices[indices[idx_sort[-add_num:]]] = 1  # 对应位置设为1，返回bool类型
    return add_indices.astype('bool')
 
 
def update_train_untrain(add_id,X_data,y_data, untrain_data, pred_y):
    new_train_data = pd.concat([X_data,untrain_data[add_id]])
    new_train_y = np.concatenate([y_data, pred_y[add_id]])
    return new_train_data, new_train_y
 
 
def update_label_data(unlabel_data,y_data,pred_probs,pred_y,num):
    score=pred_probs[0]+pred_probs[1]  # 综合俩模型的预测值
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y_data)
    label_num=0
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]  # 对应预测label为每一类 的索引
        label_num_ratio=len(y_data[y_data==cls])/len(y_data)
        cls_score = score[indices, cls]
        idx_sort = np.argsort(cls_score)  # 每一类分别预测概率值排序（升序）并取索引
        label_num = int(num*label_num_ratio) if label_num==0 else num-label_num
        print('第'+str(cls)+'类打标签样本数：', label_num)
        add_indices[indices[idx_sort[-label_num:]]] = 1  # 对应位置设为1，返回bool类型
    add_id=add_indices.astype('bool')
    X_unlabel_data = unlabel_data[add_id]
    y_unlabel_data=pred_y[add_id]
    return X_unlabel_data, y_unlabel_data
 
 
# 核心代码如下
import numpy as np
import pandas as pd
from functions import train_predict,sel_idx,update_train_untrain,update_label_data
 
 
def spaco(models,X_data,y_data,unlabel_data,max_round=5,λ=0.05,gamma=0.3,label_num=10000):
    print("starting---")
    pred_probs = []
    add_ids = []
    models = models
    num_view = len(models)
    for view in range(num_view):
        model = models[view]
        pred_probs.append(train_predict(model,X_data,y_data,unlabel_data))
        add_ids.append(sel_idx(pred_probs[view], y_data, λ)) # 俩模型各自激活的unlabel data的bool值
    pred_y = np.argmax(sum(pred_probs), axis=1)
 
    iter_step = max_round
    print("round---")
    for step in range(iter_step):
        print("第"+str(step)+"次更新样本：")
        for view in range(num_view):
            # update v_view
            ov = add_ids[1 - view]
            pred_probs[view][ov,pred_y[ov]] += gamma  # 对应另一view中激活位置的prob更新
            add_id = sel_idx(pred_probs[view],y_data,λ)
 
            # update w_view
            new_train_data,new_train_y= update_train_untrain(
                add_id,X_data,y_data, unlabel_data, pred_y)
            model.fit(new_train_data,new_train_y)
 
            # update y
            pred_probs[view] = model.predict_proba(unlabel_data)
            pred_y = np.argmax(pred_probs[0]+pred_probs[1],axis=1)
 
            # udpate v_view for next view
            λ += 0.05
            add_ids[view] = sel_idx(pred_probs[view], y_data, λ)
 
    return update_label_data(unlabel_data, y_data, pred_probs, pred_y, label_num)
 
 
if __name__ == '__main__':
    # 载入模型eg：xgboost、randomforest
    from sklearn.ensemble import RandomForestClassifier
    #import xgboost as xgb
    # 设置两模型view
    models = [RandomForestClassifier(), RandomForestClassifier()]
 
    # 加载数据
    data = pd.read_csv('data/all_feature_data.csv')
    unlabel_data = pd.read_csv('data/unlabel_data.csv')
    # 数据预处理
    unlabel_data = unlabel_data.fillna(0)
    unlabel_data['p_face_score'].fillna(data['p_face_score'].mean(), inplace=True)
    for i in ['os_type', 'channel_id']:
        unlabel_data[i].fillna(data[i].mode(), inplace=True)
    features = ['est_dis', 'setup_hour', 'is_county_type', 'p_weighted_awn_avg', 'asr_middle_wandan_rate_sum',
                'mw_asr_middle_wandan_rate_7days_sum', 'sex_asr_wandan_rate_sum', 'mw_sex_asr_wandan_rate_sum',
                'passenger_num', 'asr_middle_wandan_cnt_7days_sum']
    X_data = data[features]
    X_data = X_data.fillna(0)
    y_data = data['label']
    print("训练数据集shape：", X_data.shape, y_data.shape)
    # unlabel_initial_data = unlabel_data
    print("无标签数据集shape：", unlabel_data.shape)
    unlabel_data = unlabel_data[features]
    X_unlabel_data, y_unlabel_data = spaco(models, X_data, y_data, unlabel_data, max_round=10, λ=0.05, gamma=0.3,
                                           label_num=8000)

'''
代码说明如下：
# Self-paced two-view Co-training for classification

## Examples
you can run with：
```shell
python main.py 
```
## 说明

- spaco 实现对无标签数据打标签功能
- train_predict 训练并输出预测分数
- sel_idx 获得每一类激活的unlabel_data的bool值
- update_train_untrain 更新训练数据（加入激活后的labeled_data）
- update_label_data 综合俩模型的预测值，按类别比例获取一定数量的正负labeled_data


Please cite spaco in your publications if it helps your research:  
@inproceedings{ma2017self,  
  title={Self-Paced Co-training},   
  author={Ma, Fan and Meng, Deyu and Xie, Qi and Li, Zina and Dong, Xuanyi},  
  booktitle={International Conference on Machine Learning},  
  pages={2275--2284},  
  year={2017}  
}
'''