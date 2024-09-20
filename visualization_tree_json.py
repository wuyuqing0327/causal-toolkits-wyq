import json

import numpy as np


def search_node(root):
    """
    按层次遍历二叉树
    :param root:根节点
    :return:层次遍历的节点列表
    """
    queue = [root]
    out_list = []
    i = 0
    while len(queue) > 0 and root:
        if queue[0] == -1:
            out_list.append([-1, None])
            queue.pop(0)
            continue
        else:
            out_list.append([i, queue[0]])
        i = i + 1
        """
        这里情况比较特殊，不存在某次分裂只存在左或者右节点的情况，
        因此当发现某个节点不存在左或者右节点的情况，我们通常采取在队列中添加[-1,-1]表示该节点左右子节点为空
        """
        # 存在左右子节点
        if queue[0].trueBranch and queue[0].falseBranch:
            queue.append(queue[0].trueBranch)
            queue.append(queue[0].falseBranch)
        else:
            queue.append(-1)  # 表示当前节点未分裂，-1，-1提示该节点无左右节点
            queue.append(-1)

        queue.pop(0)
    return out_list


def sorted_dict(data):
    """
    字典排序取值
    :param data:字典
    :return:排序后的字典值(这里就是treatment_0到treatment_k排序后对应的lift值)
    """
    if data is None:
        return
    data_keys = sorted(data)
    res = []
    control_value = data['control']
    for key in data_keys:
        if key != 'control':
            res.append(data[key] - control_value)
    return res


def handle_result(root):
    """
    处理树的结构
    :param root:
    :return:
    """
    node_lists = search_node(root)
    child_nodes = []
    split_features = []
    split_values = []
    treatment_values = []
    for node in node_lists:
        if node[0] == -1 and node[1] is None:
            # print(node[0], 0, -1, [])  # (node_num, split_feature, split_value, treatment)
            child_nodes.append(0)
        elif node[1].col == -1:
            # print(node[0], 0, -1, sorted_dict(node[1].results))
            child_nodes.append(node[0])
            split_features.append(0)
            split_values.append(-1)
            treatment_values.append(sorted_dict(node[1].results))
        else:
            # print(node[0], node[1].col, node[1].value, [])
            child_nodes.append(node[0])
            split_features.append(node[1].col)
            split_values.append(node[1].value)
            treatment_values.append([])

    child_nodes = child_nodes[1:]
    left_child_nodes = child_nodes[0::2]
    right_child_nodes = child_nodes[1::2]
    return [[left_child_nodes, right_child_nodes], split_features, split_values, treatment_values]


def merge(tree_list):
    """"
    整合所有树的结果
    :param tree_list 树列表
    :return 子节点,划分特征,划分值,uplift值
    """
    child_nodes_list = []
    split_features_list = []
    split_values_list = []
    treatment_values_list = []
    for tree in tree_list:
        res = handle_result(tree)
        child_nodes_list.append(res[0])
        split_features_list.append(res[1])
        split_values_list.append(res[2])
        treatment_values_list.append(res[3])
    return child_nodes_list, split_features_list, split_values_list, treatment_values_list


class NpEncoder(json.JSONEncoder):
    """
    JSON编码器
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def rf_generate_json(tree_list, file_path='model_structure.json'):
    """
    根据树的结构信息生成JSON文件
    :param file_path: 文件路径
    :param tree_list: 树列表
    :return:无
    """
    result = merge(tree_list)
    data = {
        "child_nodes": result[0],
        "split_vars": result[1],
        "split_values": result[2],
        "pv_values": result[3]
    }
    json_data = json.dumps(data, indent=4, separators=(',', ': '), cls=NpEncoder)
    with open(file_path, 'w+') as f:
        f.write(json_data)
    f.close()
    print("已生成json文件")
