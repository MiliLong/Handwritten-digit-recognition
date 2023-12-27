import cupy as cp
import pandas as pd

# cp.fuse()装饰器可以将多个函数融合为一个函数，从而减少函数调用的开销, 相较于numpy提高了25%左右的性能,
# 相较于未优化的cupy提高了近50%的性能(由于GPU性能波动，使得其性能最低跌至CPU的50%，上限仅仅为CPU的
# 90%，这里假定50%提高，实际上可能会由于波动有上下浮涨)

@cp.fuse()
# 激活函数
def get_act(x):
    return 1 / (1 + cp.exp(-x))
@cp.fuse()
# 激活函数的导数
def get_act_derivative(x):
    return x * (1 - x)

class DNN():
    def __init__(self, sample_config):
        # 加载数据
        self.sample_config = sample_config
        self.train_images = cp.array(pd.read_csv(self.sample_config["train_file"]).values.tolist()) / 256.0
        self.train_targets = pd.read_csv(self.sample_config["train_labels_file"]).values
        self.test_images = cp.array(pd.read_csv(self.sample_config["test_file"]).values.tolist()) / 256.0
        self.test_targets = pd.read_csv(self.sample_config["test_labels_file"]).values
        self.pkl_file = self.sample_config["savemodels_file"]
        self.dist = self.initialize_parameters()

    # 激活函数
    def get_act(self, x):
        return 1 / (1 + cp.exp(-x))

    # 激活函数的导数
    def get_act_derivative(self, x):
        return x * (1 - x)

    # 不调用原因, 调用对象本身函数无法被@cp.fuse()优化, 开销较大

    # 初始化权重和偏置
    def initialize_parameters(self):
        # 配置神经网络参数
        samples_num, input_num = self.train_images.shape
        output_num = self.sample_config["output_num"]
        hidden_nodes = list(map(int, self.sample_config["hidden_nodes"].split(', ')))
        hidden_num = len(hidden_nodes)
        learn_rate = self.sample_config["learn_rate"]
        # 初始化权值和偏置
        weights = [i for i in range(hidden_num)]
        weights[0] = 0.2 * cp.random.random((input_num, hidden_nodes[0])) - 0.1

        for i in range(1, hidden_num):
            weights[i] = 0.2 * cp.random.random((hidden_nodes[i-1], hidden_nodes[i])) - 0.1
        offsets = [cp.zeros(hidden_nodes[i]) for i in range(hidden_num)]

        dist = {
            "samples_num": samples_num,
            "input_num": input_num,
            "output_num": output_num,
            "hidden_num": hidden_num,
            "hidden_nodes": hidden_nodes,
            "weights": weights,
            "offsets": offsets,
            "learn_rate": learn_rate,
        }

        return dist

    # 前向传播
    def forward_propagation(self, image):
        hidden_values = [0] * self.dist["hidden_num"]
        hidden_acts = [0] * self.dist["hidden_num"]
        hidden_values[0] = cp.dot(image, self.dist["weights"][0]) + self.dist["offsets"][0]
        hidden_acts[0] = get_act(hidden_values[0])
        for i in range(1, self.dist["hidden_num"]):
            hidden_values[i] = cp.dot(hidden_acts[i-1], self.dist["weights"][i]) + self.dist["offsets"][i]
            hidden_acts[i] = get_act(hidden_values[i])

        return hidden_acts

    # 反向传播
    def backward_propagation(self, image, hidden_acts, target):
        # 误差反传
        e = target - hidden_acts[-1]
        deltas = [0] * self.dist["hidden_num"]
        deltas[-1] = e * get_act_derivative(hidden_acts[-1])
        for i in range(self.dist["hidden_num"] - 2, -1, -1):
            deltas[i] = get_act_derivative(hidden_acts[i]) * cp.dot(self.dist["weights"][i+1], deltas[i+1])

        # 调整权值和偏置
        self.dist["weights"][-1] += self.dist["learn_rate"] * cp.outer(hidden_acts[-2], deltas[-1])
        for i in range(self.dist["hidden_num"] - 2, -1, -1):
            self.dist["weights"][i] += self.dist["learn_rate"] * cp.outer(hidden_acts[i-1] if (i - 1) != -1 else image, deltas[i])

        for i in range(self.dist["hidden_num"]):
            self.dist["offsets"][i] += self.dist["learn_rate"] * deltas[i]