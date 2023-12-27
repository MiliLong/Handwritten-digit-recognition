import cupy as cp
import time
from tqdm import tqdm
import json
import pickle
from DNN.DNN_BP.net import DNN
from PIL import Image, ImageTk

import tkinter as tk
class TrainDNN(DNN):
    def __init__(self, sample_config):
        # 初始化
        super(TrainDNN, self).__init__(sample_config)
        self.savemodels_file = self.sample_config["savemodels_file"]

    def train(self):
        start_time = time.time()
        train_count = int(len(self.train_images))
        epochs = self.sample_config["epochs"]

        for i in range(epochs):
            for count in tqdm(range(train_count), desc="The {0}th epoch".format(i+1), unit="iteration", bar_format="{l_bar}{bar}{r_bar}", colour='blue'):

                image = self.train_images[count]
                target = cp.zeros(self.dist["output_num"]) + 0.001
                target[self.train_targets[count]] = 0.999

                self.backward_propagation(image, self.forward_propagation(image), target)

        end_time = time.time()
        print("训练完成, 用时{0}秒".format(round(end_time - start_time, 5)))

        self.save(self.dist)

    def test(self):

        with open(self.savemodels_file, 'rb') as pkl_file:
            self.dist = pickle.load(pkl_file)

        print("开始测试模型")

        right_count = cp.zeros(10)
        expect_count = cp.zeros(10)
        test_count = len(self.test_images)

        for i in range(test_count):
            expect_count[self.test_targets[i]] += 1

        for count in tqdm(range(test_count), desc="Testing", unit="iteration", bar_format="{l_bar}{bar}{r_bar}", colour='green'):
            image = self.test_images[count]
            target = self.test_targets[count]

            hidden_acts = self.forward_propagation(image)
            output_act = hidden_acts[-1]

            if cp.argmax(output_act) == cp.array(target):
                right_count[target] += 1

        text = []

        text.append("训练数据总数: {0} ".format(self.dist["samples_num"]))
        text.append("神经网络结构: {0}".format(' '.join([str(i) for i in self.dist["hidden_nodes"]])))

        right_sum = right_count.sum()
        text.append("测试数据总数: {0}".format(test_count))
        text.append("正确结果数: {0}".format(right_sum))

        rate = right_sum / test_count
        text.append("识别率: {0}%".format(rate * 100))
        text.append("期望输出中每个数字的数量: {0}".format(' '.join([str(int(i)) for i in expect_count.get()])))
        text.append("每个数字正确结果的数量: {0}".format(' '.join([str(int(i)) for i in right_count.get()])))

        rate_arr = right_count / expect_count
        text.append("每个数字的识别率: {0}".format(' '.join(["{0}%".format(round(i * 100, 2)) for i in rate_arr.get()])))

        text = '\n'.join(text)

        return text

    def save(self, dist):
        with open(self.savemodels_file, 'wb') as pkl_file:
            pickle.dump(dist, pkl_file)

        return dist

    def run(self):

        with open(self.savemodels_file, 'rb') as pkl_file:
            self.dist = pickle.load(pkl_file)

        image = Image.open(self.sample_config["test_image_file"])
        image = image.resize((28, 28)).convert("L")

        # 进行预测
        hidden_acts = self.forward_propagation(cp.array(list(image.getdata())) / 256.0)
        output_act = hidden_acts[-1]

        output_probability = [str(round(i, 4)) for i in output_act.get()]
        output_probability = ' '.join([str(i) + ": " + output_probability[i] for i in range(len(output_probability))])

        # 用tkinter显示图片和结果
        root = tk.Tk()
        root.title("识别结果")
        root.geometry("625x375")

        img = ImageTk.PhotoImage(image.resize((300, 300)))
        label_img = tk.Label(root, image=img)
        label_img.pack()

        label_text = tk.Label(root, text= "预测值: " + str(cp.argmax(output_act)) + "\n" + "概率 " + output_probability)
        label_text.pack()

        root.mainloop()

def main():
    # 读取配置文件
    with open("../config/config.json", "r", encoding="utf-8") as f:
        sample_config = json.load(f)

    # cp.cuda.Device(0).use()

    dnn = TrainDNN(sample_config)
    dnn.train()
    print(dnn.test())
    dnn.run()

if __name__ == "__main__":
    main()