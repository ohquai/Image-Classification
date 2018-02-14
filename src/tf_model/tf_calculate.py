import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np


class SquareLoss:
    def __init__(self):
        self.loss = 0

    def forward(self, y, t):
        """
        根据真实值y和输出值计算loss
        :param y:
        :param t:
        :return:
        """
        self.loss = y - t
        return np.sum(self.loss * self.loss) / self.loss.shape[1] / 2

    def backward(self):
        """
        返回目前模型的loss
        :return:
        """
        return self.loss


class FC:
    def __init__(self, in_num, out_num, lr=0.1):
        """
        初始化隐藏层的节点个数、权重矩阵weight、偏置bias和学习率lr
        :param in_num:
        :param out_num:
        :param lr:
        """
        self._in_num = in_num
        self._out_num = out_num
        self.w = np.random.randn(in_num, out_num)  # weight initialization
        self.b = np.zeros((out_num, 1))
        self.lr = lr

    @staticmethod
    def _sigmoid(in_data):
        """
        定义激活函数
        :param in_data:
        :return:
        """
        return 1 / (1 + np.exp(-in_data))

    def forward(self, in_data):
        self.topVal = self._sigmoid(np.dot(self.w.T, in_data) + self.b)  # 计算输出y
        self.bottomVal = in_data  # 保留这层的输入
        return self.topVal

    def backward(self, loss):
        """
        此处的loss不是真实的 (y-t)^2/2，而是直接的y-t,而y-t也正是真是loss的导数
        :param loss:
        :return:
        """
        residual_z = loss * self.topVal * (1 - self.topVal)  # loss对z的偏导，loss对y的偏导在loss中已经算出，即y-t
        grad_w = np.dot(self.bottomVal, residual_z.T)  # w的梯度，需要用这层的输入x代入计算
        grad_b = np.sum(residual_z)  # bias的梯度
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
        residual_x = np.dot(self.w, residual_z)  # 为了计算前一层的w的梯度，则需要计算对前一层的z的梯度，则需要计算前一层的y即这一层的x的梯度，而对这一层x的梯度，可以通过这一层的z对这一层的w求导得出
        return residual_x


class Net:
    def __init__(self, input_num=2, hidden_num=4, out_num=1, lr=0.1):
        self.fc1 = FC(input_num, hidden_num, lr)
        self.fc2 = FC(hidden_num, out_num, lr)
        self.loss = SquareLoss()

    def train(self, X, y): # X are arranged by col
        for i in range(10000):
            # forward step
            layer1out = self.fc1.forward(X)
            layer2out = self.fc2.forward(layer1out)
            loss = self.loss.forward(layer2out, y)

            # backward step
            layer2loss = self.loss.backward()
            layer1loss = self.fc2.backward(layer2loss)
            saliency = self.fc1.backward(layer1loss)
        layer1out = self.fc1.forward(X)
        layer2out = self.fc2.forward(layer1out)
        print('X={0}'.format(X))
        print('t={0}'.format(y))
        print('y={0}'.format(layer2out))


class ConvLayer:
    def __init__(self, in_channel, out_channel, kernel_size):
        self.w = np.random.randn(in_channel, out_channel, kernel_size, kernel_size)
        self.b = np.zeros((out_channel))

    def gabor_fn(self, sigma, theta, Lambda, psi, gamma):
        sigma_x = sigma
        sigma_y = float(sigma) / gamma
        (y, x) = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
            2 * np.pi / Lambda * x_theta + psi)
        return gb

    def _relu(self, x):
        x[x < 0] = 0
        return x

    def conv2(self, X, k):
        x_row, x_col = X.shape
        k_row, k_col = k.shape
        ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
        ret = np.empty((ret_row, ret_col))
        for y in range(ret_row):
            for x in range(ret_col):
                sub = X[y: y + k_row, x: x + k_col]
                ret[y, x] = np.sum(sub * k)
        return ret

    def forward(self, in_data):
        # assume the first index is channel index
        in_channel, in_row, in_col = in_data.shape
        out_channel, kernel_row, kernel_col = self.w.shape[1], self.w.shape[2], self.w.shape[3]
        self.top_val = np.zeros((out_channel, in_row - kernel_row + 1, in_col - kernel_col + 1))
        for j in range(out_channel):
            for i in range(in_channel):
                self.top_val[j] += self.conv2(in_data[i], self.w[i, j])
            self.top_val[j] += self.b[j]
            self.top_val[j] = self._relu(self.top_val[j])
        return self.top_val


def image_transform():
    mat = cv2.imread('me.jpg', 0)
    row, col = mat.shape
    print("shape of img is ({0},{1})".format(row, col))
    in_data = mat.reshape(1, row, col)
    in_data = in_data.astype(np.float) / 255
    plt.imshow(in_data[0], cmap='Greys_r')
    plt.show()

    meanConv = ConvLayer(1, 1, 5)
    meanConv.w[0, 0] = np.ones((5, 5)) / (5 * 5)
    mean_out = meanConv.forward(in_data)
    plt.imshow(mean_out[0], cmap='Greys_r')
    plt.show()

    sobelConv = ConvLayer(1, 1, 3)
    sobelConv.w[0, 0] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_out = sobelConv.forward(in_data)
    plt.imshow(sobel_out[0], cmap='Greys_r')
    plt.show()

    gaborConv = ConvLayer(1, 1, 3)
    print(gaborConv.gabor_fn(2, 0, 0.3, 0, 2))
    gaborConv.w[0, 0] = gaborConv.gabor_fn(2, 0, 0.3, 1, 2)
    gabor_out = gaborConv.forward(in_data)
    plt.imshow(gabor_out[0], cmap='Greys_r')
    plt.show()


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([[0], [0], [0], [1]]).T

    net = Net(2, 4, 1, 0.1)
    net.train(X, y)

    image_transform()
