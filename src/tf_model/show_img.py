import cv2
from matplotlib import pyplot as plt
import numpy as np


def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    (y, x) = np.meshgrid(np.arange(-1,2), np.arange(-1,2))
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

class ConvLayer:
    def __init__(self, in_channel, out_channel, kernel_size):
        self.w = np.random.randn(in_channel, out_channel, kernel_size, kernel_size)
        self.b = np.zeros((out_channel))

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


if __name__ == "__main__":
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

    print(gabor_fn(2, 0, 0.3, 0, 2))
    gaborConv = ConvLayer(1, 1, 3)
    gaborConv.w[0, 0] = gabor_fn(2, 0, 0.3, 1, 2)
    gabor_out = gaborConv.forward(in_data)
    plt.imshow(gabor_out[0], cmap='Greys_r')
    plt.show()
