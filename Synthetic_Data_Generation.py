import torch
import numpy as np
import torch.nn as nn


# input_dimension = d

def synthetic_data(c, person, input_dimension=100, N=200, m=5):
    # 利用QR分解随机矩阵生成正交基
    random_matrix = np.random.rand(input_dimension, 2)
    q, r = np.linalg.qr(random_matrix)
    u1 = q[:, 0]  # 100*1
    u2 = q[:, 1]
    # w1, w2
    w1 = c * u1
    w2 = c * (person * u1 + np.sqrt(1 - person ** 2) * u2)
    # 准备工作完成，接下来重复进行获得n个数据, m用来控制相位
    alpha = np.random.uniform(0, 2, size=m)
    beta = np.random.uniform(0, 2*np.pi, size=m)
    x = np.random.multivariate_normal(mean=np.zeros(input_dimension), cov=np.eye(input_dimension), size=N) #(N, 100)
    sinusoidal_part1 = np.zeros(N)
    sinusoidal_part2 = np.zeros(N)
    w1_transpose_x = x@w1
    w2_transpose_x = x@w2
    for j in range(m):
        sinusoidal_part1 += np.sin(alpha[j] * w1_transpose_x + beta[j])
        sinusoidal_part2 += np.sin(alpha[j] * w2_transpose_x + beta[j])
    y1 = w1_transpose_x + sinusoidal_part1 + np.random.normal(0,0.01, size=N)
    y2 = w2_transpose_x + sinusoidal_part2 + np.random.normal(0,0.01, size=N)
    y1 = y1.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)
    dataset1 = np.hstack((x, y1))
    dataset2 = np.hstack((x, y2))
    return x, y1, y2

if __name__ == '__main__':
    dataset1, dataset2, _ = synthetic_data(c=0.3, person=0.5, input_dimension=100, N=1000, m = 5)
    print(dataset1[:5])
    print(dataset2[:5])
