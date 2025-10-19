import numpy as np
import torch
import torch.nn as nn


class SharedBottom(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim1, output_dim2):
        super().__init__()
        # 底层继承模型
        self.shared_bottom = nn.Sequential(nn.Linear(input_dim, hidden1_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden1_dim, hidden2_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden2_dim, hidden3_dim),
                                           nn.ReLU())
        #Tower 1
        self.Tower1 = nn.Sequential(nn.Linear(hidden3_dim, hidden2_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden2_dim, output_dim1))

        #Tower 2

        self.Tower2 = nn.Sequential(nn.Linear(hidden3_dim, hidden2_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden2_dim, output_dim2))

    def forward(self, x):
        shared_output = self.shared_bottom(x)

        #任务1
        task1_output = self.Tower1(shared_output)
        task2_output = self.Tower2(shared_output)

        return task1_output, task2_output




