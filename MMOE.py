import numpy as np
import torch
import torch.nn as nn

class MMOE(nn.Module):
    def __init__(self, n_expert, n_gate, hidden_dimensions, input_dim, output_dim):
        super(MMOE, self).__init__()
        self.n_expert = n_expert
        self.n_gate = n_gate
        #所有模型默认为单层16的网络（可以自主替换）
        #门网络
        self.gate1 = nn.Sequential(nn.Linear(input_dim, self.n_expert),
                                  nn.Softmax())
        self.gate2 = nn.Sequential(nn.Linear(input_dim, self.n_expert),
                                   nn.Softmax())
        #塔模型
        self.tower1 = nn.Sequential(nn.Linear(hidden_dimensions, output_dim))
        self.tower2 = nn.Sequential(nn.Linear(hidden_dimensions, output_dim))
        #专家模型集合
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dimensions)) for _ in range(self.n_expert)
        ])


    def forward(self, x):
        experts_out = [expert(x) for expert in self.experts]

        experts_output_tensor = torch.stack(experts_out) #(n_experts, bach_size, hidden_dimensions)
        experts_output_tensor = experts_output_tensor.permute(1, 0, 2) # (bach_size, n_expert, hidden_dimensions)

        #对专家模型的输出进行加权求和
        gate1_out = self.gate1(x) # (bach_size, n_expert)
        gate2_out = self.gate2(x)

        gating1_output = gate1_out.unsqueeze(-1)
        gating2_output = gate2_out.unsqueeze(-1)
        # 现在 shape 是 (bach_size, n_expert, 1)
        final_output1 = torch.sum(experts_output_tensor * gating1_output, dim=1)
        final_output2 = torch.sum(experts_output_tensor * gating2_output, dim=1)
        # final_output shape 会是 (bach_size, hidden_dimensions)

        tower1_out = self.tower1(final_output1)
        tower2_out = self.tower2(final_output2)

        return tower1_out, tower2_out


if __name__ == '__main__':
    print("测试MMOE模型模块...")
    # 模拟输入
    dummy_input = torch.randn(10, 100)  # (batch_size, input_dim)
    model = MMOE(n_expert=8, n_gate=2, hidden_dimensions=32, input_dim=100, output_dim=1)

    # 测试前向传播
    outputs = model(dummy_input)
    print(f"模型有 {len(outputs)} 个输出。")
    print("任务1的输出 shape:", outputs[0].shape)
    print("任务2的输出 shape:", outputs[1].shape)



