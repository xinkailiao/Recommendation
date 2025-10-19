# run.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 从我们自己的文件中导入函数和类
from Synthetic_Data_Generation import synthetic_data
from MMOE import MMOE

# ===================================================================
# 实验主流程
# ===================================================================
if __name__ == '__main__':
    # --- 超参数设置 ---
    # 数据参数
    INPUT_DIM = 100
    N_SAMPLES = 2000
    C_PARAM = 0.3
    M_PARAM = 5

    # 模型参数
    N_EXPERTS = 8
    N_TASKS = 2
    HIDDEN_DIM = 64
    OUTPUT_DIM = 1  # 每个任务都是回归任务，输出1个值, 当然也可做分类任务（加上sigmoid）

    # 训练参数
    EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001

    # --- 定义要测试的相关度列表 ---
    correlations_to_test = [0.0, 0.2, 0.5, 0.8, 1.0]
    results = {}

    for person_corr in correlations_to_test:
        print(f"\n{'=' * 20} 测试任务相关度 (person) = {person_corr} {'=' * 20}")

        # 1. 生成数据
        X_np, y1_np, y2_np = synthetic_data(
            c=C_PARAM,
            person=person_corr,
            input_dimension=INPUT_DIM,
            N=N_SAMPLES,
            m=M_PARAM
        )
        print(f"生成数据: X shape={X_np.shape}, y1 shape={y1_np.shape}, y2 shape={y2_np.shape}")

        # 2. 准备PyTorch数据集
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y1_tensor = torch.tensor(y1_np, dtype=torch.float32)
        y2_tensor = torch.tensor(y2_np, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y1_tensor, y2_tensor)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 3. 初始化模型、损失函数和优化器
        model = MMOE(
            n_expert=N_EXPERTS,
            n_gate=N_TASKS,
            hidden_dimensions=HIDDEN_DIM,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM
        )

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 4. 训练模型
        model.train()
        final_epoch_loss = 0.0
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for x_batch, y1_batch, y2_batch in data_loader:
                optimizer.zero_grad()

                task1_out, task2_out = model(x_batch)

                loss1 = loss_fn(task1_out, y1_batch)
                loss2 = loss_fn(task2_out, y2_batch)
                total_loss = loss1 + loss2

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            final_epoch_loss = epoch_loss / len(data_loader)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], 平均损失: {final_epoch_loss:.4f}")

        results[person_corr] = final_epoch_loss
        print(f"相关度 {person_corr} 的最终训练损失: {final_epoch_loss:.4f}")

    # --- 5. 打印最终结果总结 ---
    print(f"\n{'=' * 20} 实验结果总结 {'=' * 20}")
    for corr, loss in results.items():
        print(f"任务相关度 = {corr:.1f}, 最终平均损失 = {loss:.4f}")

### 如何运行
