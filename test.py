import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import yaml
from feeders import tools

# --- 1. 导入你的模型和 Feeder ---
# 确保你的项目结构能找到这些文件
from model.ctrgcn import Model  # 假设这是你的模型文件
from feeders.feeder_ntu import Feeder_LT_nosample  # 我们需要它来进行数据增强

# --- 2. 超参 & 设备 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG_PATH = 'config/nturgbd-cross-subject/long_tailed.yaml'  # <-- 使用你最好模型的配置文件
TEACHER_WEIGHTS_PATH = '100.pt'  # <-- 你最好的模型权重 (假设在当前目录)
STUDENT_SAVE_PATH = 'student4.0.pt'

# 蒸馏超参
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-5
TEMP = 4.0

# --- 3. 加载配置文件和数据 ---
with open('./config/nturgbd-cross-subject/test.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
test_feeder_args = config['test_feeder_args']
augment_helper = Feeder_LT_nosample(**test_feeder_args)
npz_path = test_feeder_args['data_path']
x_test_raw = np.load(npz_path)['x_test']

# --- 4. 生成或加载“标准姿态”的软标签 ---
SOFT_LABELS_PATH = 'soft_labels_standard_pose.pt'


def generate_soft_labels(raw_data, config, weights_path, helper):
    print("\n--- Generating soft labels on STANDARD, NON-AUGMENTED data ---")
    model_args = config['model_args']
    teacher = Model(**model_args).to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    teacher.load_state_dict(state_dict)
    teacher.eval()

    processed_data = []
    for i in tqdm(range(len(raw_data)), desc="Preprocessing data for teacher"):
        sample = raw_data[i]
        T, _ = sample.shape
        sample = sample.reshape((T, 2, 25, 3)).transpose(3, 0, 2, 1)
        valid_num = np.sum(sample.sum(0).sum(-1).sum(-1) != 0)
        # --- 使用确定的中心裁剪，不进行随机旋转 ---
        sample = tools.valid_crop_resize(sample, valid_num, p_interval=[0.95], window=helper.window_size)
        processed_data.append(sample)

    x_test_standard = torch.from_numpy(np.stack(processed_data)).float()
    loader = DataLoader(TensorDataset(x_test_standard), batch_size=BATCH_SIZE, shuffle=False)

    all_soft_logits = []
    with torch.no_grad():
        for (x_batch,) in tqdm(loader, desc='Teacher inference'):
            logits = teacher(x_batch.to(DEVICE))
            all_soft_logits.append(logits.cpu())

    soft_labels = torch.cat(all_soft_logits, dim=0)
    torch.save(soft_labels, SOFT_LABELS_PATH)
    print(f"Soft labels for standard poses saved to {SOFT_LABELS_PATH}")
    return soft_labels


if os.path.exists(SOFT_LABELS_PATH):
    print(f"Loading existing soft labels from {SOFT_LABELS_PATH}")
    soft_labels = torch.load(SOFT_LABELS_PATH)
else:
    soft_labels = generate_soft_labels(x_test_raw, config, TEACHER_WEIGHTS_PATH, augment_helper)


# --- 5. 蒸馏训练 ---
def distill(config, teacher_weights, raw_data, soft_labels, helper):
    print("\n--- Starting distillation training on AUGMENTED data ---")
    model_args = config['model_args']
    student = Model(**model_args).to(DEVICE)

    print("Initializing student with teacher's weights for finetuning.")
    state_dict = torch.load(teacher_weights, map_location=DEVICE)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    student.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)

    distill_loader = DataLoader(TensorDataset(torch.arange(len(raw_data))), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0
        pbar = tqdm(distill_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for (indices,) in pbar:
            soft_batch = soft_labels[indices].to(DEVICE)

            processed_batch = []
            for idx in indices:
                sample = raw_data[idx.item()]
                T, _ = sample.shape
                sample = sample.reshape((T, 2, 25, 3)).transpose(3, 0, 2, 1)
                valid_num = np.sum(sample.sum(0).sum(-1).sum(-1) != 0)

                # --- 学生看到的是带随机性的数据 ---
                sample = tools.valid_crop_resize(sample, valid_num, helper.p_interval, helper.window_size)
                if helper.random_rot:
                    sample = tools.random_rot(sample)
                processed_batch.append(torch.from_numpy(sample))

            x_augmented = torch.stack(processed_batch).float().to(DEVICE)
            student_logits = student(x_augmented)

            loss = F.kl_div(
                F.log_softmax(student_logits / TEMP, dim=1),
                F.softmax(soft_batch / TEMP, dim=1),
                reduction='batchmean'
            ) * (TEMP ** 2)

            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        print(f'Epoch {epoch + 1} average distillation loss: {total_loss / len(distill_loader):.4f}')

    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f'Distillation done! Model saved to -> {STUDENT_SAVE_PATH}')


# --- 主入口 ---
if __name__ == '__main__':
    distill(config, TEACHER_WEIGHTS_PATH, x_test_raw, soft_labels, augment_helper)