import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import yaml

# --- 1. 导入你的模型和 Tools ---
from model.ctrgcn import Model
from feeders import tools

# --- 2. 超参 & 设备 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG_PATH = './config/nturgbd-cross-subject/test.yaml'  # 替换为你的配置文件路径
TEACHER_WEIGHTS_PATH = './100.pt'  # 替换为你的最佳老师模型路径
STUDENT_SAVE_PATH = 'student111.pt'  # 新的输出文件名

# 蒸馏超参
BATCH_SIZE = 64
EPOCHS = 30  # 增加轮数，让模型有更充分的时间学习混合损失
LR = 5e-6  # 对于精细微调，一个更小的学习率可能更好
TEMP = 2.5  # 可以微调的温度
ALPHA = 0.9  # 蒸馏损失的权重，一个非常关键的超参数

# --- 3. 加载配置和数据 ---
print("--- Loading Config and Data ---")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 加载原始测试数据
npz_path = config['test_feeder_args']['data_path']
raw_data = np.load(npz_path)
x_test_raw = raw_data['x_test']
y_test_one_hot = raw_data['y_test']
# 将 one-hot 标签转换为索引标签 (硬标签)
y_test_labels = torch.from_numpy(np.where(y_test_one_hot > 0)[1])
print(f'Official raw test samples loaded: {len(x_test_raw)}')

# --- 4. 准备蒸馏所需的数据集 ---
# 我们需要三个东西：增强后的输入，老师的软标签，真实的硬标签
AUGMENTED_DATA_PATH = 'x_test_augmented.npy'
SOFT_LABELS_PATH = 'soft_labels.pt'


# 封装数据预处理和增强
def preprocess_and_augment_data(raw_data, p_interval, window_size, use_random_rot):
    print("\n--- Preprocessing and Augmenting entire test set ONCE ---")
    processed_data_list = []
    for i in tqdm(range(len(raw_data)), desc="Augmenting data"):
        sample = raw_data[i]
        T, _ = sample.shape
        sample = sample.reshape((T, 2, 25, 3)).transpose(3, 0, 2, 1)
        valid_num = np.sum(sample.sum(0).sum(-1).sum(-1) != 0)
        sample = tools.valid_crop_resize(sample, valid_num, p_interval, window_size)
        if use_random_rot:
            sample = tools.random_rot(sample)
        processed_data_list.append(sample)

    final_data = np.stack(processed_data_list, axis=0)
    np.save(AUGMENTED_DATA_PATH, final_data)
    print(f"Augmented data saved to {AUGMENTED_DATA_PATH}. Shape: {final_data.shape}")
    return torch.from_numpy(final_data).float()


# 封装软标签生成
def generate_soft_labels(data_to_predict, config, weights_path):
    print("\n--- Generating soft labels from a given dataset ---")
    model_args = config['model_args']
    teacher = Model(**model_args).to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    teacher.load_state_dict(state_dict)
    teacher.eval()
    loader = DataLoader(TensorDataset(data_to_predict), batch_size=BATCH_SIZE, shuffle=False)
    all_soft_logits = []
    with torch.no_grad():
        for (x_batch,) in tqdm(loader, desc='Teacher inference'):
            logits = teacher(x_batch.to(DEVICE))
            all_soft_logits.append(logits.cpu())
    soft_labels = torch.cat(all_soft_logits, dim=0)
    torch.save(soft_labels, SOFT_LABELS_PATH)
    print(f"Soft labels saved to {SOFT_LABELS_PATH}")
    return soft_labels


# 创建“标准姿态”数据，用于生成软标签
p_interval_std = [0.95]  # 确定的中心裁剪
window_size_std = config['test_feeder_args']['window_size']
x_test_standard = preprocess_and_augment_data(x_test_raw, p_interval_std, window_size_std, use_random_rot=False)

# 生成软标签
if os.path.exists(SOFT_LABELS_PATH):
    print(f"Loading existing soft labels from {SOFT_LABELS_PATH}")
    soft_labels = torch.load(SOFT_LABELS_PATH)
else:
    soft_labels = generate_soft_labels(x_test_standard, config, TEACHER_WEIGHTS_PATH)


# --- 5. 蒸馏训练 (混合损失版本) ---
def distill(config, teacher_weights, raw_data, soft_labels_data, true_labels_data):
    print("\n--- Starting distillation with Mixed Loss ---")

    model_args = config['model_args']
    student = Model(**model_args).to(DEVICE)

    print("Initializing student with teacher's weights for finetuning.")
    state_dict = torch.load(teacher_weights, map_location=DEVICE)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    student.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)

    # DataLoader 现在包含索引和真实硬标签
    distill_dataset = TensorDataset(torch.arange(len(raw_data)), true_labels_data)
    distill_loader = DataLoader(distill_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 获取数据增强的配置
    train_feeder_args = config['train_feeder_args']  # 使用训练时的增强配置
    p_interval_aug = train_feeder_args.get('p_interval', [0.5, 1.0])
    use_random_rot_aug = train_feeder_args.get('random_rot', True)
    window_size_aug = train_feeder_args.get('window_size', 64)

    for epoch in range(EPOCHS):
        student.train()
        total_loss, total_loss_distill, total_loss_ce = 0, 0, 0
        pbar = tqdm(distill_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for (indices, true_labels) in pbar:
            soft_batch = soft_labels_data[indices].to(DEVICE)
            true_labels = true_labels.to(DEVICE)

            # --- 在循环内进行随机数据增强 ---
            processed_batch = []
            for idx in indices:
                sample = raw_data[idx.item()]
                T, _ = sample.shape
                sample = sample.reshape((T, 2, 25, 3)).transpose(3, 0, 2, 1)
                valid_num = np.sum(sample.sum(0).sum(-1).sum(-1) != 0)

                sample = tools.valid_crop_resize(sample, valid_num, p_interval_aug, window_size_aug)
                if use_random_rot_aug:
                    sample = tools.random_rot(sample)
                processed_batch.append(sample)

            x_augmented = torch.stack(processed_batch).float().to(DEVICE)
            student_logits = student(x_augmented)

            # --- 计算混合损失 ---
            # 1. 软损失 (蒸馏)
            loss_distill = F.kl_div(
                F.log_softmax(student_logits / TEMP, dim=1),
                F.softmax(soft_batch / TEMP, dim=1),
                reduction='batchmean'
            ) * (TEMP ** 2)

            # 2. 硬损失 (交叉熵)
            loss_ce = F.cross_entropy(student_logits, true_labels)

            # 3. 加权求和
            loss = ALPHA * loss_distill + (1 - ALPHA) * loss_ce

            optimizer.zero_grad();
            loss.backward();
            optimizer.step()

            total_loss += loss.item()
            total_loss_distill += loss_distill.item()
            total_loss_ce += loss_ce.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'distill': f'{loss_distill.item():.4f}',
                'ce': f'{loss_ce.item():.4f}'
            })

        print(f'Epoch {epoch + 1} Avg Loss: {total_loss / len(distill_loader):.4f} | '
              f'Distill Loss: {total_loss_distill / len(distill_loader):.4f} | '
              f'CE Loss: {total_loss_ce / len(distill_loader):.4f}')

    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f'Distillation done! Final model saved to -> {STUDENT_SAVE_PATH}')


# --- 主入口 ---
if __name__ == '__main__':
    distill(config, TEACHER_WEIGHTS_PATH, x_test_raw, soft_labels, y_test_labels)