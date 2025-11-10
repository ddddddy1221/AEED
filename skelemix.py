import random
import numpy as np
import torch
import torch.nn.functional as F

trunk_ori_index = [4, 3, 21, 2, 1]
left_hand_ori_index = [9, 10, 11, 12, 24, 25]
right_hand_ori_index = [5, 6, 7, 8, 22, 23]
left_leg_ori_index = [17, 18, 19, 20]
right_leg_ori_index = [13, 14, 15, 16]

trunk = [i - 1 for i in trunk_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_leg = [i - 1 for i in left_leg_ori_index]
right_leg = [i - 1 for i in right_leg_ori_index]
body_parts = [trunk, left_hand, right_hand, left_leg, right_leg]

trunk_ori_index_k400 = [1, 2, 3, 4, 5]
left_hand_ori_index_k400 = [6, 8, 10]
right_hand_ori_index_k400 = [7, 9, 11]
left_leg_ori_index_k400 = [12, 14, 16]
right_leg_ori_index_k400 = [13, 15, 17]

trunk_k400 = [i - 1 for i in trunk_ori_index_k400]
left_hand_k400 = [i - 1 for i in left_hand_ori_index_k400]
right_hand_k400 = [i - 1 for i in right_hand_ori_index_k400]
left_leg_k400 = [i - 1 for i in left_leg_ori_index_k400]
right_leg_k400 = [i - 1 for i in right_leg_ori_index_k400]
body_parts_k400 = [trunk_k400, left_hand_k400, right_hand_k400, left_leg_k400, right_leg_k400]


@torch.no_grad()
def ske_swap_randscale_k400(x, spa_l, spa_u, tem_l, tem_u, p=None):
    '''
    swap a batch skeleton
    T   64 --> 32 --> 16    # 8n
    S   25 --> 25 --> 25 (5 parts)
    '''
    N, C, T, V, M = x.size()
    tem_downsample_ratio = 4

    # generate swap swap idx
    idx = torch.arange(N)
    n = torch.randint(1, N - 1, (1,))
    randidx = (idx + n) % N

    # ------ Spatial ------ #
    if 1:
        Cs = random.randint(spa_l, spa_u)
        # sample the parts index
        parts_idx = random.sample(body_parts_k400, Cs)
        # generate spa_idx
        spa_idx = []
        for part_idx in parts_idx:
            spa_idx += part_idx
        spa_idx.sort()

    # ------ Temporal ------ #
    Ct = random.randint(tem_l, tem_u)
    tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
    rt = Ct * tem_downsample_ratio

    xst = x.clone()
    if p == None:
        p = random.random()
    if p > 0.25:
        N, C, T, V, M = xst.size()

        Ct_2 = random.randint(Ct, 25)
        tem_idx_2 = random.randint(0, T // tem_downsample_ratio - Ct_2)
        rt_2 = Ct_2 * tem_downsample_ratio

        xst_temp = xst[:, :, tem_idx_2 * tem_downsample_ratio: tem_idx_2 * tem_downsample_ratio + rt_2]

        xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
        xst_temp = xst_temp.view(N * M, V * C, -1)
        xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
        xst_temp = xst_temp.view(N, M, V, C, rt)
        xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
        xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
            xst_temp[randidx][:, :, :, spa_idx, :]
        # xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
        #        x[randidx][:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :]
        mask = torch.zeros(T // tem_downsample_ratio, V)
        mask[tem_idx:tem_idx + Ct, spa_idx] = 1
    else:
        lamb = random.random()
        xst = xst * (1 - lamb) + xst[randidx] * lamb
        mask = torch.zeros(T // tem_downsample_ratio, V) + lamb

    return randidx, xst, mask


@torch.no_grad()
def ske_swap_randscale(x, spa_l, spa_u, tem_l, tem_u):
    """
    Performs simple random ST-Mix.
    MODIFIED to use a robust loop and return a unified tuple.
    """
    N, C, T, V, M = x.size()
    device = x.device
    tem_downsample_ratio = 4

    # Generate a random pairing for the batch
    idx = torch.arange(N, device=device)
    # Ensure n is always a valid value
    n_val = random.randint(1, N - 1) if N > 1 else 0
    randidx = (idx + n_val) % N

    xst = x.clone()

    # Randomly decide between part-swapping and simple mixup
    if random.random() > 0.5:  # Part swapping
        # Select spatial and temporal regions ONCE for the whole batch
        Cs = random.randint(spa_l, spa_u)
        parts_to_mix = random.sample(body_parts, Cs)
        spa_idx = [joint for part in parts_to_mix for joint in part]
        spa_idx.sort()

        Ct = random.randint(tem_l, tem_u)
        tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
        rt = Ct * tem_downsample_ratio

        start_frame = tem_idx * tem_downsample_ratio
        end_frame = start_frame + rt

        # Use a robust loop for assignment to avoid broadcasting issues
        for i in range(N):
            xst[i, :, start_frame:end_frame, spa_idx, :] = \
                x[randidx[i], :, start_frame:end_frame, spa_idx, :]

        lamb_val = (rt * len(spa_idx)) / (T * V)
        physical_lambd = torch.full((N, 1), lamb_val, device=device)
    else:  # Simple Mixup
        lamb_val = np.random.beta(0.5, 0.5)
        xst = x * (1 - lamb_val) + x[randidx] * lamb_val
        physical_lambd = torch.full((N, 1), lamb_val, device=device)

    return randidx, xst, physical_lambd


@torch.no_grad()
def ske_swap_randscale_sample_noweighted(x, spa_l, spa_u, tem_l, tem_u, part_dist, randidx_pre=None):
    """
    Performs Shapley-value guided mixing.
    MODIFIED to:
    1. Accept a pre-computed randidx.
    2. Remove all internal label_dist and diffset logic.
    3. Use a robust loop for assignment.
    4. Always return (used_randidx, mixed_data, physical_lambda_tensor).
    """
    N, C, T, V, M = x.size()
    device = x.device
    tem_downsample_ratio = 4

    # Use pre-computed randidx if provided, otherwise generate a random one
    if randidx_pre is not None:
        randidx = randidx_pre.to(device)
    else:
        idx = torch.arange(N, device=device)
        n_val = random.randint(1, N - 1) if N > 1 else 0
        randidx = (idx + n_val) % N

    # Common temporal selection for the batch
    Ct = random.randint(tem_l, tem_u)
    tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
    rt = Ct * tem_downsample_ratio

    xst = x.clone()

    mapping = [
        [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4],
        [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4], [0, 3, 4],
        [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4],
    ]

    temp = 0.2
    part_prob_dist = F.softmax(part_dist / temp, dim=1)

    lamb_vals = []
    # Use a robust loop for assignment
    for i in range(N):
        dist = part_prob_dist[i]
        parts_combination_idx = random.choices(list(range(20)), weights=dist, k=1)[0]
        selected_parts = mapping[parts_combination_idx]

        spa_idx = [joint for part_idx in selected_parts for joint in body_parts[int(part_idx)]]
        spa_idx.sort()

        start_frame = tem_idx * tem_downsample_ratio
        end_frame = start_frame + rt

        # Perform the swap for the i-th sample
        xst[i, :, start_frame:end_frame, spa_idx, :] = \
            x[randidx[i], :, start_frame:end_frame, spa_idx, :]

        lamb_vals.append((rt * len(spa_idx)) / (T * V))

    physical_lambd = torch.tensor(lamb_vals, device=device).view(N, 1)

    return randidx, xst, physical_lambd

@torch.no_grad()
def ske_swap_randscale_sample_noweighted_k400(x, spa_l, spa_u, tem_l, tem_u, part_dist, label_dist=None, adatemp=None):
    '''
    swap a batch skeleton
    T   100 --> 50 --> 25    # 8n
    S   17 --> 17 --> 17 (5 parts)
    label dist: for long tailed data augmentation, labe_dist[i] is the frequency of the i-th data's (in a batch) label in the dataset
    '''
    N, C, T, V, M = x.size()

    tem_downsample_ratio = 4

    # generate swap swap idx
    idx = torch.arange(N)
    n = torch.randint(1, N - 1, (1,))
    randidx = (idx + n) % N

    # ------ Temporal ------ #
    Ct = random.randint(tem_l, tem_u)
    tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
    rt = Ct * tem_downsample_ratio

    xst = x.clone()

    p = random.random()

    if p > 0.25:
        # ------ Spatial ------ #

        Cs = random.randint(spa_l, spa_u)
        N, C, T, V, M = xst.size()

        Ct_2 = random.randint(Ct, 25)
        tem_idx_2 = random.randint(0, T // tem_downsample_ratio - Ct_2)
        rt_2 = Ct_2 * tem_downsample_ratio

        xst_temp = xst[:, :, tem_idx_2 * tem_downsample_ratio: tem_idx_2 * tem_downsample_ratio + rt_2]
        xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
        xst_temp = xst_temp.view(N * M, V * C, -1)
        xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
        xst_temp = xst_temp.view(N, M, V, C, rt)
        xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
        lamb = []
        mapping = [
            [0, 1], [0, 2], [0, 3], [0, 4], [1, 2],
            [1, 3], [1, 4], [2, 3], [2, 4], [3, 4],
            [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4],
            [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4],
        ]

        all_part = set(range(5))
        replace2remain = {}
        for replace_idx, item in enumerate(mapping):
            remain_idx = mapping.index(list(all_part.difference(set(item))))
            replace2remain[replace_idx] = remain_idx

        if adatemp == None:
            temp = 0.2
        else:
            temp = adatemp  # N,1
        part_prob_dist = part_dist / part_dist.sum(dim=1, keepdim=True)
        part_prob_dist = F.softmax(part_dist / temp, dim=1)

        for i in range(N):
            if label_dist != None:
                if label_dist[i] > label_dist[randidx[i]]:
                    dist = part_prob_dist[randidx[i]]
                    diffset = False
                else:
                    dist = part_prob_dist[i]
                    diffset = True

            spa_idx = []
            # print(len(part_dist))
            parts_idx = random.choices(list(range(20)), weights=dist, k=1)[0]

            p_choice = parts_idx
            if diffset:
                p_choice = replace2remain[p_choice]
            parts_idx = mapping[p_choice]
            for part_idx in parts_idx:
                spa_idx += body_parts_k400[int(part_idx)]
            spa_idx.sort()
            xst[i, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                xst_temp[randidx[i]][:, :, spa_idx, :]
            lamb.append(rt * len(spa_idx) / (T * V))
        lambd = torch.tensor(lamb).reshape((N, 1)).cuda()

    else:
        lamb = random.random()
        xst = xst * (1 - lamb) + xst[randidx] * lamb
        mask = torch.zeros(T // tem_downsample_ratio, V) + lamb
        lambd = torch.full((N, 1), lamb).cuda()

    return randidx, xst, lambd
