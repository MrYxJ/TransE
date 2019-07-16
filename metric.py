import numpy as np
import scipy
from scipy import io
import time

def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
    time1 =time.time()
    print('Embedding shape: ',vec.shape)
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')  # 曼哈顿距离计算相似度
    top_lr = [0] * len(top_k)
    MR_lr = 0
    ans_l = {}
    ans_r = {}
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
        MR_lr += rank_index + 1
    top_rl = [0] * len(top_k)
    MR_rl = 0
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
        MR_rl += rank_index + 1
    print('For each left:')
    print('Mean Rank:', MR_lr / len(test_pair))
    for i in range(len(top_lr)):
        print('Hits@%d: %.4f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
        ans_l[top_k[i]] = top_lr[i] / len(test_pair) * 100

    print('For each right:')
    print('Mean Rank:', MR_rl / len(test_pair))
    for i in range(len(top_rl)):
        print('Hits@%d: %.4f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
        ans_r[top_k[i]] = top_rl[i] / len(test_pair) * 100
    print('Hits cost time: %s s' % (time.time() - time1))
    return ans_l, ans_r