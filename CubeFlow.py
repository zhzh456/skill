# CubeFlow: Money Laundering Detection with Coupled Tensors
# CubeFlow is a scalable, flow-based approach to spot fraud from a mass of transactions by modeling them as two coupled tensors and applying a novel multi-attribute metric which can reveal the transfer chains accurately.

# Remember to add spartan to you PATH
import sys
# sys.path.append("/<dir to spartan2>/spartan2")
import spartan as st
import numpy as np

# Set parameters
# alpha: the coefficientof imbalance cost rate in the range of 0 to 1
# k: find top k dense blocks
# dim: dimensions of input data (support 3 or 4)
alpha = 0.8
k = 1
dim = 3

# Path of input data
xy_path = f'./inputData/CFD-{dim}/fs1.csv'
zy_path = f'./inputData/CFD-{dim}/fs2.csv'
gt_path = f'./inputData/CFD-{dim}/gt.npy'

# Path to save results
outpath = '' # '': not save results

# Load example data:
amt_tensor = st.loadTensor(path=xy_path, header=None)
cmt_tensor = st.loadTensor(path=zy_path, header=None)

amt_stensor = amt_tensor.toSTensor(hasvalue=True)
cmt_stensor = cmt_tensor.toSTensor(hasvalue=True)
print(amt_stensor.shape)
print(cmt_stensor.shape)

# Run as a model
cf = st.CubeFlow([amt_stensor, cmt_stensor], alpha=alpha, k=k, dim=dim, outpath=outpath)

# Run the algorithm and obtain two dense blocks as return value:
# $maxsize$ is the block size limit.
# $maxsize$ can be an integer (-1 or positive), and $maxsize==-1$ means no size limit.
# $maxsize$ can be a tuple which contains the node size limit for each dimension of the block, e.g.,(5,10,5). Similarly, each element of $maxsize$ should be an integer (-1 or positive).
# The results is a list of top-k suspicious blocks. Each block constains [[detected nodes in each partite], score]
res = cf.run(del_type=1, maxsize=-1)

# Evaluation
# Functions for calculating scores
def cal_f1(detectSet, trueSet, am_tensor, cm_tensor, has_t=False):
    # TP
    set_a = detectSet[0] & trueSet[0]
    set_m = detectSet[1] & trueSet[1]
    set_c = detectSet[2] & trueSet[2]
    fs1 = am_tensor.data
    fs2 = cm_tensor.data
    if has_t:
        set_t = detectSet[3] & trueSet[3]

        tp1 = fs1[(fs1[0].isin(set_a)) & (fs1[1].isin(set_m)) & (fs1[2].isin(set_t))]
        tp2 = fs2[(fs2[1].isin(set_m)) & (fs2[0].isin(set_c)) & (fs2[2].isin(set_t))]
        # FP+TP
        fptp1 = fs1[(fs1[0].isin(detectSet[0])) & (fs1[1].isin(detectSet[1])) & (fs1[2].isin(detectSet[3]))]
        fptp2 = fs2[(fs2[1].isin(detectSet[1])) & (fs2[0].isin(detectSet[2])) & (fs2[2].isin(detectSet[3]))]
        # FN+TP
        fntp1 = fs1[(fs1[0].isin(trueSet[0])) & (fs1[1].isin(trueSet[1])) & (fs1[2].isin(trueSet[3]))]
        fntp2 = fs2[(fs2[1].isin(trueSet[1])) & (fs2[0].isin(trueSet[2])) & (fs2[2].isin(trueSet[3]))]

    else:
        tp1 = fs1[(fs1[0].isin(set_a)) & (fs1[1].isin(set_m))]
        tp2 = fs2[(fs2[1].isin(set_m)) & (fs2[0].isin(set_c))]
        # FP+TP
        fptp1 = fs1[(fs1[0].isin(detectSet[0])) & (fs1[1].isin(detectSet[1]))]
        fptp2 = fs2[(fs2[1].isin(detectSet[1])) & (fs2[0].isin(detectSet[2]))]
        # FN+TP
        fntp1 = fs1[(fs1[0].isin(trueSet[0])) & (fs1[1].isin(trueSet[1]))]
        fntp2 = fs2[(fs2[1].isin(trueSet[1])) & (fs2[0].isin(trueSet[2]))]
        
    tpm = tp1[3].sum() + tp2[3].sum()
    tpfpm = fptp1[3].sum() + fptp2[3].sum()
    tpfnm = fntp1[3].sum() + fntp2[3].sum()
    # precision
    precision = tpm / tpfpm
    print(f'precision:{precision}')
    # recall
    recall = tpm / tpfnm
    print(f'recall:{recall}')
    # F1
    f1_score = 0.0
    if (precision + recall) > 0:
        f1_score = 2*precision*recall/(precision+recall)
    print(f'F1 score:{f1_score}')
    return f1_score, precision, recall

def get_groundtruth(path):
    gt = np.load(path, allow_pickle=True)
    gt = gt.tolist()
    return gt

def find_top_k_res(real_res, k):
    top_k_res = []
    top_k_a = set()
    top_k_m = set()
    top_k_c = set()
    for i in range(k):
        top_k_a = top_k_a.union(real_res[i][0][0])
        top_k_m = top_k_m.union(real_res[i][0][1])
        top_k_c = top_k_c.union(real_res[i][0][2])
    top_k_res.append(top_k_a)
    top_k_res.append(top_k_m)
    top_k_res.append(top_k_c)
    return top_k_res

# Report F-measure
gt = get_groundtruth(gt_path)
for i in range(1,k+1):
    print(f'Top {i} block:')
    top_k_res = find_top_k_res(res, i)
    f1, precision, recall = cal_f1(top_k_res, gt, amt_tensor,cmt_tensor)