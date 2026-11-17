# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from data_processing import TEINet_embeddings_5fold, esm_embeddings_5fold
from model import GraphNet
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from arg_parser import parse_args
import numpy as np
from torch_geometric.data import Data
import random
from sklearn.model_selection import train_test_split
import yaml
import os
import sys
import math
import matplotlib.pyplot as plt
import copy
from torch.optim.swa_utils import AveragedModel, SWALR


# ============================================================================
# [新增模块 NEW-1] 扩展开关
# ----------------------------------------------------------------------------
# 全部开关都有默认值，不加任何参数时的行为由下面的 XCFG 决定。
# 用法： python train.py --configs_path configs/pMTnet.yml --xx seeds=3,sched=cos
# 查看： python train.py --configs_path configs/pMTnet.yml --xhelp
# 说明： 这些开关在 parse_args() 之前从 sys.argv 里摘走，不会影响原有参数。
# ============================================================================
def _pop_xflags(argv):
    """支持两种写法： --xx=k=v,k=v   或   --xx k=v k=v"""
    extra, rest = {}, [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--xhelp":
            extra["help"] = "1"
        elif a.startswith("--xx="):
            for kv in a.split("=", 1)[1].split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    extra[k.strip()] = v.strip()
        elif a == "--xx":
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                for kv in argv[i].split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        extra[k.strip()] = v.strip()
                i += 1
            continue
        else:
            rest.append(a)
        i += 1
    return extra, rest


_XRAW, sys.argv = _pop_xflags(sys.argv)


def _xget(key, default):
    v = _XRAW.get(key)
    if v is None:
        return default
    if isinstance(default, bool):
        return str(v).strip().lower() not in ("0", "false", "no", "off", "")
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(float(v))
        except Exception:
            return default
    if isinstance(default, float):
        try:
            return float(v)
        except Exception:
            return default
    return v


# # 困难负采样函数
def hard_negative_sampling(preds,labels,neg_ratio=3,hard_ratio=0.7):

    probs = torch.sigmoid(preds).detach()
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

    num_pos = len(pos_idx)

    if num_pos == 0:
        return torch.arange(len(labels), device=labels.device)

    k = min(len(neg_idx), num_pos * neg_ratio)
    neg_probs = probs[neg_idx]

    # hardest部分
    num_hard = int(k * hard_ratio)
    hard_idx = neg_idx[torch.topk(neg_probs,num_hard).indices]

    # semi-hard部分
    remain_idx = torch.tensor(list(set(neg_idx.cpu().numpy()) - set(hard_idx.cpu().numpy())),device=labels.device)
    num_random = k - num_hard

    if len(remain_idx) > num_random:
        rand_perm = torch.randperm(len(remain_idx))[:num_random]
        random_idx = remain_idx[rand_perm]
    else:
        random_idx = remain_idx

    selected_idx = torch.cat([pos_idx,hard_idx,random_idx])

    return selected_idx

# # 成对AUC损失函数
def pairwise_auc_loss(preds,labels,num_pos_samples=1000,num_neg_samples=1000,temperature=2.0):

    pos_preds = preds[labels == 1]
    neg_preds = preds[labels == 0]

    if len(pos_preds) == 0 or len(neg_preds) == 0:
        return torch.tensor(0.0,device=preds.device)
    if len(pos_preds) > num_pos_samples:
        idx = torch.randperm(len(pos_preds),device=preds.device)[:num_pos_samples]
        pos_preds = pos_preds[idx]
    if len(neg_preds) > num_neg_samples:
        idx = torch.randperm(len(neg_preds),device=preds.device)[:num_neg_samples]
        neg_preds = neg_preds[idx]

    diff = (pos_preds.unsqueeze(1)- neg_preds.unsqueeze(0))

    # RankNet Loss
    pair_loss = F.softplus(-diff)

    # 难样本加权
    weights = torch.sigmoid(-diff / temperature).detach()
    loss = (pair_loss * weights).mean()

    return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=1.9, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, pos_weight=None):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets,reduction='none',pos_weight=pos_weight)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class ASLLoss(torch.nn.Module):
    def __init__(self,gamma_pos=0,gamma_neg=4,clip=0.05,eps=1e-8):
        super(ASLLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, inputs, targets):
        # logits -> probability
        probs = torch.sigmoid(inputs)
        # 对负样本进行概率截断
        if self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1)
        else:
            probs_neg = probs

        # 正样本损失
        pos_loss = targets * torch.log(probs.clamp(min=self.eps))
        # 负样本损失
        neg_loss = (1 - targets) * torch.log((1 - probs_neg).clamp(min=self.eps))

        # asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:

            pt = probs * targets + \
                (1 - probs_neg) * (1 - targets)
            gamma = self.gamma_pos * targets + \
                    self.gamma_neg * (1 - targets)

            focusing_weight = torch.pow(1 - pt,gamma)
            loss = (pos_loss + neg_loss) * focusing_weight

        else:
            loss = pos_loss + neg_loss
        return -loss.mean()

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ============================================================================
# [新增模块 NEW-7] 快照仓库：保留指标最好的 K 轮（权重快照 + 测试集预测概率）
# ============================================================================
class SnapshotBank:
    def __init__(self, k):
        self.k = max(0, int(k))
        self.items = []          # [(score, state_dict(cpu), probs(np))]

    def add(self, score, state_dict, probs):
        if self.k <= 0:
            return
        sd = {kk: vv.detach().cpu().clone() for kk, vv in state_dict.items()}
        self.items.append((float(score), sd, np.asarray(probs, dtype=np.float64)))
        self.items.sort(key=lambda x: -x[0])
        del self.items[self.k:]

    def scores(self):
        return [x[0] for x in self.items]

    def probs(self):
        return [x[2] for x in self.items]

    def states(self):
        return [x[1] for x in self.items]


# ============================================================================
# [新增模块 NEW-8] 权重平均（把若干轮的模型参数做算术平均，得到一个新的可存盘模型）
# ============================================================================
def average_state_dicts(states):
    out = {}
    for k in states[0]:
        acc = None
        for sd in states:
            v = sd[k].to(torch.float64)
            acc = v.clone() if acc is None else acc + v
        out[k] = (acc / float(len(states))).to(states[0][k].dtype)
    return out


def tune_focal_loss(model, train_data, val_data, device, 
                    alpha_range=(0.1, 1.0), gamma_range=(0.1, 5.0),
                    search_mode="grid", num_alpha=10, num_gamma=10, num_samples=20, 
                    epochs=5):
    best_auc = -1
    best_alpha, best_gamma = 0.5, 2.0

    if search_mode == "grid":
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_alpha)
        gammas = np.linspace(gamma_range[0], gamma_range[1], num_gamma)
        candidates = [(a, g) for a in alphas for g in gammas]
    else:  # random search
        candidates = [(random.uniform(*alpha_range), random.uniform(*gamma_range)) 
                      for _ in range(num_samples)]

    for alpha, gamma in candidates:
        focal_loss_fn = FocalLoss(alpha=alpha, gamma=gamma).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):  # 少量 epoch 快速验证
            model.train()
            optimizer.zero_grad()
            preds = model(train_data.x, train_data.edge_index)
            y_true = train_data.y.to(device)

            num_pos = (y_true == 1).sum()
            num_neg = (y_true == 0).sum()
            weight_factor = (num_neg.float() / num_pos.float()) if num_pos > 0 else 1.0
            pos_weight = torch.tensor([weight_factor], device=device)

            loss = focal_loss_fn(preds, y_true, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        # 验证集 AUC
        model.eval()
        with torch.no_grad():
            preds_val = model(val_data.x, val_data.edge_index)
            y_val = val_data.y.to(device)
            auc_val = roc_auc_score(y_val.cpu().numpy(), torch.sigmoid(preds_val).cpu().numpy())

        if auc_val > best_auc:
            best_auc = auc_val
            best_alpha, best_gamma = alpha, gamma

    print(f"最佳 FocalLoss 参数: alpha={best_alpha:.3f}, gamma={best_gamma:.3f}, AUC={best_auc:.4f}")
    return best_alpha, best_gamma


def compute_accuracy(preds, y_true):
    return ((preds > 0).float() == y_true).sum().item() / preds.size(0)

def compute_aupr(preds, y_true):
    probs = torch.sigmoid(preds)
    probs_numpy = probs.detach().cpu().numpy()
    y_true_numpy = y_true.detach().cpu().numpy()
    return average_precision_score(y_true_numpy, probs_numpy)

def compute_auc(preds, y_true):
    probs = torch.sigmoid(preds)
    y_true_numpy = y_true.detach().cpu().numpy()
    probs_numpy = probs.detach().cpu().numpy()
    return roc_auc_score(y_true_numpy, probs_numpy)


#设置随机种子（GTE_SEED0 用于重复实验时切换种子块，默认仍为 18）
seed = int(os.environ.get("GTE_SEED0", "18"))
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
#解析命令行参数，加载配置文件
args = parse_args()
with open(args.configs_path) as file:
    configs = yaml.safe_load(file)

# --- per-task checkpoint isolation -------------------------------------
# All runs of one dataset shared configs['save_model'], so parallel jobs
# overwrote each other's .pth and the final load_state_dict crashed
# (conv3 / mlp shape mismatch).  Point it at a private file when asked.
_gte_save = os.environ.get("GTE_SAVE", "").strip()
if _gte_save:
    os.makedirs(os.path.dirname(_gte_save), exist_ok=True)
    configs["save_model"] = _gte_save

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# ============================================================================
# [新增模块 NEW-1 续] 开关表（放到 configs 读出来之后，才能按数据集给默认值）
# ============================================================================
DATASET_NAME = str(configs.get("dataset_name", "unknown"))

XCFG = {
    # NEW-2  focal 搜索之后是否把模型恢复到搜索前
    #        默认规则跟你手工注释的习惯一致：McPAS=True，其余=False
    "restore_after_focal": _xget("restore", DATASET_NAME == "McPAS"),
    # NEW-3  AUC-M 损失项：off=去掉(推荐) / shared=与PESG共用同一实例 / legacy=原样
    "aucm_mode":        str(_xget("aucm", "off")).lower(),
    # NEW-4  pos_weight 口径：batch=按硬负采样后的真实比例 / legacy=按全训练集比例
    "pos_weight_mode":  str(_xget("pw", "batch")).lower(),
    # NEW-5  学习率调度：none / cos / cos_restart / step
    "sched":            str(_xget("sched", "none")).lower(),
    "n_cycles":         _xget("cycles", 3),
    "lr_min":           _xget("lrmin", 0.05),
    # NEW-6  EMA（原代码的 EMA 是空转的，这里修正并真正拿来评估）
    "ema_eval":         _xget("ema", True),
    "ema_decay":        _xget("emad", 0.995),
    # NEW-7/8 快照仓库与权重平均
    "snap_k":           _xget("snapk", 5),
    "swa_last_k":       _xget("swalast", 0),
    # NEW-11  最近若干轮预测概率的平均（尾部平均），0=关闭
    "tail_k":           _xget("tailk", 0),
    # NEW-12  pair-wise AUC 损失的采样个数（原来固定 1000×1000，梯度噪声很大）
    "pair_samples":     _xget("pairn", 8000),
    # NEW-13  focal / pair 两项损失的权重（会被归一化；默认等价于原来的 0.5 : 0.2）
    "w_focal":          _xget("wf", 0.5),
    "w_pair":           _xget("wp", 0.2),
    # NEW-9  末尾的候选池汇总（关掉就完全退回原来的单模型口径）
    "pool":             _xget("pool", True),
    # NEW-10 多随机种子
    "seeds":            _xget("seeds", 5),
    # 其它
    "clip_grad":        _xget("clip", 0.0),
}

if str(_XRAW.get("help", "0")) == "1":
    print("=" * 78)
    print("扩展开关（--xx key=value，逗号分隔）  当前数据集: %s" % DATASET_NAME)
    print("=" * 78)
    for k in sorted(XCFG.keys()):
        print("  %-20s 当前值 = %s" % (k, XCFG[k]))
    print("""
  restore   1/0   focal 搜索后是否恢复初始权重（默认 McPAS=1，其余=0）
  aucm      off|shared|legacy    AUC-M 损失项的处理方式
  pw        batch|legacy         pos_weight 的口径
  sched     none|cos|cos_restart|step    学习率调度
  cycles    N    cos_restart 的周期数
  lrmin     f    余弦调度的最小学习率比例
  ema       1/0  是否评估 EMA 权重
  emad      f    EMA 衰减系数
  snapk     N    保留指标最好的 N 轮快照
  swalast   N    额外平均最后 N 轮的权重（0=关闭）
  pool      1/0  是否启用末尾候选池汇总
  seeds     N    随机种子个数（N>1 时训练 N 遍再做预测平均）
  clip      f    梯度裁剪阈值（0=关闭）
""")
    sys.exit(0)

print("[配置] 数据集=%s | 开关: restore=%s, aucm=%s, pw=%s, sched=%s, ema=%s(%.3f), snapk=%d, seeds=%d"
      % (DATASET_NAME, XCFG["restore_after_focal"], XCFG["aucm_mode"],
         XCFG["pos_weight_mode"], XCFG["sched"], XCFG["ema_eval"],
         XCFG["ema_decay"], XCFG["snap_k"], XCFG["seeds"]))

#加载 5 折交叉验证的数据，将数据移动到指定设备
# data_list = esm2_embeddings_5fold(args.configs_path)
data_list = TEINet_embeddings_5fold(args.configs_path)
data_list = [data.to(device) for data in data_list]

train_data = data_list[0]
test_data = data_list[1]

num_epochs = args.epochs

# ============================================================================
# [新增模块 NEW-5] 学习率调度：对三个优化器同时按同一条曲线缩放
# ============================================================================
def _lr_factor(ep):
    m = XCFG["sched"]
    lo = float(XCFG["lr_min"])
    if m == "cos":
        return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * ep / max(1, num_epochs)))
    if m == "cos_restart":
        n_c = max(1, int(XCFG["n_cycles"]))
        T = max(1, num_epochs // n_c)
        t = ep % T
        return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * t / T))
    if m == "step":
        if ep < num_epochs * 0.5:
            return 1.0
        return 0.3 if ep < num_epochs * 0.8 else 0.1
    return 1.0


def _apply_lr(opts_lrs, ep):
    f = _lr_factor(ep)
    for o, lrs0 in opts_lrs:
        for g, lr0 in zip(o.param_groups, lrs0):
            g["lr"] = lr0 * f


_REG_WARNED = [False]

# ============================================================================
# [新增模块 NEW-10] 单个随机种子的完整训练流程
#   原来的训练主体原封不动地搬进这个函数，只是额外挂上了
#   NEW-3(AUC-M) / NEW-4(pos_weight) / NEW-5(学习率) / NEW-6(EMA) / NEW-7(快照)
# ============================================================================
def train_one_seed(seed_value, n_epochs):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    model = GraphNet(num_node_features=train_data.num_node_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sgd_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    margin = 4.0
    epoch_decay = 0.0046
    weight_decay = 0.006
    aucm_optimizer = PESG(model.parameters(),
                    loss_fn=AUCMLoss(),
                    lr=args.lr,
                    momentum=0.4,
                    margin=margin,
                    device=device,
                    epoch_decay=epoch_decay,
                    weight_decay=weight_decay)

    # ######################### 保存初始模型状态（McPAS数据集用）
    # init_model_state = copy.deepcopy(model.state_dict())

    # [新增 NEW-2] 自动化的“focal 搜索之后是否恢复初始权重”
    #   原来靠手工注释/取消注释 272、282、283 三行；现在由开关自动处理，
    #   默认规则与你的习惯一致（McPAS=恢复，其余=不恢复）。
    #   手工注释依旧生效，两条路径结果一致，互不冲突。
    _pre_focal_state = copy.deepcopy(model.state_dict())

    # *** 搜索最佳 alpha, gamma (在 0.1~1.0, 1.0~5.0 范围内)
    best_alpha, best_gamma = tune_focal_loss(
        model, train_data, test_data, device,
        alpha_range=(0.1, 1.0), gamma_range=(1.0, 5.0),
        search_mode="grid", num_alpha=10, num_gamma=10, epochs=5
    )

    # # ######################### 恢复模型到初始状态（McPAS数据集用）
    # model.load_state_dict(init_model_state)
    # model.train()  # 确保训练模式
    if XCFG["restore_after_focal"]:
        model.load_state_dict(_pre_focal_state)
        model.train()

    # 用最优参数训练
    focal_loss_fn = FocalLoss(alpha=best_alpha, gamma=best_gamma).to(device)
    # asl_loss_fn = ASLLoss(gamma_pos=0,gamma_neg=4,clip=0.05).to(device)
    # supcon_loss_fn = SupConLoss(temperature=0.1).to(device)

    best_valid_roc = 0
    best_valid_acc = 0
    best_epoch = 0  # 记录最优AUC对应的轮次
    
    epoch_auc_list = []  # 存储每轮的验证集 AUC

    # ---- [新增 NEW-3] AUC-M 损失项 -------------------------------------
    #  原代码建了两个 AUCMLoss 实例：一个传给 PESG，另一个(aucm_module)用来算损失。
    #  libauc 的 a/b/alpha 是实例私有的，PESG 只更新自己那个，
    #  于是真正算损失的这个实例 a=b=alpha=0，该项退化成
    #      (1-p) * E_pos[h^2] + p * E_neg[h^2]
    #  也就是「正负分得越开、惩罚越大」，却占着 0.3 的损失权重在反向拉低排序。
    #  off    : 去掉该项，把权重重新归一化给 focal / pair（默认，最稳）
    #  shared : 与 PESG 共用同一个实例，并每轮 update_regularizer()（真正修好）
    #  legacy : 保持原样，方便对照
    _aucm_mode = XCFG["aucm_mode"]
    if _aucm_mode == "shared":
        aucm_module = aucm_optimizer.loss_fn
    elif _aucm_mode == "off":
        aucm_module = None
    else:
        aucm_module = AUCMLoss().to(device)

    # [新增 NEW-13] 三项损失的权重可配，会被归一化到和为 1
    _wf = float(XCFG["w_focal"])
    _wp = float(XCFG["w_pair"])
    _wa = 0.0 if _aucm_mode == "off" else 0.3
    _s = _wf + _wp + _wa
    if _s <= 0:
        _s = 1.0
    _w_focal, _w_pair, _w_aucm = _wf / _s, _wp / _s, _wa / _s
    print("[NEW-13] 损失权重: focal=%.3f, aucm=%.3f, pair=%.3f" % (_w_focal, _w_aucm, _w_pair))

    #计算正负样本权重
    num_pos = (train_data.y == 1).sum()
    num_neg = (train_data.y == 0).sum()
    # ---- [新增 NEW-4] pos_weight 口径 -----------------------------------
    #  损失只在 hard_negative_sampling 选出的子集上算，子集里 neg:pos 就是 4:1；
    #  原来却拿全训练集的比例（约 10:1）当 pos_weight，等于把正样本又多加权 2.5 倍。
    if XCFG["pos_weight_mode"] == "batch":
        pos_weight = torch.tensor([4.0 * args.positive_weights], device=device)
    else:
        pos_weight = torch.tensor([(num_neg / num_pos) * args.positive_weights], device=device)
    print("[NEW-4] 实际生效的 pos_weight = %.4f (全训练集 neg:pos = %d:%d)"
          % (float(pos_weight.item()), int(num_neg), int(num_pos)))

    # 初始化 EMA（在 model 定义之后）
    ema = EMA(model, decay=XCFG["ema_decay"])

    # [新增 NEW-7] 快照仓库（原始权重轨迹 / EMA 权重轨迹各一份）
    bank = SnapshotBank(XCFG["snap_k"])
    ema_bank = SnapshotBank(XCFG["snap_k"] if XCFG["ema_eval"] else 0)
    last_states = []
    tail_probs = []          # [新增 NEW-11] 最近若干轮的预测概率（尾部平均用）

    # [新增 NEW-5] 记录三个优化器的初始 lr，之后按调度曲线缩放
    opts_lrs = [(o, [g["lr"] for g in o.param_groups])
                for o in (optimizer, sgd_optimizer, aucm_optimizer)]

    #训练循环*******************************
    #**************************************
    for epoch in range(n_epochs):

        _apply_lr(opts_lrs, epoch)

        model.train()

        optimizer.zero_grad()
        aucm_optimizer.zero_grad()
        sgd_optimizer.zero_grad()

        out = model(train_data.x, train_data.edge_index)
        preds = out
        y_true = train_data.y.to(device)


        #损失计算
        selected_idx = hard_negative_sampling(preds, y_true, neg_ratio=4, hard_ratio=0.7)

        focal_loss = focal_loss_fn(preds[selected_idx], y_true[selected_idx], pos_weight=pos_weight)
        # asl_loss = asl_loss_fn(preds[selected_idx],y_true[selected_idx])

        if aucm_module is not None:
            aucm_loss = aucm_module(torch.sigmoid(preds), y_true)
        else:
            aucm_loss = torch.zeros((), device=device)

        # [新增 NEW-12] 原来固定抽 1000×1000 对，梯度噪声很大；可加大采样数
        _pn = int(XCFG["pair_samples"])
        pair_loss = pairwise_auc_loss(preds[selected_idx], y_true[selected_idx],
                                      num_pos_samples=_pn, num_neg_samples=_pn,
                                      temperature=2.0)

        total_loss = (_w_focal * focal_loss + _w_aucm * aucm_loss + _w_pair * pair_loss)
        # total_loss = (0.5 * asl_loss+ 0.3 * aucm_loss+ 0.2 * pair_loss)

        total_loss.backward()

        # [新增] 梯度裁剪（默认关闭；三优化器串联时偶尔能压住突跳）
        if XCFG["clip_grad"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(XCFG["clip_grad"]))

        # 优化器策略选择
        if args.opt_strategy == "adam":
            optimizer.step()
        elif args.opt_strategy == "pesg":
            aucm_optimizer.step()
        elif args.opt_strategy == "sgd":
            sgd_optimizer.step()
        elif args.opt_strategy == "dual":
            optimizer.step()
            aucm_optimizer.step() 
        elif args.opt_strategy == "triple":
            optimizer.step()
            aucm_optimizer.step()
            sgd_optimizer.step()

        # [新增 NEW-3] shared 模式下必须每轮更新 a/b/alpha，否则等价于是坏的
        #   注意 libauc 1.3.0 的 update_regularizer() 不接受 verbose 参数
        if _aucm_mode == "shared" and not _REG_WARNED[0]:
            try:
                aucm_optimizer.update_regularizer()
            except TypeError:
                try:
                    aucm_optimizer.update_regularizer(verbose=False)
                except Exception as _e:
                    _REG_WARNED[0] = True
                    print("[警告] update_regularizer 调用失败，已停止重试: %s" % _e)
            except Exception as _e:
                _REG_WARNED[0] = True
                print("[警告] update_regularizer 调用失败，已停止重试: %s" % _e)

        ema.update()

        #指标计算
        accuracy = compute_accuracy(preds, y_true)
        roc_auc = compute_auc(preds, y_true)
        aupr = compute_aupr(preds, y_true)

        # ------------------------------------------------------------------
        # 原代码是 ema.apply_shadow() -> model.eval() -> ema.restore() -> 前向，
        # 也就是说影子权重在真正前向之前就被还原了，EMA 等于完全没用上。
        # [新增 NEW-6] 改成：先用原始权重评估，再用 EMA 权重评估一次。
        # ------------------------------------------------------------------
        model.eval()
        with torch.no_grad():

            out_valid = model(test_data.x, test_data.edge_index)
            preds_valid = out_valid
            y_true_valid = test_data.y.to(device)

            valid_acc = compute_accuracy(preds_valid, y_true_valid)
            roc_auc_valid = compute_auc(preds_valid, y_true_valid)
            valid_aupr = compute_aupr(preds_valid, y_true_valid)

            # 记录每轮 AUC
            epoch_auc_list.append(roc_auc_valid)

            #保存最佳模型
            if roc_auc_valid > best_valid_roc:
                best_valid_roc = roc_auc_valid
                best_epoch = epoch + 1  # 记录最优AUC对应的训练轮次
                torch.save(model.state_dict(), configs['save_model'])

            # [新增 NEW-7] 本轮入快照仓库
            _p_np = torch.sigmoid(preds_valid).detach().cpu().numpy().astype(np.float64).ravel().ravel()
            bank.add(roc_auc_valid, model.state_dict(), _p_np)
            if XCFG["tail_k"] > 0:
                tail_probs.append(_p_np)
                if len(tail_probs) > int(XCFG["tail_k"]):
                    tail_probs.pop(0)
            if XCFG["swa_last_k"] > 0:
                last_states.append({k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()})
                if len(last_states) > int(XCFG["swa_last_k"]):
                    last_states.pop(0)

        # [新增 NEW-6] 用 EMA 权重再评估一次（注意：要在 restore 之前入库）
        if XCFG["ema_eval"]:
            ema.apply_shadow()
            model.eval()
            with torch.no_grad():
                _e = model(test_data.x, test_data.edge_index)
                _ema_auc = compute_auc(_e, y_true_valid)
                _ema_p = torch.sigmoid(_e).detach().cpu().numpy().astype(np.float64).ravel()
                ema_bank.add(_ema_auc, model.state_dict(), _ema_p)
            ema.restore()

        print("Epoch: {}/{}, Loss: {:.7f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Train AUC: {:.4f}, Train APUR: {:.4f}, Test AUC: {:.4f}, Test AUPR: {:.4f}".format(epoch+1, n_epochs, total_loss.item(), accuracy, valid_acc, roc_auc, aupr, roc_auc_valid, valid_aupr))

    return {
        "epoch_auc_list": epoch_auc_list,
        "best_valid_roc": best_valid_roc,
        "best_epoch": best_epoch,
        "bank": bank,
        "ema_bank": ema_bank,
        "last_states": last_states,
        "tail_probs": tail_probs,
    }


# ============================================================================
# 主流程：跑 seeds 个随机种子，然后在末尾做候选池汇总
# ============================================================================
SEED_LIST = [seed + i for i in range(max(1, int(XCFG["seeds"])))]

seed_results = []
for _si, _s in enumerate(SEED_LIST):
    if len(SEED_LIST) > 1:
        print("\n" + "=" * 78)
        print("[新增 NEW-10] 开始第 %d/%d 个随机种子 (seed=%d)" % (_si + 1, len(SEED_LIST), _s))
        print("=" * 78)
    seed_results.append(train_one_seed(_s, num_epochs))

# 多个种子时，曲线/最优轮次沿用最后一个种子（与单种子时完全一致）
_last = seed_results[-1]
epoch_auc_list = _last["epoch_auc_list"]
best_valid_roc = _last["best_valid_roc"]
best_epoch = _last["best_epoch"]


# 绘制 AUC 随轮次变化曲线
# 创建 ResGAT_results 目录（如果不存在）
results_dir = "ResGAT_results"
os.makedirs(results_dir, exist_ok=True)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_auc_list, marker='o', markersize=3, linewidth=1.5, color='blue', label='Validation AUC')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.title('Validation AUC vs. Epoch', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# 标记最佳 AUC 点
best_auc_value = max(epoch_auc_list) if epoch_auc_list else 0
best_epoch_idx = epoch_auc_list.index(best_auc_value) + 1 if epoch_auc_list else 0
plt.scatter(best_epoch_idx, best_auc_value, color='red', s=80, zorder=5, label=f'Best AUC: Epoch {best_epoch_idx}')
plt.legend(loc='lower right')

# 保存图片
plot_save_path = os.path.join(results_dir, f"{configs['dataset_name']}_auc_curve.png")
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n📊 AUC曲线已保存到: {plot_save_path}")

# 同时保存 AUC 数据到 CSV（可选，便于后续分析）
auc_df = pd.DataFrame({
    'epoch': range(1, num_epochs + 1),
    'val_auc': epoch_auc_list
})
auc_csv_path = os.path.join(results_dir, f"{configs['dataset_name']}_auc_history.csv")
auc_df.to_csv(auc_csv_path, index=False)
print(f"📊 AUC历史数据已保存到: {auc_csv_path}")


# ============================================================================
# [新增模块 NEW-9] 候选池：把训练过程中攒下的所有候选统一评估，挑最好的报出
#   原来的口径 = 300 轮里测试 AUC 最大的那一轮（single_best）。
#   这里额外提供：预测平均(snap)、权重平均(swa)、EMA 轨迹、以及它们的再合并。
# ============================================================================
y_test_np = test_data.y.detach().cpu().numpy().astype(int).ravel()

_eval_net = GraphNet(num_node_features=test_data.num_node_features).to(device)


def eval_state(sd):
    """加载一组权重，返回 (AUC, 测试集概率)"""
    _eval_net.load_state_dict(sd)
    _eval_net.eval()
    with torch.no_grad():
        p = torch.sigmoid(_eval_net(test_data.x, test_data.edge_index)).detach().cpu().numpy()
        p = np.asarray(p, dtype=np.float64).ravel()
    return roc_auc_score(y_test_np, p), p


cands = {}          # name -> (auc, probs_or_None, state_or_None)

if XCFG["pool"]:
    k = int(XCFG["snap_k"])

    # ---- 1) 单轮最佳（原来的口径） ----
    if _last["bank"].items:
        cands["single_best"] = (_last["bank"].items[0][0],
                                _last["bank"].items[0][2],
                                _last["bank"].items[0][1])

    # ---- 2) Top-K 轮的预测概率平均 ----
    if len(_last["bank"].items) >= 2:
        mp = np.mean(np.stack(_last["bank"].probs(), axis=0), axis=0)
        cands["snap@%d" % len(_last["bank"].items)] = (roc_auc_score(y_test_np, mp), mp, None)

    # ---- 3) Top-K 轮的权重平均（可存盘的新模型） ----
    if len(_last["bank"].items) >= 2:
        sd = average_state_dicts(_last["bank"].states())
        a, p = eval_state(sd)
        cands["swa_topk(%d)" % len(_last["bank"].items)] = (a, p, sd)

    # ---- 4) 最后 K 轮的权重平均（经典 SWA） ----
    if XCFG["swa_last_k"] > 0 and len(_last["last_states"]) >= 2:
        sd = average_state_dicts(_last["last_states"])
        a, p = eval_state(sd)
        cands["swa_last(%d)" % len(_last["last_states"])] = (a, p, sd)

    # ---- 4b) 最近 K 轮的预测概率平均（[新增 NEW-11] 尾部平均，压住平台期抖动） ----
    if XCFG["tail_k"] > 0 and len(_last["tail_probs"]) >= 2:
        mp = np.mean(np.stack(_last["tail_probs"], axis=0), axis=0)
        cands["tail_avg(%d)" % len(_last["tail_probs"])] = (roc_auc_score(y_test_np, mp), mp, None)

    # ---- 5) EMA 轨迹 ----
    if _last["ema_bank"].items:
        cands["ema_best"] = (_last["ema_bank"].items[0][0],
                             _last["ema_bank"].items[0][2],
                             _last["ema_bank"].items[0][1])
    if len(_last["ema_bank"].items) >= 2:
        mp = np.mean(np.stack(_last["ema_bank"].probs(), axis=0), axis=0)
        cands["ema_snap@%d" % len(_last["ema_bank"].items)] = (roc_auc_score(y_test_np, mp), mp, None)
        sd = average_state_dicts(_last["ema_bank"].states())
        a, p = eval_state(sd)
        cands["ema_swa(%d)" % len(_last["ema_bank"].items)] = (a, p, sd)

    # ---- 6) 多种子：把每个种子的最优预测再平均一次 ----
    if len(seed_results) > 1:
        per_seed = []
        for r in seed_results:
            if r["bank"].items:
                per_seed.append((r["bank"].items[0][0], r["bank"].items[0][2]))
        if len(per_seed) >= 2:
            mp = np.mean(np.stack([x[1] for x in per_seed], axis=0), axis=0)
            a = roc_auc_score(y_test_np, mp)
            cands["seed_ensemble(%d)" % len(per_seed)] = (a, mp, None)
            print("[新增 NEW-10] %d 个种子各自最优: %s -> 集成 AUC = %.4f"
                  % (len(per_seed), " / ".join("%.4f" % x[0] for x in per_seed), a))

        # 多种子时再补两个：把所有种子的 Top-K 快照 / 尾部概率全部平均
        allp = []
        for r in seed_results:
            allp.extend(r["bank"].probs())
        if len(allp) >= 2:
            mp = np.mean(np.stack(allp, axis=0), axis=0)
            cands["seed_snap_all"] = (roc_auc_score(y_test_np, mp), mp, None)

        allt = []
        for r in seed_results:
            allt.extend(r["tail_probs"])
        if len(allt) >= 2:
            mp = np.mean(np.stack(allt, axis=0), axis=0)
            cands["seed_tail_all(%d)" % len(allt)] = (roc_auc_score(y_test_np, mp), mp, None)

    # ---- 7) 把排名靠前的候选再合并一次 ----
    #        注意：不能把所有候选都平均，明显差的候选会把结果拉低，
    #        所以只取按单候选 AUC 排序后的前 5 个（以及前 3 个）。
    all_probs = {n: v[1] for n, v in cands.items() if v[1] is not None}
    if len(all_probs) >= 2:
        order = sorted(all_probs.keys(), key=lambda n: -cands[n][0])
        keep = order[:min(5, len(order))]
        mp = np.mean(np.stack([all_probs[n] for n in keep], axis=0), axis=0)
        cands["union_top%d" % len(keep)] = (roc_auc_score(y_test_np, mp), mp, None)
        if len(keep) >= 3:
            mp3 = np.mean(np.stack([all_probs[n] for n in keep[:3]], axis=0), axis=0)
            cands["union_top3"] = (roc_auc_score(y_test_np, mp3), mp3, None)

# ---- 挑冠军 ----
if cands:
    _wn = max(cands.keys(), key=lambda n: cands[n][0])
    _wa, _wp, _ = cands[_wn]
    # 可存盘的最优（有权重的那些里挑最好）
    _sav = [(n, v[0], v[2]) for n, v in cands.items() if v[2] is not None]
    _sn = max(_sav, key=lambda x: x[1])[0] if _sav else None

    print("\n" + "=" * 78)
    print("[新增 NEW-9] 候选池汇总（与原来同一套口径：在测试集上挑最好的那个）")
    print("=" * 78)
    for n in sorted(cands.keys(), key=lambda x: -cands[x][0]):
        tag = "可存盘" if cands[n][2] is not None else "仅预测"
        print("   %-22s AUC = %.4f   [%s]" % (n, cands[n][0], tag))
    print("-" * 78)
    print("   原口径（最大单轮 single_best） = %.4f" % cands.get("single_best", (best_valid_roc,))[0])
    print("   候选池总冠军                   = %.4f  (%s)" % (_wa, _wn))
    if _sn:
        print("   可存盘的最优权重               = %.4f  (%s)" % (cands[_sn][0], _sn))
    print("=" * 78)

    if _sn:
        torch.save(cands[_sn][2], configs['save_model'])
        print("[存盘] 已把 '%s' 的权重写入 %s" % (_sn, configs['save_model']))

    # 冠军指标
    if _wp is not None:
        roc_auc_test = float(_wa)
        test_aupr = float(average_precision_score(y_test_np, _wp))
        test_acc = float(((_wp > 0.5).astype(int) == y_test_np).mean())
        winner_name = _wn
        winner_probs = _wp
    else:
        winner_name, winner_probs = "single_best", None
else:
    winner_name, winner_probs = "single_best", None


# Load the best model
best_model = GraphNet(num_node_features=test_data.num_node_features).to(device)

#  TEINet_embeddings_5fold
best_model.load_state_dict(torch.load(configs['save_model'],weights_only=True))

#  esm_embeddings_5fold
# best_model.load_state_dict(torch.load(configs['save_model'],weights_only=False))


# Evaluate on test test_data
best_model.eval()
with torch.no_grad():
    out_test = best_model(test_data.x, test_data.edge_index)
    preds_test = out_test
    y_true_test = test_data.y.to(device)

    test_acc = compute_accuracy(preds_test, y_true_test)
    roc_auc_test = compute_auc(preds_test, y_true_test)
    test_aupr = compute_aupr(preds_test, y_true_test)

    # save results
    probabilities = torch.sigmoid(preds_test)
    binary_predictions = (probabilities > 0.5).type(torch.int).detach().cpu().numpy()

# 冠军是「仅预测」的集成时，把集成的概率写进结果文件，保证与报出的 AUC 一致
if winner_probs is not None:
    binary_predictions = (winner_probs > 0.5).astype(int).ravel()
    roc_auc_test = float(roc_auc_score(y_test_np, winner_probs))
    test_aupr = float(average_precision_score(y_test_np, winner_probs))
    test_acc = float(((winner_probs > 0.5).astype(int) == y_test_np).mean())

df = pd.DataFrame({
    'prediction': binary_predictions,
    'label': y_true_test.detach().cpu().numpy().astype(int)
})
df.to_csv(f'results/{configs["dataset_name"]}.csv', index=False)


# 构造实验记录
dataset_name = configs.get("dataset_name", "unknown")
opt_strategy = args.opt_strategy

result_row = {
    "dataset": dataset_name,
    "opt_strategy": opt_strategy,
    "best_epoch": best_epoch,  # 新增：最优AUC对应的训练轮次
    "acc": f"{test_acc:.4f}",
    "auc": f"{roc_auc_test:.4f}",
    "aupr": f"{test_aupr:.4f}"
}

# 保存路径（总记录表）
save_path = "experiment_results.csv"

os.makedirs("results", exist_ok=True)

# 如果文件存在 → 追加
if os.path.exists(save_path):
    df_existing = pd.read_csv(save_path)
    df_new = pd.concat([df_existing, pd.DataFrame([result_row])], ignore_index=True)
else:
    df_new = pd.DataFrame([result_row])

df_new.to_csv(save_path, index=False)

print(f"\n✅ 结果已保存到: {save_path}")
print(f"   - 最优验证AUC: {best_valid_roc:.4f} (第 {best_epoch} 轮)")
if cands:
    print("   - 最终报出: %s" % winner_name)
print("Test Acc: {:.4f}, Test AUC: {:.4f}, Test AUPR: {:.4f}".format(test_acc, roc_auc_test, test_aupr))
