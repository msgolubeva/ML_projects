# dr_model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ==========================
#   Общие хелперы
# ==========================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

mu = 1.0/3.0

def make_action_features(X: np.ndarray, actions: np.ndarray, n_actions: int) -> np.ndarray:
    """
    Строим расширенные признаки [X, one_hot(a)].
    X      : [n, d]
    actions: [n] (int)
    -> [n, d + n_actions]
    """
    X = np.asarray(X)
    actions = np.asarray(actions, dtype=int)
    n, d = X.shape
    one_hot = np.zeros((n, n_actions), dtype=np.float32)
    one_hot[np.arange(n), actions] = 1.0
    return np.concatenate([X, one_hot], axis=1)


def evaluate_best_static(actions: np.ndarray,
                         rewards: np.ndarray,
                         n_actions: int,
                         mu: float) -> tuple[int, float]:
    """
    Находим лучшую статическую политику: всегда выбирать один arm.
    Возвращаем (argmax_a V_static(a), V_best_static).
    IPS формально тут не нужен, т.к. μ константа и сокращается.
    """
    actions = np.asarray(actions, dtype=int)
    rewards = np.asarray(rewards, dtype=float)

    best_a = 0
    best_v = -1e9
    for a in range(n_actions):
        mask = actions == a
        if mask.sum() == 0:
            continue
        v = rewards[mask].mean()  # эквивалент IPS при константном μ
        if v > best_v:
            best_v = v
            best_a = a
    return best_a, best_v


def snips_from_probs(mu: float,
                     actions: np.ndarray,
                     rewards: np.ndarray,
                     policy_probs: np.ndarray) -> float:
    """
    SNIPS-оценка V^SNIPS(π) по готовым вероятностям π(a|x).
    mu — вероятность действия у логирующей политики (здесь 1/3).
    """
    actions = np.asarray(actions, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    policy_probs = np.asarray(policy_probs, dtype=float)

    pi_a = policy_probs[np.arange(len(actions)), actions]
    w = pi_a / mu
    num = np.sum(w * rewards)
    den = np.sum(w) + 1e-8
    return float(num / den)


def offline_score_from_probs(mu: float,
                             actions: np.ndarray,
                             rewards: np.ndarray,
                             policy_probs: np.ndarray,
                             n_actions: int) -> tuple[float, float, float]:
    """
    Основная метрика соревнования:
        score = V_SNIPS(π) - V_best_static
    """
    v_snips = snips_from_probs(mu, actions, rewards, policy_probs)
    _, v_best_static = evaluate_best_static(actions, rewards, n_actions, mu)
    score = v_snips - v_best_static
    return score, v_snips, v_best_static


# ==========================
#   RewardNet : q̂(x, a)
# ==========================

class RewardNet(nn.Module):
    """
    Модель q̂(x,a) ~ P(r=1 | x,a).
    На вход подаём [X, one_hot(a)].
    """
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def train_reward_model(X_train: np.ndarray,
                       a_train: np.ndarray,
                       r_train: np.ndarray,
                       X_val: np.ndarray,
                       a_val: np.ndarray,
                       r_val: np.ndarray,
                       n_actions: int,
                       device: str = "cpu",
                       seed: int = 42) -> RewardNet:
    """
    Обучаем RewardNet по BCE с учётом дисбаланса классов.
    """
    set_seed(seed)

    # расширяем признаки
    Xtr_ext = make_action_features(X_train, a_train, n_actions)
    Xva_ext = make_action_features(X_val, a_val, n_actions)

    Xtr_t = torch.tensor(Xtr_ext, dtype=torch.float32)
    Xva_t = torch.tensor(Xva_ext, dtype=torch.float32)
    ytr_t = torch.tensor(r_train, dtype=torch.float32)
    yva_t = torch.tensor(r_val, dtype=torch.float32)

    dataset_tr = TensorDataset(Xtr_t, ytr_t)
    dataset_va = TensorDataset(Xva_t, yva_t)

    loader_tr = DataLoader(dataset_tr, batch_size=4096, shuffle=True)
    loader_va = DataLoader(dataset_va, batch_size=4096, shuffle=False)

    in_dim = Xtr_ext.shape[1]
    model = RewardNet(in_dim=in_dim).to(device)

    # pos_weight для борьбы с дисбалансом
    pos_frac = float(r_train.mean() + 1e-6)
    pos_weight = (1.0 - pos_frac) / pos_frac
    print(f"[DR] RewardNet pos_weight = {pos_weight:.2f}")

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best_val = 1e9
    patience = 5
    stale = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    EPOCHS = 50
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in loader_tr:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(dataset_tr)

        # валидация
        model.eval()
        with torch.no_grad():
            va_loss = 0.0
            for xb, yb in loader_va:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = bce(logits, yb)
                va_loss += loss.item() * xb.size(0)
            va_loss /= len(dataset_va)

        print(f"[DR] epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("[DR] Early stopping.")
                break

    print(f"[DR] RewardNet best val_loss = {best_val:.4f}")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model


def reward_net_to_q_matrix(model: RewardNet,
                           X: np.ndarray,
                           n_actions: int,
                           device: str = "cpu") -> np.ndarray:
    """
    Получаем матрицу q̂(x,a) для всех x и всех actions.
    Возвращаем shape [n, n_actions].
    """
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    # дублируем X по действиям
    X_rep = np.repeat(X, n_actions, axis=0)
    actions_rep = np.tile(np.arange(n_actions), n)
    feats = make_action_features(X_rep, actions_rep, n_actions)
    with torch.no_grad():
        logits = model(torch.tensor(feats, dtype=torch.float32, device=device))
        probs = torch.sigmoid(logits).cpu().numpy().reshape(n, n_actions)
    return probs  # q̂(x,a) \approx P(r=1 | x,a)


# ==========================
#   PolicyNet с регуляризацией к softmax(q̂)
# ==========================

class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3),  # 3 действия
        )

    def forward(self, x):
        return self.net(x)  # логиты [B,3]


def snips_objective_from_logits(logits: torch.Tensor,
                                actions: torch.Tensor,
                                rewards: torch.Tensor,
                                mu,
                                clip: float = 10.0,
                                entropy_coef: float = 2e-3,
                                temp: float = 1.0):
    """
    logits: [B,3]
    actions: [B] long
    rewards: [B] float
    mu: вероятность действия у логирующей политики (может прийти строкой)
    """
    # на всякий случай приводим к float
    mu = float(mu)

    probs = F.softmax(logits / temp, dim=1)              # πθ
    pi_a = probs[torch.arange(probs.size(0)), actions]   # πθ(a_i|x_i)

    w = (pi_a / mu).clamp(max=clip)
    num = torch.sum(w * rewards)
    den = torch.sum(w) + 1e-8
    snips = num / den

    entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=1))
    loss = -(snips + entropy_coef * entropy)
    return loss, snips.detach(), entropy.detach(), probs



def train_policy_with_teacher(X_train: np.ndarray,
                              a_train: np.ndarray,
                              r_train: np.ndarray,
                              X_val: np.ndarray,
                              a_val: np.ndarray,
                              r_val: np.ndarray,
                              q_train: np.ndarray,
                              q_val: np.ndarray,
                              mu: float,
                              device: str = "cpu",
                              seed: int = 42,
                              tau_q: float = 0.7,
                              lambda_align: float = 0.1) -> tuple[PolicyNet, dict]:
    """
    Обучаем PolicyNet по SNIPS + KL(πθ || π_q), где π_q строится из q̂.
    """
    set_seed(seed)

    n_features = X_train.shape[1]
    model = PolicyNet(in_dim=n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    Xva_t = torch.tensor(X_val, dtype=torch.float32)
    atr = np.asarray(a_train, dtype=int)
    ava = np.asarray(a_val, dtype=int)
    rtr = np.asarray(r_train, dtype=float)
    rva = np.asarray(r_val, dtype=float)

    qtr = np.asarray(q_train, dtype=np.float32)
    qva = np.asarray(q_val, dtype=np.float32)

    BATCH = 4096
    EPOCHS = 60
    patience = 7
    best_val = -1e9
    stale = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    hist = {"val_snips": []}

    n = X_train.shape[0]
    indices = np.arange(n)

    _, v_best_static = evaluate_best_static(a_val, r_val, n_actions=3, mu=mu)

    for epoch in range(1, EPOCHS + 1):
        np.random.shuffle(indices)
        model.train()
        tr_snips = 0.0

        for i in range(0, n, BATCH):
            batch_idx = indices[i:i + BATCH]
            xb = Xtr_t[batch_idx].to(device)
            ab = torch.tensor(atr[batch_idx], dtype=torch.long, device=device)
            rb = torch.tensor(rtr[batch_idx], dtype=torch.float32, device=device)
            qb = torch.tensor(qtr[batch_idx], dtype=torch.float32, device=device)

            opt.zero_grad()
            logits = model(xb)
            loss_snips, snips_val, _, probs_pi = snips_objective_from_logits(
                logits, ab, rb, mu=mu, clip=10.0, entropy_coef=2e-3, temp=1.0
            )

            # teacher policy π_q
            pi_q = F.softmax(qb / tau_q, dim=1)
            log_pi = torch.log(probs_pi + 1e-12)
            log_pi_q = torch.log(pi_q + 1e-12)
            kl = torch.mean(torch.sum(probs_pi * (log_pi - log_pi_q), dim=1))

            loss = loss_snips + lambda_align * kl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tr_snips += snips_val.item()

        # validation: считаем SNIPS по валу
        model.eval()
        with torch.no_grad():
            logits_va = model(Xva_t.to(device))
            probs_va = F.softmax(logits_va, dim=1).cpu().numpy()
            _, v_snips, _ = offline_score_from_probs(
                mu=mu,
                actions=a_val,
                rewards=r_val,
                policy_probs=probs_va,
                n_actions=3,
            )
        hist["val_snips"].append(v_snips)
        score_epoch = v_snips - v_best_static

        print(
            f"[PolicyNet] epoch {epoch:02d} | "
            f"val_SNIPS={v_snips:.6f} | "
            f"val_score={score_epoch:.6f}"
        )

        if v_snips > best_val + 1e-6:
            best_val = v_snips
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("[PolicyNet] Early stopping.")
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, hist


def predict_policy_probs(model: PolicyNet,
                         X: np.ndarray,
                         device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    # нормировка на всякий случай
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.clip(row_sums, 1e-12, None)
    return probs


# ==========================
#   Главная точка входа: run_dr
# ==========================

def run_dr(X_train: np.ndarray,
           X_val: np.ndarray,
           a_train: np.ndarray,
           a_val: np.ndarray,
           r_train: np.ndarray,
           r_val: np.ndarray,
           X_test: np.ndarray,
           device: str = "cpu",
           seed: int = 42) -> np.ndarray:
    """
    Основной запуск:
      1) обучаем RewardNet;
      2) по ней считаем q̂(x,a) на train/val;
      3) обучаем PolicyNet по SNIPS + KL(πθ || π_q);
      4) считаем offline-скор на валидации;
      5) выдаём вероятности действий на test.
    """
    set_seed(seed)
    n_actions = 3

    print("[DR/Policy+Teacher] shapes:",
          f"Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 1) RewardNet
    reward_model = train_reward_model(
        X_train, a_train, r_train,
        X_val, a_val, r_val,
        n_actions=n_actions,
        device=device,
        seed=seed,
    )

    # 2) q̂(x,a) на train/val
    q_train = reward_net_to_q_matrix(reward_model, X_train, n_actions, device=device)
    q_val = reward_net_to_q_matrix(reward_model, X_val, n_actions, device=device)

    # 3) PolicyNet c регуляризацией
    policy_model, _ = train_policy_with_teacher(
        X_train, a_train, r_train,
        X_val, a_val, r_val,
        q_train, q_val,
        mu=mu,
        device=device,
        seed=seed,
        tau_q=0.7,
        lambda_align=0.1,
    )

    # 4) offline-скор на валидации для чистого PolicyNet
    probs_val = predict_policy_probs(policy_model, X_val, device=device)
    base_score, base_v_snips, base_v_best_static = offline_score_from_probs(
        mu=mu,
        actions=a_val,
        rewards=r_val,
        policy_probs=probs_val,
        n_actions=n_actions,
    )

    # найдём лучшую статическую политику (на всякий случай — именно по val)
    best_arm, v_best_static = evaluate_best_static(
        actions=a_val,
        rewards=r_val,
        n_actions=n_actions,
        mu=mu,
    )

    print("======================================================")
    print(f"[DR/Policy+Teacher] PolicyNet only | score={base_score:.6f}")
    print(f"[DR/Policy+Teacher]   V_SNIPS(π)    = {base_v_snips:.6f}")
    print(f"[DR/Policy+Teacher]   V_best_static = {v_best_static:.6f}")
    print("======================================================")

    # 4.1) Миксуем PolicyNet с лучшим статиком
    alphas = np.linspace(0.001, 0.1, 100)
    best_mix_score = base_score
    best_alpha = 0.0
    best_probs_val = probs_val

    # статическая политика: всегда best_arm
    static_probs_val = np.zeros_like(probs_val)
    static_probs_val[:, best_arm] = 1.0

    for alpha in alphas:
        mix_val = (1.0 - alpha) * static_probs_val + alpha * probs_val
        # нормировка на всякий случай
        mix_val = mix_val / np.clip(mix_val.sum(axis=1, keepdims=True), 1e-12, None)

        mix_score, mix_v_snips, mix_v_best_static = offline_score_from_probs(
            mu=mu,
            actions=a_val,
            rewards=r_val,
            policy_probs=mix_val,
            n_actions=n_actions,
        )

        print(
            f"[DR/Mix] alpha={alpha:.2f} | "
            f"score={mix_score:.6f} | "
            f"V_SNIPS={mix_v_snips:.6f}"
        )

        if mix_score > best_mix_score + 1e-6:
            best_mix_score = mix_score
            best_alpha = alpha
            best_probs_val = mix_val

    print("======================================================")
    print(f"[DR/Policy+Teacher] Best mixture on val:")
    print(f"   best_alpha    = {best_alpha:.2f}")
    print(f"   best_mix_score= {best_mix_score:.6f}")
    print("======================================================")

    # 5) Предсказания на test
    probs_test = predict_policy_probs(policy_model, X_test, device=device)

    if best_alpha > 0.0:
        static_probs_test = np.zeros_like(probs_test)
        static_probs_test[:, best_arm] = 1.0

        probs_test = (1.0 - best_alpha) * static_probs_test + best_alpha * probs_test
        probs_test = probs_test / np.clip(probs_test.sum(axis=1, keepdims=True), 1e-12, None)

    return probs_test

