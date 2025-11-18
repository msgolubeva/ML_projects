import numpy as np

def make_action_features(X, actions, n_actions=3):
    """
    X        : [n, d]  — исходные фичи контекста
    actions  : [n]     — 0/1/2 (действие, которое реально сделали)
    return   : [n, d + n_actions] — X с one-hot по действию
    """
    X = np.asarray(X, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)

    n = X.shape[0]
    one_hot = np.zeros((n, n_actions), dtype=np.float32)
    one_hot[np.arange(n), actions] = 1.0

    return np.concatenate([X, one_hot], axis=1)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class DMNet(nn.Module):
    """
    Модель q(x,a) ~ P(visit=1 | x,a) по признакам [x, onehot(a)].
    """
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)   # логит вероятности visit
        )

    def forward(self, x):
        # x: [B, in_dim]
        logits = self.mlp(x)   # [B, 1]
        return logits.squeeze(-1)  # [B]


def train_dm_model(
        X_tr, a_tr, r_tr,
        X_val, a_val, r_val,
        DEVICE="cpu",
        n_actions=3,
        batch_size=4096,
        epochs=40,
        lr=1e-3,
        weight_decay=1e-5,
        patience=6,
):
    # ------- расширяем фичи -------
    X_tr_ext  = make_action_features(X_tr, a_tr, n_actions)
    X_val_ext = make_action_features(X_val, a_val, n_actions)

    in_dim = X_tr_ext.shape[1]
    model = DMNet(in_dim).to(DEVICE)

    # pos_weight: сколько раз 0 больше, чем 1
    r_tr_np = np.asarray(r_tr, dtype=np.float32)
    n_pos = r_tr_np.sum()
    n_neg = len(r_tr_np) - n_pos
    pos_weight_val = n_neg / max(n_pos, 1.0)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=DEVICE)
    print(f"[DM] pos_weight = {pos_weight_val:.2f}")

    train_ds = TensorDataset(
        torch.tensor(X_tr_ext, dtype=torch.float32),
        torch.tensor(r_tr,     dtype=torch.float32),
    )
    val_x = torch.tensor(X_val_ext, dtype=torch.float32, device=DEVICE)
    val_y = torch.tensor(r_val,     dtype=torch.float32, device=DEVICE)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y).item()

        print(f"[DM] epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("[DM] Early stopping.")
                break

    print(f"[DM] best val_loss = {best_val:.4f}")
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    return model

def build_policy_from_q(q_mat, temp=1.0, mix_with_logging=0.2):
    """
    q_mat : [n, n_actions] — оценки P(visit|x,a) или что-то пропорциональное.
    temp  : температура softmax ( >1 — мягче, <1 — жёстче).
    mix_with_logging : доля примеси логирующей (uniform) политики.
    """
    q = np.asarray(q_mat, dtype=np.float32)
    logits = q / temp
    logits = logits - logits.max(axis=1, keepdims=True)  # стабильность
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)         # чистый softmax

    if mix_with_logging > 0.0:
        uniform = 1/3 * np.ones_like(probs)
        probs = (1.0 - mix_with_logging) * probs + mix_with_logging * uniform

    # нормировка
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.clip(row_sums, 1e-12, None)
    return probs

def dm_policy_probs(model, X, n_actions=3, temp=1.5, mix_with_logging=0.2, DEVICE="cpu"):
    """
    Строим q(x,a) и затем безопасную политику π(a|x).
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    all_q = []

    model.to(DEVICE).eval()
    with torch.no_grad():
        for a in range(n_actions):
            acts = np.full(shape=(n,), fill_value=a, dtype=np.int64)
            X_ext = make_action_features(X, acts, n_actions)
            x_t = torch.tensor(X_ext, dtype=torch.float32, device=DEVICE)
            logits = model(x_t)                       # [n]
            q = torch.sigmoid(logits).cpu().numpy()   # P(visit|x,a)
            all_q.append(q.reshape(-1, 1))

    q_mat = np.concatenate(all_q, axis=1)  # [n, n_actions]
    probs = build_policy_from_q(q_mat, temp=temp, mix_with_logging=mix_with_logging)
    return probs.astype(np.float32)


def evaluate_best_static(actions, rewards, mu=1.0/3.0, n_actions=3):
    v_static = np.zeros(n_actions, dtype=float)
    for a in range(n_actions):
        mask = (actions == a)
        if not np.any(mask):
            v_static[a] = 0.0
            continue

        r_a = rewards[mask].astype(float)
        num = np.sum(r_a / mu)
        den = np.sum(np.ones_like(r_a) / mu)
        v_static[a] = num / (den + 1e-8)

    v_best = v_static.max()
    return v_static, v_best

from sklearn.linear_model import LogisticRegression

def fit_lr_per_action(X, actions, rewards, C=1.0):
    """
    Для каждого действия a учим LogisticRegression на тех примерах,
    где это действие реально было выбрано.
    """
    X = np.asarray(X, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.int64)

    models = []
    for a in range(3):
        mask = (actions == a)
        X_a = X[mask]
        y_a = rewards[mask]
        if len(y_a) == 0:
            models.append(None)
            continue

        clf = LogisticRegression(
            max_iter=1000,
            C=C,
            solver="lbfgs",
        )
        clf.fit(X_a, y_a)
        models.append(clf)
    return models


def lr_policy_probs(models, X, temp=1.5, mix_with_logging=0.2):
    """
    Строит политику на основе трёх логрегов.
    """
    X = np.asarray(X, dtype=np.float32)
    qs = []
    for a, clf in enumerate(models):
        if clf is None:
            qs.append(np.zeros((X.shape[0], 1), dtype=np.float32))
        else:
            proba = clf.predict_proba(X)[:, 1]   # P(visit=1)
            qs.append(proba.reshape(-1, 1))
    q_mat = np.concatenate(qs, axis=1)
    return build_policy_from_q(q_mat, temp=temp, mix_with_logging=mix_with_logging)

def offline_score_from_probs(probs, actions, rewards, mu=1/3, n_actions=3):
    probs = np.asarray(probs, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)

    pi_ai = probs[np.arange(len(actions)), actions]  # π(a_i|x_i)
    w = pi_ai / mu
    num = np.sum(w * rewards)
    den = np.sum(w) + 1e-8
    v_snips = num / den

    _, v_best_static = evaluate_best_static(actions, rewards, mu=mu, n_actions=n_actions)
    score = v_snips - v_best_static
    return score, v_snips, v_best_static


def offline_score_dm(model, X, actions, rewards, mu=1.0/3.0, n_actions=3, temp=1.0, DEVICE="cpu"):
    """
    Считает:
      score = V_SNIPS(π) - V_best_static
    где π — политика, построенная из DM-модели.
    """
    # π(a|x) для ВСЕХ действий
    probs = dm_policy_probs(model, X, n_actions=n_actions, temp=temp, DEVICE=DEVICE)  # [n,3]
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)

    pi_ai = probs[np.arange(len(actions)), actions]  # π(a_i|x_i)

    w = pi_ai / mu
    num = np.sum(w * rewards)
    den = np.sum(w) + 1e-8
    v_snips = num / den

    _, v_best_static = evaluate_best_static(actions, rewards, mu=mu, n_actions=n_actions)
    score = v_snips - v_best_static
    return score, v_snips, v_best_static



def run_dm(X_train, a_train, r_train,
           X_val, a_val, r_val, X_data_test, DEVICE="cpu", MU=1/3):

    dm_model = train_dm_model(
        X_train, a_train, r_train,
        X_val, a_val, r_val,
        DEVICE=DEVICE,
        n_actions=3,
        batch_size=4096,
        epochs=40,
        lr=1e-3,
        weight_decay=1e-5,
        patience=6,
    )

    # --- исправлено ---
    probs_val = dm_policy_probs(dm_model, X_val, n_actions=3, temp=1.0, DEVICE=DEVICE)
    score_val, v_snips_val, v_best_static_val = offline_score_from_probs(
        probs_val, a_val, r_val, mu=MU, n_actions=3
    )

    print("=" * 60)
    print(f"[DM] Val score = {score_val:.6f}")
    print("=" * 60)

    # логрег baseline
    lr_models = fit_lr_per_action(X_train, a_train, r_train)
    probs_val_lr = lr_policy_probs(lr_models, X_val, temp=1.5, mix_with_logging=0.2)
    score_val_lr, _, _ = offline_score_from_probs(
        probs_val_lr, a_val, r_val, mu=MU, n_actions=3
    )
    print("LR score:", score_val_lr)

    # FINALLY TEST
    probs_test = dm_policy_probs(dm_model, X_data_test, n_actions=3, temp=1.0, DEVICE=DEVICE)
    return probs_test
