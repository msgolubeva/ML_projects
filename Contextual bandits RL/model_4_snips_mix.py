# # snips_mix_teacher_model.py
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils import evaluate_best_static  # уже есть в твоём проекте
# from sklearn.ensemble import HistGradientBoostingClassifier
#
#
# # ===========================
# #  RewardNet (учитель q(x,a))
# # ===========================
#
# # ===========================
# #  GBM-учитель q(x,a)
# # ===========================
#
# def train_gbm_teacher(X, a, r, n_actions=3):
#     """
#     Для каждого действия a учим отдельный бустинг:
#     q_a(x) = P(r=1 | x, a)
#     Возвращает список моделей длины n_actions.
#     """
#     X = np.asarray(X, dtype=np.float32)
#     a = np.asarray(a, dtype=np.int64)
#     r = np.asarray(r, dtype=np.int64)  # для классификатора
#
#     models = []
#     for k in range(n_actions):
#         mask = (a == k)
#         X_k = X[mask]
#         r_k = r[mask]
#
#         clf = HistGradientBoostingClassifier(
#             max_depth=5,
#             learning_rate=0.05,
#             max_iter=250,
#             min_samples_leaf=20,
#             l2_regularization=1.0,
#             random_state=42,
#         )
#         clf.fit(X_k, r_k)
#         models.append(clf)
#
#         print(f"[GBM teacher] action={k} | n={len(X_k)}")
#
#     return models
#
#
# def gbm_q_matrix(models, X):
#     """
#     models: список из 3 бустингов (по одному на arm)
#     X: [n,d]
#     -> q(x,a): [n,3]
#     """
#     X = np.asarray(X, dtype=np.float32)
#     n = X.shape[0]
#     n_actions = len(models)
#     q = np.zeros((n, n_actions), dtype=np.float32)
#
#     for k, clf in enumerate(models):
#         # predict_proba[:,1] = P(r=1)
#         q[:, k] = clf.predict_proba(X)[:, 1]
#
#     return q
#
#
# def _make_action_features(X, actions, n_actions=3):
#     """
#     X: [n, d]
#     actions: [n] int
#     -> [n, d + n_actions] (one-hot по action)
#     """
#     X = np.asarray(X, dtype=np.float32)
#     actions = np.asarray(actions, dtype=np.int64)
#     n, d = X.shape
#     oh = np.zeros((n, n_actions), dtype=np.float32)
#     oh[np.arange(n), actions] = 1.0
#     return np.concatenate([X, oh], axis=1)
#
#
# def train_reward_net(X, a, r, device="cpu",
#                      hidden=128, lr=1e-3,
#                      weight_decay=1e-4,
#                      epochs=20, batch_size=2048):
#     """
#     Обучаем q(x,a) по обычному BCE, без IPS.
#     """
#     X_ext = _make_action_features(X, a)
#     X_t = torch.tensor(X_ext, dtype=torch.float32, device=device)
#     r_t = torch.tensor(r, dtype=torch.float32, device=device)
#
#     ds = torch.utils.data.TensorDataset(X_t, r_t)
#     dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
#
#     model = RewardNet(X_ext.shape[1], hidden=hidden).to(device)
#
#     pos_frac = float(r.mean())
#     pos_weight = (1 - pos_frac) / max(pos_frac, 1e-6) / 3
#     print(pos_weight)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,
#                                                              device=device))
#     opt = torch.optim.Adam(model.parameters(), lr=lr,
#                            weight_decay=weight_decay)
#
#     for epoch in range(1, epochs + 1):
#         model.train()
#         tot_loss = 0.0
#         n_samples = 0
#         for xb, rb in dl:
#             opt.zero_grad()
#             logits = model(xb)
#             loss = criterion(logits, rb)
#             loss.backward()
#             opt.step()
#             tot_loss += loss.item() * xb.size(0)
#             n_samples += xb.size(0)
#         print(f"[RewardNet] epoch {epoch:02d} | loss={tot_loss / n_samples:.4f}")
#
#     model.eval()
#     return model
#
#
# def predict_q_matrix(model, X, device="cpu", n_actions=3):
#     """
#     Возвращает q_hat(x,a) для всех a: [n, n_actions]
#     """
#     X = np.asarray(X, dtype=np.float32)
#     n, d = X.shape
#     # дублируем X по действиям
#     X_rep = np.repeat(X, n_actions, axis=0)
#     a_rep = np.tile(np.arange(n_actions, dtype=np.int64), n)
#     X_ext = _make_action_features(X_rep, a_rep, n_actions=n_actions)
#
#     with torch.no_grad():
#         xb = torch.tensor(X_ext, dtype=torch.float32, device=device)
#         logits = model(xb).cpu().numpy().reshape(n, n_actions)
#         probs = 1 / (1 + np.exp(-logits))  # сигмоида
#
#     return probs  # q_hat(x,a)
#
#
# # ===========================
# #  PolicyNet + SNIPS
# # ===========================
#
# class PolicyNet(nn.Module):
#     def __init__(self, in_dim, n_actions=3, hidden=128, dropout=0.1, mu=1/3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, hidden // 2),
#             nn.ReLU(),
#             nn.Linear(hidden // 2, n_actions)
#         )
#         self.mu = mu
#         self.n_actions = n_actions
#
#     def forward(self, x):
#         return self.net(x)  # логиты [B,3]
#
#     def snips_objective(
#             self,
#             logits,
#             a,
#             r,
#             clip=10.0,
#             entropy_coef=1e-3,
#             temp=1.0,
#     ):
#         probs = F.softmax(logits / temp, dim=1)
#         pi_a = probs[torch.arange(probs.size(0)), a]
#         w = (pi_a / self.mu).clamp(max=clip)
#         snips = torch.sum(w * r) / (torch.sum(w) + 1e-8)
#
#         entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=1))
#         loss = -(snips + entropy_coef * entropy)
#         return loss, snips.detach(), entropy.detach()
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         self.eval()
#         with torch.no_grad():
#             xb = torch.tensor(X, dtype=torch.float32)
#             logits = self(xb)
#             probs = F.softmax(logits, dim=1).cpu().numpy()
#         probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
#         return probs
#
#
# def _compute_snips(probs, actions, rewards, mu):
#     actions = np.asarray(actions, dtype=np.int64)
#     rewards = np.asarray(rewards, dtype=np.float32)
#
#     pi_a = probs[np.arange(len(actions)), actions]
#     w = pi_a / mu
#     num = np.sum(w * rewards)
#     den = np.sum(w) + 1e-8
#     return num / den
#
#
# def _iterate_minibatches(X, a, r, batch_size):
#     idx = np.random.permutation(X.shape[0])
#     for i in range(0, X.shape[0], batch_size):
#         j = idx[i : i + batch_size]
#         yield X[j], a[j], r[j]
#
#
# def train_policy_with_teacher(
#         X_train,
#         X_val,
#         a_train,
#         a_val,
#         r_train,
#         r_val,
#         teacher_probs_train,
#         teacher_probs_val,
#         mu,
#         device="cpu",
#         n_actions=3,
#         hidden=128,
#         dropout=0.1,
#         lr=3e-3,
#         weight_decay=1e-4,
#         epochs=50,
#         batch_size=4096,
#         patience=7,
#         lambda_align=0.2,
# ):
#     """
#     SNIPS + KL(teacher || policy) регуляризация.
#     """
#     in_dim = X_train.shape[1]
#     model = PolicyNet(in_dim, n_actions=n_actions,
#                       hidden=hidden, dropout=dropout, mu=mu).to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#     # best_static по валу
#     v_per_arm, v_best_static = evaluate_best_static(
#         actions=a_val,
#         rewards=r_val,
#         mu=mu,
#         n_actions=n_actions,
#     )
#
#     best_val_snips = -1e9
#     best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#     stale = 0
#
#     teacher_train = np.asarray(teacher_probs_train, dtype=np.float32)
#     teacher_val = np.asarray(teacher_probs_val, dtype=np.float32)
#
#     for epoch in range(1, epochs + 1):
#         model.train()
#         tr_snips = tr_kl = 0.0
#         n_batches = 0
#
#         for xb, ab, rb in _iterate_minibatches(X_train, a_train, r_train, batch_size):
#             tb = teacher_train[np.random.permutation(len(xb))[: len(xb)]]
#             xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
#             ab_t = torch.tensor(ab, dtype=torch.long, device=device)
#             rb_t = torch.tensor(rb, dtype=torch.float32, device=device)
#             tb_t = torch.tensor(tb, dtype=torch.float32, device=device)
#
#             opt.zero_grad()
#             logits = model(xb_t)
#             loss_snips, snips_val, _ = model.snips_objective(
#                 logits, ab_t, rb_t, clip=10.0, entropy_coef=2e-3, temp=1.0
#             )
#
#             # KL(teacher || policy)
#             probs = F.softmax(logits, dim=1)
#             kl = torch.mean(
#                 torch.sum(tb_t * (torch.log(tb_t + 1e-12) - torch.log(probs + 1e-12)),
#                           dim=1)
#             )
#
#             loss = loss_snips + lambda_align * kl
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             opt.step()
#
#             tr_snips += snips_val.item()
#             tr_kl += kl.item()
#             n_batches += 1
#
#         tr_snips /= max(n_batches, 1)
#         tr_kl /= max(n_batches, 1)
#
#         # ---- валидация ----
#         model.eval()
#         with torch.no_grad():
#             xb = torch.tensor(X_val, dtype=torch.float32, device=device)
#             ab = torch.tensor(a_val, dtype=torch.long, device=device)
#             rb = torch.tensor(r_val, dtype=torch.float32, device=device)
#             logits = model(xb)
#             _, va_snips_t, _ = model.snips_objective(
#                 logits, ab, rb, clip=10.0, entropy_coef=0.0, temp=1.0
#             )
#
#         va_snips = va_snips_t.item()
#         va_score = va_snips - v_best_static
#
#         print(
#             f"[Policy] epoch {epoch:02d} | "
#             f"train_SNIPS={tr_snips:.4f} | "
#             f"train_KL={tr_kl:.4f} | "
#             f"val_SNIPS={va_snips:.4f} | "
#             f"val_score={va_score:.6f}"
#         )
#
#         if va_snips > best_val_snips + 1e-6:
#             best_val_snips = va_snips
#             best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             stale = 0
#         else:
#             stale += 1
#             if stale >= patience:
#                 print("Early stopping.")
#                 break
#
#     model.load_state_dict(best_state)
#     model.to("cpu")
#     model.eval()
#
#     return model, v_per_arm, v_best_static
#
#
# # ===========================
# #  Главный раннер
# # ===========================
#
# def run(
#         X_train: np.ndarray,
#         X_val: np.ndarray,
#         a_train: np.ndarray,
#         a_val: np.ndarray,
#         r_train: np.ndarray,
#         r_val: np.ndarray,
#         X_test: np.ndarray,
#         MU: float,
#         DEVICE: str,
# ):
#     n_actions = 3
#
#     # 1) RewardNet (учитель)
#     print("==== Train GBM teacher ====")
#     gbm_models = train_gbm_teacher(X_train, a_train, r_train, n_actions=3)
#
#     q_train = gbm_q_matrix(gbm_models, X_train)
#     q_val   = gbm_q_matrix(gbm_models, X_val)
#     q_test  = gbm_q_matrix(gbm_models, X_test)
#
#
#     # teacher policy π_q (softmax по q/τ_q)
#     tau_q = 0.9
#     def _softmax_q(q):
#         z = q / tau_q
#         z = z - np.max(z, axis=1, keepdims=True)
#         e = np.exp(z)
#         return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)
#
#     teacher_train = _softmax_q(q_train)
#     teacher_val = _softmax_q(q_val)
#     teacher_test = _softmax_q(q_test)  # пригодится, если захочешь его посмотреть
#
#     # 2) PolicyNet с SNIPS + KL к учителю
#     print("==== Train PolicyNet with teacher & SNIPS ====")
#     policy, v_per_arm, v_best_static = train_policy_with_teacher(
#         X_train=X_train,
#         X_val=X_val,
#         a_train=a_train,
#         a_val=a_val,
#         r_train=r_train,
#         r_val=r_val,
#         teacher_probs_train=teacher_train,
#         teacher_probs_val=teacher_val,
#         mu=MU,
#         device=DEVICE,
#         n_actions=n_actions,
#         hidden=128,
#         dropout=0.1,
#         lr=3e-3,
#         weight_decay=1e-4,
#         epochs=50,
#         batch_size=2048,
#         patience=7,
#         lambda_align=0.2,
#     )
#
#     # 3) SNIPS для чистого PolicyNet + лучшая статическая
#     probs_val = policy.predict(X_val)
#
#     v_snips_base = _compute_snips(probs_val, a_val, r_val, MU)
#     base_score = v_snips_base - v_best_static
#     best_arm = int(np.argmax(v_per_arm))
#
#     print("======================================================")
#     print(f"[PolicyNet only] V_SNIPS(π)    = {v_snips_base:.6f}")
#     print(f"[PolicyNet only] V_best_static = {v_best_static:.6f}")
#     print(f"[PolicyNet only] score         = {base_score:.6f}")
#     print(f"Best static arm on val = {best_arm}, "
#           f"V_static={v_per_arm[best_arm]:.6f}")
#     print("======================================================")
#
#     # 4) Микс со статиком
#     alphas = np.linspace(0, 1, 30)
#     static_val = np.zeros_like(probs_val)
#     static_val[:, best_arm] = 1.0
#
#     best_alpha = 0.0
#     best_mix_score = base_score
#     best_mix_snips = v_snips_base
#
#     for alpha in alphas:
#         mix_val = (1.0 - alpha) * static_val + alpha * probs_val
#         mix_val = mix_val / np.clip(mix_val.sum(axis=1, keepdims=True), 1e-12, None)
#
#         v_mix = _compute_snips(mix_val, a_val, r_val, MU)
#         score_mix = v_mix - v_best_static
#
#         print(
#             f"[Mix alpha={alpha:.2f}] "
#             f"V_SNIPS={v_mix:.6f} | score={score_mix:.6f}"
#         )
#
#         if score_mix > best_mix_score + 1e-6:
#             best_mix_score = score_mix
#             best_mix_snips = v_mix
#             best_alpha = alpha
#
#     print("======================================================")
#     print(f"[Best mixture] alpha   = {best_alpha:.2f}")
#     print(f"[Best mixture] V_SNIPS = {best_mix_snips:.6f}")
#     print(f"[Best mixture] score   = {best_mix_score:.6f}")
#     print("======================================================")
#
#     # 5) Предсказания на test
#     probs_test = policy.predict(X_test)
#
#     if best_alpha > 0.0:
#         static_test = np.zeros_like(probs_test)
#         static_test[:, best_arm] = 1.0
#         probs_test = (1.0 - best_alpha) * static_test + best_alpha * probs_test
#         probs_test = probs_test / np.clip(
#             probs_test.sum(axis=1, keepdims=True), 1e-12, None
#         )
#
#     return probs_test

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

from utils import evaluate_best_static


# ===========================
#  GBM-учитель q(x,a)
# ===========================

def tune_histgb_reward_models(X, a, r, n_actions=3, random_state=42, n_iter=20):
    """
    Подбираем reward-модель для каждого arm с помощью HistGradientBoostingClassifier.

    X : np.ndarray [n, d] – уже предобработанные признаки (после process_data)
    a : np.ndarray [n]    – действия (0,1,2)
    r : np.ndarray [n]    – награды (0/1)

    Возвращает:
        models : list из n_actions обученных HGB-моделей (по одной на arm).
    """
    X = np.asarray(X, dtype=np.float32)
    a = np.asarray(a, dtype=np.int64)
    r = np.asarray(r, dtype=np.int64)

    models = []

    for arm in range(n_actions):
        print("=" * 60)
        print(f"[HistGB Reward] arm = {arm}")

        mask = (a == arm)
        X_arm = X[mask]
        r_arm = r[mask]

        print(f"  samples for arm {arm}: {X_arm.shape[0]}")

        if X_arm.shape[0] == 0:
            # На всякий случай – если вдруг для arm нет примеров.
            print(f"  WARNING: no samples for arm {arm}, using dummy model.")
            dummy = HistGradientBoostingClassifier(random_state=random_state)
            dummy.classes_ = np.array([0, 1])
            dummy.n_features_in_ = X.shape[1]
            models.append(dummy)
            continue

        # train/val внутри данного arm, чтобы выбирать гиперы по AUC
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_arm,
                r_arm,
                test_size=0.2,
                random_state=random_state,
                stratify=r_arm if r_arm.sum() > 0 else None,
            )
        except ValueError:
            # если stratify не сработал (все 0 или все 1) – без стратификации
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_arm,
                r_arm,
                test_size=0.2,
                random_state=random_state,
            )

        base = HistGradientBoostingClassifier(
            random_state=random_state,
        )

        # Небольшая, но разумная сетка гиперпараметров
        param_dist = {
            "max_depth": [3, 5, 7],
            "learning_rate": [3e-4, 0.03, 0.05, 0.08, 0.1],
            "max_iter": [150, 250, 350],
            "min_samples_leaf": [10, 20, 50],
            "l2_regularization": [0.0, 0.25, 0.5, 1.0],
        }

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=3,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(X_tr, y_tr)

        best_model = search.best_estimator_
        from sklearn.metrics import roc_auc_score

        y_pred = best_model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, y_pred)

        print(f"  [arm {arm}] best AUC on val: {auc:.4f}")
        print(f"  [arm {arm}] best params: {search.best_params_}")

        models.append(best_model)

    print("=" * 60)
    print("[HistGB Reward] tuning finished.")
    return models


def gbm_q_matrix(models, X):
    """
    models: список из HGB-моделей (по одной на arm)
    X     : [n, d]
    -> q(x,a): [n, K]
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    K = len(models)

    q = np.zeros((n, K), dtype=np.float32)
    for k, clf in enumerate(models):
        # predict_proba[:,1] = P(r=1)
        q[:, k] = clf.predict_proba(X)[:, 1]

    return q


# ===========================
#  PolicyNet + SNIPS
# ===========================

class PolicyNet(nn.Module):
    def __init__(self, in_dim, n_actions=3, hidden=128, dropout=0.1, mu=1/3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )
        self.mu = mu
        self.n_actions = n_actions

    def forward(self, x):
        return self.net(x)  # логиты [B,3]

    def snips_objective(
            self,
            logits,
            a,
            r,
            clip=10.0,
            entropy_coef=1e-3,
            temp=1.0,
    ):
        # πθ(a|x)
        probs = F.softmax(logits / temp, dim=1)
        pi_a = probs[torch.arange(probs.size(0)), a]
        w = (pi_a / self.mu).clamp(max=clip)
        snips = torch.sum(w * r) / (torch.sum(w) + 1e-8)

        entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=1))
        loss = -(snips + entropy_coef * entropy)
        return loss, snips.detach(), entropy.detach()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32)
            logits = self(xb)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        return probs


def _compute_snips(probs, actions, rewards, mu):
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)

    pi_a = probs[np.arange(len(actions)), actions]
    w = pi_a / mu
    num = np.sum(w * rewards)
    den = np.sum(w) + 1e-8
    return num / den


def _iterate_minibatches_idx(n, batch_size):
    idx = np.random.permutation(n)
    for i in range(0, n, batch_size):
        yield idx[i : i + batch_size]


def train_policy_with_teacher(
        X_train,
        X_val,
        a_train,
        a_val,
        r_train,
        r_val,
        teacher_probs_train,
        teacher_probs_val,
        mu,
        device="cpu",
        n_actions=3,
        hidden=128,
        dropout=0.1,
        lr=3e-3,
        weight_decay=1e-4,
        epochs=50,
        batch_size=4096,
        patience=7,
        lambda_align=0.2,
):
    """
    PolicyNet: SNIPS + KL(teacher || policy)
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    a_train = np.asarray(a_train, dtype=np.int64)
    a_val = np.asarray(a_val, dtype=np.int64)
    r_train = np.asarray(r_train, dtype=np.float32)
    r_val = np.asarray(r_val, dtype=np.float32)
    teacher_probs_train = np.asarray(teacher_probs_train, dtype=np.float32)
    teacher_probs_val = np.asarray(teacher_probs_val, dtype=np.float32)

    in_dim = X_train.shape[1]
    model = PolicyNet(in_dim, n_actions=n_actions,
                      hidden=hidden, dropout=dropout, mu=mu).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # best_static по валу
    v_per_arm, v_best_static = evaluate_best_static(
        actions=a_val,
        rewards=r_val,
        mu=mu,
        n_actions=n_actions,
    )

    best_val_snips = -1e9
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    stale = 0

    n_train = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        tr_snips = tr_kl = 0.0
        n_batches = 0

        for batch_idx in _iterate_minibatches_idx(n_train, batch_size):
            xb = X_train[batch_idx]
            ab = a_train[batch_idx]
            rb = r_train[batch_idx]
            tb = teacher_probs_train[batch_idx]

            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            ab_t = torch.tensor(ab, dtype=torch.long, device=device)
            rb_t = torch.tensor(rb, dtype=torch.float32, device=device)
            tb_t = torch.tensor(tb, dtype=torch.float32, device=device)

            opt.zero_grad()
            logits = model(xb_t)
            loss_snips, snips_val, _ = model.snips_objective(
                logits, ab_t, rb_t, clip=10.0, entropy_coef=2e-3, temp=1.0
            )

            # KL(teacher || policy)
            probs = F.softmax(logits, dim=1)
            kl = torch.mean(
                torch.sum(tb_t * (torch.log(tb_t + 1e-12) - torch.log(probs + 1e-12)),
                          dim=1)
            )

            loss = loss_snips + lambda_align * kl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tr_snips += snips_val.item()
            tr_kl += kl.item()
            n_batches += 1

        tr_snips /= max(n_batches, 1)
        tr_kl /= max(n_batches, 1)

        # ---- валидация ----
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(X_val, dtype=torch.float32, device=device)
            ab = torch.tensor(a_val, dtype=torch.long, device=device)
            rb = torch.tensor(r_val, dtype=torch.float32, device=device)
            logits = model(xb)
            _, va_snips_t, _ = model.snips_objective(
                logits, ab, rb, clip=10.0, entropy_coef=0.0, temp=1.0
            )

        va_snips = va_snips_t.item()
        va_score = va_snips - v_best_static

        print(
            f"[Policy] epoch {epoch:02d} | "
            f"train_SNIPS={tr_snips:.4f} | "
            f"train_KL={tr_kl:.4f} | "
            f"val_SNIPS={va_snips:.4f} | "
            f"val_score={va_score:.6f}"
        )

        if va_snips > best_val_snips + 1e-6:
            best_val_snips = va_snips
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()

    return model, v_per_arm, v_best_static


# ===========================
#  Главный раннер
# ===========================

def run(
        X_train: np.ndarray,
        X_val: np.ndarray,
        a_train: np.ndarray,
        a_val: np.ndarray,
        r_train: np.ndarray,
        r_val: np.ndarray,
        X_test: np.ndarray,
        MU: float,
        DEVICE: str,
):
    n_actions = 3

    # 1) GBM-учитель
    print("==== Train HistGB teacher ====")
    gbm_models = tune_histgb_reward_models(
        X_train,
        a_train,
        r_train,
        n_actions=n_actions,
        random_state=42,
        n_iter=20,
    )

    # q(x,a) на train/val/test
    q_train = gbm_q_matrix(gbm_models, X_train)
    q_val   = gbm_q_matrix(gbm_models, X_val)
    q_test  = gbm_q_matrix(gbm_models, X_test)

    # teacher policy π_q (softmax по q / τ_q)
    # при желании можно подбором искать τ_q по сетке
    tau_q = 0.00002

    def _softmax_q(q):
        z = q / tau_q
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

    teacher_train = _softmax_q(q_train)
    teacher_val   = _softmax_q(q_val)

    # 2) PolicyNet с SNIPS + KL к учителю
    print("==== Train PolicyNet with teacher & SNIPS ====")
    policy, v_per_arm, v_best_static = train_policy_with_teacher(
        X_train=X_train,
        X_val=X_val,
        a_train=a_train,
        a_val=a_val,
        r_train=r_train,
        r_val=r_val,
        teacher_probs_train=teacher_train,
        teacher_probs_val=teacher_val,
        mu=MU,
        device=DEVICE,
        n_actions=n_actions,
        hidden=128,
        dropout=0.1,
        lr=3e-3,
        weight_decay=1e-4,
        epochs=50,
        batch_size=4096,
        patience=7,
        lambda_align=0.00002,
    )

    # 3) SNIPS для чистого PolicyNet + лучшая статическая
    probs_val = policy.predict(X_val)

    v_snips_base = _compute_snips(probs_val, a_val, r_val, MU)
    base_score = v_snips_base - v_best_static
    best_arm = int(np.argmax(v_per_arm))

    print("======================================================")
    print(f"[PolicyNet only] V_SNIPS(π)    = {v_snips_base:.6f}")
    print(f"[PolicyNet only] V_best_static = {v_best_static:.6f}")
    print(f"[PolicyNet only] score         = {base_score:.6f}")
    print(f"Best static arm on val = {best_arm}, "
          f"V_static={v_per_arm[best_arm]:.6f}")
    print("======================================================")

    # 4) Микс со статиком
    alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    static_val = np.zeros_like(probs_val)
    static_val[:, best_arm] = 1.0

    best_alpha = 0.0
    best_mix_score = base_score
    best_mix_snips = v_snips_base

    for alpha in alphas:
        mix_val = (1.0 - alpha) * static_val + alpha * probs_val
        mix_val = mix_val / np.clip(mix_val.sum(axis=1, keepdims=True), 1e-12, None)

        v_mix = _compute_snips(mix_val, a_val, r_val, MU)
        score_mix = v_mix - v_best_static

        print(
            f"[Mix alpha={alpha:.2f}] "
            f"V_SNIPS={v_mix:.6f} | score={score_mix:.6f}"
        )

        if score_mix > best_mix_score + 1e-6:
            best_mix_score = score_mix
            best_mix_snips = v_mix
            best_alpha = alpha

    print("======================================================")
    print(f"[Best mixture] alpha   = {best_alpha:.2f}")
    print(f"[Best mixture] V_SNIPS = {best_mix_snips:.6f}")
    print(f"[Best mixture] score   = {best_mix_score:.6f}")
    print("======================================================")

    # 5) Предсказания на test
    probs_test = policy.predict(X_test)

    if best_alpha > 0.0:
        static_test = np.zeros_like(probs_test)
        static_test[:, best_arm] = 1.0
        probs_test = (1.0 - best_alpha) * static_test + best_alpha * probs_test
        probs_test = probs_test / np.clip(
            probs_test.sum(axis=1, keepdims=True), 1e-12, None
        )

    return probs_test
