# ---------- Policy model πθ(a|x) ----------
def modelPolicyNet(size_train, MU, DEVICE, lr=3e-3, weight_decay=1e-4):

    class PolicyNet(nn.Module):

        def __init__(self, in_dim, hidden=128, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 3)  # 3 действия -> логиты
            )
            self.mu = MU


        def forward(self, x):
            return self.net(x)  # логиты


        def snips_objective(self, logits, a, r, clip=10.0, entropy_coef=1e-3, temp=1.0):
            """
            logits: [B,3], a: [B] (int64), r: [B] (float)
            SNIPS = sum(w_i r_i) / sum(w_i), w_i = (πθ(a_i|x_i)/μ_i)
            """
            # temperature (по желанию: делает распределение мягче/увереннее)
            probs = torch.softmax(logits / temp, dim=1)  # πθ
            pi_a = probs[torch.arange(probs.size(0)), a]  # πθ(a_i|x_i)
            w = (pi_a / self.mu).clamp_max(clip)  # IPS-клиппинг для стабилизации
            snips = torch.sum(w * r) / (torch.sum(w) + 1e-8)

            # лёгкая энтропийная регуляризация (чтоб не коллапсировало в один arm)
            entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=1))
            # Оптимизируем МАКСИМИЗАЦИЮ SNIPS -> минимизируем -SNIPS - λ*Entropy
            loss = -(snips + entropy_coef * entropy)
            return loss, snips.detach(), entropy.detach()


        # ---------- Инференс на test ----------
        def predict(self, data):
            self.eval()
            with torch.no_grad():
                logits_test = self(torch.tensor(data, dtype=torch.float32))
                probs_test = torch.softmax(logits_test, dim=1).numpy()

            # нормировка
            row_sums = probs_test.sum(axis=1, keepdims=True)
            predictions = probs_test / np.clip(row_sums, 1e-12, None)

            return predictions


    policy = PolicyNet(size_train).to(DEVICE)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    return policy, opt

# ---------- Training ----------
def train(X_train, X_val, a_train, a_val, r_train, r_val, model, opt, DEVICE):
    BATCH = 4096
    EPOCHS = 50
    best_val = -1e9
    patience = 7
    stale = 0
    _, va_best_static = evaluate_best_static(
        actions=a_val,
        rewards=r_val,
        mu=model.mu,
        n_actions=3,
    )

    def iterate_minibatches(X, a, r, batch):
        idx = np.random.permutation(X.shape[0])
        for i in range(0, X.shape[0], batch):
            j = idx[i:i + batch]
            yield X[j], a[j], r[j]

    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = tr_snips = tr_ent = 0.0
        for xb, ab, rb in iterate_minibatches(X_train, a_train, r_train, BATCH):
            xb = torch.tensor(xb, dtype=torch.float32, device=DEVICE)
            ab = torch.tensor(ab, dtype=torch.long, device=DEVICE)
            rb = torch.tensor(rb, dtype=torch.float32, device=DEVICE)

            opt.zero_grad()
            logits = model(xb)
            loss, snips_val, ent = model.snips_objective(logits, ab, rb, clip=10.0, entropy_coef=2e-3, temp=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tr_loss += loss.item()
            tr_snips += snips_val.item()
            tr_ent += ent.item()

        # валидация по SNIPS
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
            ab = torch.tensor(a_val, dtype=torch.long, device=DEVICE)
            rb = torch.tensor(r_val, dtype=torch.float32, device=DEVICE)
            logits = model(xb)
            _, va_snips, _ = model.snips_objective(logits, ab, rb, clip=10.0, entropy_coef=0.0)

        score_epoch = va_snips.item() - va_best_static
        print(f"epoch {epoch:02d} | "
              f"train_SNIPS={tr_snips:.4f} | "
              f"val_SNIPS={va_snips.item():.4f} | "
              f"val_score={score_epoch:.6f} | ")

        if va_snips.item() > best_val + 1e-6:
            best_val = va_snips.item()
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

    return model

def run(X_train, X_val, a_train, a_val, r_train, r_val, X_data_test, MU, DEVICE):
    model, opt = modelPolicyNet(X_train.shape[1], MU, DEVICE)

    # Обучение модели
    model_trained = train(X_train, X_val, a_train, a_val, r_train, r_val,
                  model, opt, DEVICE)

    # Предсказание
    predictions = model_trained.predict(X_data_test)
    return predictions
