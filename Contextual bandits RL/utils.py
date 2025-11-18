import os
import numpy as np
import torch
import pandas as pd
def create_submission(predictions, test):
    """
    Cоздание файла submission.csv в папку results
    """

    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'

    submission = pd.DataFrame({
        "id": test["id"].values,
        "p_mens_email": predictions[:, 0],
        "p_womens_email": predictions[:, 1],
        "p_no_email": predictions[:, 2],
    })
    assert np.allclose(submission[["p_mens_email", "p_womens_email", "p_no_email"]].sum(1), 1.0, atol=1e-6)

    submission.to_csv(submission_path, index=False)

    print(f"Submission файл сохранен: {submission_path}")

    return submission_path


def initialize():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    MU = 1.0 / 3.0

    map_segment = {
        "Mens E-Mail": "mens_email",
        "Womens E-Mail": "womens_email",
        "No E-Mail": "no_email",
    }
    train_data["segment_std"] = train_data["segment"].map(map_segment)

    return DEVICE, RANDOM_STATE, MU, train_data, test_data


def evaluate_policy_snips(policy, X, actions, rewards, mu=1.0/3.0):
    """
    policy  : объект PolicyNet с методом predict(X) -> [n,3] (π(a|x))
    X       : np.ndarray, [n, d]   – признаки
    actions : np.ndarray, [n]      – фактические действия 0/1/2 (segment_std)
    rewards : np.ndarray, [n]      – r_i (0/1)
    mu      : float или массив     – пропенсити логирующей политики (у тебя 1/3)
    """
    # π(a|x) от нашей политики
    probs = policy.predict(X)          # shape: [n,3]
    pi_a = probs[np.arange(len(actions)), actions]  # πθ(a_i|x_i)

    # важностные веса w_i = πθ(a_i|x_i) / μ_i
    w = pi_a / mu

    num = np.sum(w * rewards)
    den = np.sum(w) + 1e-8
    v_snips = num / den
    return v_snips


def evaluate_best_static(actions, rewards, mu=1.0/3.0, n_actions=3):
    """
    Возвращает:
      v_static_per_arm : np.ndarray [n_actions] – V^static(a) для каждого arm
      v_best_static    : float                  – max_a V^static(a)
    """
    v_static = np.zeros(n_actions, dtype=float)
    for a in range(n_actions):
        mask = (actions == a)
        if not np.any(mask):
            v_static[a] = 0.0
            continue

        r_a = rewards[mask].astype(float)
        # IPS-формула (на случай, если μ не константа)
        num = np.sum(r_a / mu)
        den = np.sum(np.ones_like(r_a) / mu)
        v_static[a] = num / (den + 1e-8)

    v_best = v_static.max()
    return v_static, v_best

def score(policy, X, actions, rewards, mu=1.0/3.0, n_actions=3):
    v_snips = evaluate_policy_snips(policy, X, actions, rewards, mu=mu)
    _, v_best_static = evaluate_best_static(actions, rewards, mu=mu, n_actions=n_actions)
    score = v_snips - v_best_static
    return score, v_snips, v_best_static