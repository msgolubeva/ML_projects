"""
Основной файл с решением соревнования
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import create_submission, initialize, evaluate_best_static
from model_1_policynet import run as run_policynet
from model_2_dm import run_dm
from model_3_dr import run_dr
from model_4_snips_mix import run as run_snips_mix



# -------------------------------------------------
#  1. Feature engineering
# -------------------------------------------------

SEG_MID_MAP = {
    "1) $0 - $100":      50.0,
    "2) $100 - $200":    150.0,
    "3) $200 - $350":    275.0,
    "4) $350 - $500":    425.0,
    "5) $500 - $750":    625.0,
    "6) $750 - $1,000":  875.0,
    "7) $1,000 +":       1200.0,  # условный high
}


def _add_fe(df):
    """
    Лёгкий feature engineering:
      - логарифм history
      - биннинг recency
      - выделение структуры из history_segment:
          * hist_seg_idx: номер сегмента (1..7)
          * hist_seg_mid: «середина» денежного интервала
    """
    df = df.copy()

    # логарифм истории (смягчаем хвосты)
    df["history_log1p"] = np.log1p(df["history"].clip(lower=0))

    # биннинг recency в фиксированные интервалы
    bins = [0, 1, 3, 6, 12, 24, 10_000]
    df["recency_bin"] = pd.cut(
        df["recency"],
        bins=bins,
        labels=False,
        include_lowest=True,
    ).astype("float").fillna(0).astype(int)

    # --------- новая логика для history_segment ---------

    # 1) индекс сегмента из строки "1) $0 - $100" -> 1
    seg_idx = (
        df["history_segment"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )
    df["hist_seg_idx"] = seg_idx.fillna(0).astype(int)

    # 2) mid-value интервала по маппингу
    seg_mid = df["history_segment"].map(SEG_MID_MAP)
    # если вдруг встретится незнакомая категория — подстрахуемся history
    df["hist_seg_mid"] = seg_mid.fillna(df["history"]).astype(float)

    return df



def _build_preprocessor(train_data):
    """
    Создаёт sklearn-пайплайн предобработки под DR/Policy-модель.
    """
    # признаки после FE
    train_data = _add_fe(train_data)

    # числовые признаки
    num_cols = [
        "recency",
        "history",
        "history_log1p",
        "hist_seg_idx",   # новый ordinal-признак
        "hist_seg_mid",   # новый "денежный" признак
    ]

    # бинарные как есть
    bin_cols = ["mens", "womens", "newbie"]

    # категориальные (one-hot)
    # оставляем history_segment и recency_bin как категориальные
    cat_cols = ["zip_code", "channel", "history_segment", "recency_bin"]

    feature_cols = num_cols + bin_cols + cat_cols

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("bin", "passthrough", bin_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    pipe = Pipeline([("prep", preprocess)])
    pipe.feature_cols_ = feature_cols  # сохраним для удобства

    return pipe


def process_data(train_data, test_data):
    """
    Готовит данные:
      - делает feature engineering
      - строит и обучает препроцессор
      - кодирует действия в 0/1/2
      - возвращает X_train, a_train, r_train, X_test
    """
    # feature engineering
    train_fe = _add_fe(train_data)
    test_fe  = _add_fe(test_data)

    # препроцессор (знает, какие колонки использовать)
    pipe = _build_preprocessor(train_data)

    X_all_sparse = pipe.fit_transform(train_fe[pipe.feature_cols_])
    X_test_sparse = pipe.transform(test_fe[pipe.feature_cols_])

    # преобразуем в плотный numpy (для torch и т.п.)
    X_data_train = (
        X_all_sparse.toarray()
        if hasattr(X_all_sparse, "toarray")
        else np.asarray(X_all_sparse)
    )
    X_data_test = (
        X_test_sparse.toarray()
        if hasattr(X_test_sparse, "toarray")
        else np.asarray(X_test_sparse)
    )

    # действия: segment -> {0,1,2}
    action_map = {
        "Mens E-Mail":   0,
        "Womens E-Mail": 1,
        "No E-Mail":     2,
    }
    a_data_train = train_data["segment"].map(action_map).astype(int).to_numpy()
    r_data_train = train_data["visit"].astype(int).to_numpy().astype(np.float32)

    return X_data_train, a_data_train, r_data_train, X_data_test


    # num_cols = ["recency", "history"]
    # bin_cols = ["mens", "womens", "newbie"]
    # cat_cols = ["zip_code", "channel", "history_segment"]
    # feature_cols = num_cols + bin_cols + cat_cols
    #
    # preprocess = ColumnTransformer(
    #     transformers=[
    #         ("num", StandardScaler(), num_cols),
    #         ("bin", "passthrough", bin_cols),
    #         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    #     ],
    #     remainder="drop",
    # )
    #
    # pipe = Pipeline([("prep", preprocess)])
    # X_train = pipe.fit_transform(train_data[feature_cols])
    # X_test = pipe.transform(test_data[feature_cols])
    #
    # X_data_train = X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)
    # X_data_test = X_test.toarray() if hasattr(X_test, "toarray") else np.asarray(X_test)
    #
    # # действия в {0,1,2}
    # action_to_idx = {"mens_email": 0, "womens_email": 1, "no_email": 2}
    # a_data_train = train_data["segment_std"].map(action_to_idx).astype(int).to_numpy()
    #
    # r_data_train = train_data["visit"].astype(int).to_numpy().astype(np.float32)
    #
    # return X_data_train, a_data_train, r_data_train, X_data_test

def main():
    """
    Главная функция программы
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)

    # Инициализируем базовые переменные
    DEVICE, RANDOM_STATE, MU, train_data, test_data = initialize()

    # Обрабатываем train и test
    X_data_train, a_data_train, r_data_train, X_data_test = process_data(train_data, test_data)
    # Делим обучаемую выборку на train и validation
    X_train, X_val, a_train, a_val, r_train, r_val = train_test_split(
        X_data_train, a_data_train, r_data_train,
        test_size=0.2, random_state=RANDOM_STATE, stratify=a_data_train
    )

    # Создание модели

    # predictions = run_policynet(
    #     X_train, X_val,
    #     a_train, a_val,
    #     r_train, r_val,
    #     X_data_test,
    #     MU, DEVICE,
    # )

    # predictions = run_dm(X_train, a_train, r_train,
    #                      X_val, a_val, r_val,
    #                      X_data_test,
    #                      DEVICE, MU)

    # predictions = run_dr(X_train, X_val,
    #                      a_train, a_val,
    #                      r_train, r_val,
    #                      X_data_test, DEVICE)

    predictions = run_snips_mix(
        X_train, X_val,
        a_train, a_val,
        r_train, r_val,
        X_data_test,
        MU, DEVICE,
    )

    # Создание сабмита
    create_submission(predictions, test_data)

    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
