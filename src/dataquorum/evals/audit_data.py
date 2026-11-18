"""Local fixture with planted risks and separate features for negative controls."""

import numpy as np
import pandas as pd


def audit_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 160
    i = np.arange(n)
    labels = np.where(i % 2, "yes", "no")
    x = rng.normal(size=n)
    duplicate = x.copy()
    duplicate[120] = duplicate[0]
    continuous = rng.normal(size=n)
    day = pd.date_range("2024-01-01", periods=n, freq="D")
    late_train = day.to_numpy(copy=True)
    late_train[[119, 120]] = late_train[[120, 119]]
    return pd.DataFrame(
        {
            "signal": x,
            "label": labels,
            "rare": (i % 20 == 0).astype(int),
            "amount": continuous,
            "proxy": 2 * continuous + rng.normal(0, 0.005, n),
            "code": np.where(labels == "yes", "A", "B"),
            "row_id": i,
            "post_outcome_fee": rng.normal(size=n),
            "household_id": i // 4,
            "site_id": i % 11,
            "day": day,
            "shifted_day": late_train,
            "split": np.where(i < 120, "train", "test"),
            "repeated": duplicate,
        }
    )
