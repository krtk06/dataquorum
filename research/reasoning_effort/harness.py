"""Runs inside the sandbox container. Executes the generated clean() on a fixed messy frame
and checks the result against the known correct answer. Prints one JSON line."""

import json
import math
from typing import Any

import pandas as pd

AGE = [34, None, 51, 29, None, 42, 38, 61, None, 25, 47, 33]
INCOME = ["$1,200", "$3.4k", "n/a", " $2,050 ", "980", "$4,100",
          "1,750.50", None, "N/A", "$1,100", "", "$2,250"]  # fmt: skip
CITY = ["new york", "NEW YORK", "Boston", "boston ", " Chicago", "chicago",
        "Seattle", "seattle", "san  francisco", "Boston", "Chicago", "new york"]  # fmt: skip

NAN = math.nan
EXPECTED_INCOME = [1200, 3400, NAN, 2050, 980, 4100, 1750.5, NAN, NAN, 1100, NAN, 2250]
EXPECTED_CITY = [
    "New York", "New York", "Boston", "Boston", "Chicago", "Chicago",
    "Seattle", "Seattle", "San Francisco", "Boston", "Chicago", "New York",
]  # fmt: skip


def same_floats(actual: list[Any], expected: list[Any]) -> bool:
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected, strict=True):
        if pd.isna(e):
            if not pd.isna(a):
                return False
        elif pd.isna(a) or abs(float(a) - e) > 1e-6:
            return False
    return True


def main() -> None:
    checks = {"ran": False, "is_dataframe": False, "rows_kept": False,
              "income_correct": False, "city_correct": False, "age_untouched": False}  # fmt: skip
    error = None
    try:
        namespace: dict[str, Any] = {}
        with open("/io/in/code.py", encoding="utf-8") as f:
            exec(f.read(), namespace)
        df = pd.DataFrame({"age": AGE, "income": INCOME, "city": CITY})
        out = namespace["clean"](df.copy())
        checks["ran"] = True
        checks["is_dataframe"] = isinstance(out, pd.DataFrame)
        if checks["is_dataframe"]:
            checks["rows_kept"] = len(out) == 12
            out = out.reset_index(drop=True)
            if checks["rows_kept"] and {"age", "income", "city"} <= set(out.columns):
                checks["income_correct"] = pd.api.types.is_float_dtype(
                    out["income"]
                ) and same_floats(out["income"].tolist(), EXPECTED_INCOME)
                checks["city_correct"] = out["city"].tolist() == EXPECTED_CITY
                checks["age_untouched"] = same_floats(
                    out["age"].tolist(), [NAN if a is None else a for a in AGE]
                )
    except Exception as exc:  # report any failure as data
        error = f"{type(exc).__name__}: {exc}"[:300]
    print(json.dumps({"checks": checks, "passed": all(checks.values()), "error": error}))


main()
