"""Excluded annual, cost and reason fields."""

from __future__ import annotations

import numpy as np

ANNUAL = frozenset(("QALY", "DALY", "QOLS"))


def cost_free_columns(catalog):

    def depends_on_cost(c):
        name = c["name"]
        return name.startswith("v10|other|") or (
            name.startswith("v15|") and name.endswith("|mean_base_cost")
        )

    return np.array([i for i, c in enumerate(catalog) if not depends_on_cost(c)], dtype=np.int64)


def annual_row(table, row):
    return table == "observations" and (
        str(row.get("CODE", "")).strip().upper() in ANNUAL
        or str(row.get("DESCRIPTION", "")).strip().upper() in ANNUAL
    )


def quarantine_reason_fields(row):
    """Forget source reasons completely; never infer validity from future labels."""
    return {k: "" if k in ("REASONCODE", "REASONDESCRIPTION") else v for k, v in row.items()}
