"""Calendar-year windows and first-recorded-diagnosis labels."""

from __future__ import annotations

import numpy as np


def shift_years(day, years):
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)


def labels(first_dates, ids, codes, anchors):
    y = np.zeros((len(ids), len(codes)), dtype=np.float32)
    prior = np.zeros_like(y)
    for i, pid in enumerate(ids):
        start, end = anchors[pid]
        for j, code in enumerate(codes):
            day = first_dates.get((pid, code))
            if day is not None:
                prior[i, j] = day < start
                y[i, j] = start <= day <= end
    return (y, prior)
