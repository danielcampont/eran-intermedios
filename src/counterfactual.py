"""Counterfactual: Alonso fits intermediates at the end of lap 54 instead of mediums.

The first stop is common to both worlds, so his lap 54 is his real one. From the next lap
on, his lap times are the donors' (cars that went straight to inters on lap 54) shifted
by the pace offset measured on the dry laps. Verstappen's real times are held fixed.

As a bracket, the offset can also be shifted by how much faster Alonso was than the dry
offset says on real inter laps run in clear air (see `wet_shift_pairs`).

Reads data/monaco_2023_laps.parquet (written by load.py) and, for the wet shift,
data/traffic.parquet (written by build_traffic.py).
Run from the project root:
    uv run python src/counterfactual.py [horizon_lap]
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = "data/monaco_2023_laps.parquet"
TRAFFIC = "data/traffic.parquet"
SUBJECT = "ALO"
REF = "VER"
DONORS = ["HAM", "OCO", "RUS", "GAS"]  # all pitted for inters at the end of lap 54
DECISION_LAP = 54                      # last lap the two worlds share
OFFSET_LAPS = (46, 50)                 # dry laps used for the pace offset
CLEAR_LAPS = (57, 61)                  # laps both cars were on inters in reality
NEAR_COLUMN = "near_3s"                # share of a lap within 3 s of the car ahead
MAX_NEAR = 0.15                        # clear air: at most this share of a lap in traffic
# Donor laps left out of the templates, with the reason.
EXCLUDED_LAPS = {("RUS", 55): "off at Turn 5, contact with PER"}


# Last lap that is safe to simulate.
def last_clean_lap(df: pd.DataFrame, start: int) -> int:
    """Last lap, counting on from `start`, before the first lap with no clean lap in the
    field.

    A clean lap is under green flag throughout and is not an in lap or an out lap.
    """
    clean = df[df["track_status"].eq("1") & ~df["is_in_lap"] & ~df["is_out_lap"]]
    with_clean = set(clean["lap"])
    lap = start
    while lap + 1 in with_clean:
        lap += 1
    return lap


# Pace difference between two drivers on dry laps.
def pace_offset(df: pd.DataFrame, subject: str, donor: str) -> tuple[float, int]:
    """Median of (subject minus donor) lap time over the offset laps, and how many laps
    it uses.

    Only laps that are clean for both drivers count. A positive offset means the subject
    was slower than the donor.
    """
    w = df[
        df["lap"].between(*OFFSET_LAPS)
        & df["track_status"].eq("1")
        & ~df["is_in_lap"]
        & ~df["is_out_lap"]
    ]
    p = w.pivot(index="lap", columns="driver", values="lap_time_s")
    diff = (p[subject] - p[donor]).dropna()
    return (float(diff.median()) if len(diff) else float("nan")), len(diff)


# A donor's lap times after the decision.
def donor_laps(df: pd.DataFrame, donor: str, horizon: int) -> pd.Series:
    """Lap times from the lap after the decision to the horizon, NaN for excluded laps."""
    laps = range(DECISION_LAP + 1, horizon + 1)
    d = df[df["driver"] == donor].set_index("lap")["lap_time_s"].reindex(laps)
    for drv, lap in EXCLUDED_LAPS:
        if drv == donor and lap in d.index:
            d.loc[lap] = np.nan
    return d


# What each donor says the subject's laps would be.
def template_laps(
    df: pd.DataFrame, subject: str, donors: list[str], horizon: int, shift: float = 0.0
) -> pd.DataFrame:
    """The subject's lap times after the decision as each donor gives them (lap plus offset).

    `shift` is added to every offset (seconds per lap, negative means the subject is quicker).
    """
    return pd.DataFrame({
        d: donor_laps(df, d, horizon) + pace_offset(df, subject, d)[0] + shift for d in donors
    })


# Turn lap times into a gap to the reference driver.
def gap_to_ref(df: pd.DataFrame, laps: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Gap to the reference driver (positive = behind) at the end of each lap after the
    decision.

    `laps` has one column per scenario, indexed by lap. The gap at the decision lap is
    the real one. Each later lap adds the scenario's lap time and takes off the reference
    driver's real elapsed time. A missing lap makes every later gap in that column NaN.
    """
    ref_end = (
        df[(df["driver"] == REF) & df["lap"].between(DECISION_LAP, horizon)]
        .set_index("lap")["end_time_s"]
    )
    ref_elapsed = (ref_end - ref_end.loc[DECISION_LAP]).loc[DECISION_LAP + 1:]
    start = df[(df["driver"] == SUBJECT) & (df["lap"] == DECISION_LAP)]["gap_to_ref_s"].iloc[0]
    return start + laps.cumsum(skipna=False).sub(ref_elapsed, axis=0)


# The counterfactual itself.
def counterfactual(df: pd.DataFrame, horizon: int, shift: float = 0.0) -> pd.DataFrame:
    """Gap paths to the reference driver: one per donor, plus the per-lap median of the
    donors.

    In a donor's own column, an excluded lap is borrowed from the median of the other
    donors. The median column never uses an excluded lap.
    """
    templates = template_laps(df, SUBJECT, DONORS, horizon, shift)
    paths = templates.apply(
        lambda col: col.fillna(templates.drop(columns=col.name).median(axis=1))
    )
    paths["median"] = templates.median(axis=1)
    return gap_to_ref(df, paths, horizon)


# Actual against counterfactual, side by side.
def summary(df: pd.DataFrame, gaps: pd.DataFrame, horizon: int) -> pd.DataFrame:
    actual = (
        df[(df["driver"] == SUBJECT) & df["lap"].between(DECISION_LAP, horizon)]
        .set_index("lap")["gap_to_ref_s"]
    )
    out = pd.DataFrame({
        "actual": actual,
        "cf_median": gaps["median"],
        "cf_min": gaps[DONORS].min(axis=1),
        "cf_max": gaps[DONORS].max(axis=1),
    })
    out.loc[DECISION_LAP, ["cf_median", "cf_min", "cf_max"]] = actual.loc[DECISION_LAP]
    return out.sort_index()


# How much quicker the subject was in the wet than the dry offset says.
def wet_shift_pairs(
    df: pd.DataFrame,
    traffic: pd.DataFrame,
    laps=CLEAR_LAPS,
    near: str = NEAR_COLUMN,
    max_near: float = MAX_NEAR,
) -> pd.DataFrame:
    """Subject against each donor on real inter laps where both were in clear air.

    Both cars are on inters in reality from lap 57. A lap counts only if neither car was
    in a pit lap and each spent at most `max_near` of it in traffic (`near` is the
    traffic column, the share of the lap within some headway of the car ahead). `shift_s`
    is the lap time difference (subject minus donor) minus the dry pace offset, so a
    negative value means the subject was quicker than the dry laps predict. In reality
    his inters were also one lap younger than the donors', which the counterfactual does
    not give him.
    """
    w = df[df["lap"].between(*laps) & ~df["is_in_lap"] & ~df["is_out_lap"]]
    w = w.merge(traffic, on=["driver", "lap"])
    p = w[w[near] <= max_near].pivot(index="lap", columns="driver", values="lap_time_s")
    rows = []
    for d in DONORS:
        if d not in p or SUBJECT not in p:
            continue
        offset = pace_offset(df, SUBJECT, d)[0]
        rows += [
            {"driver": d, "lap": lap, "shift_s": x - offset}
            for lap, x in (p[SUBJECT] - p[d]).dropna().items()
        ]
    return pd.DataFrame(rows, columns=["driver", "lap", "shift_s"])


# How wrong the method is on cases where the answer is known.
def leave_one_out(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Predict each donor's own laps from the other donors' laps.

    Every donor really did fit inters on the decision lap, so the truth is known. Error
    is predicted minus actual lap time (positive means the prediction is too slow). `sum`
    adds the errors over the laps with data, and `laps` counts them.
    """
    rows = {}
    for subject in DONORS:
        others = [d for d in DONORS if d != subject]
        predicted = template_laps(df, subject, others, horizon).median(axis=1)
        rows[subject] = predicted - donor_laps(df, subject, horizon)
    out = pd.DataFrame(rows).T
    out["sum"] = out.sum(axis=1)
    out["laps"] = out.drop(columns="sum").notna().sum(axis=1)
    return out


if __name__ == "__main__":
    df = pd.read_parquet(DATA)
    horizon = last_clean_lap(df, DECISION_LAP)
    print(f"Horizon: lap {horizon} (lap {horizon + 1} is the first with no clean lap in the field)")

    print(
        f"\n>>> Pace offset: {SUBJECT} minus donor, median over laps "
        f"{OFFSET_LAPS[0]}-{OFFSET_LAPS[1]} <<<"
    )
    offsets = pd.DataFrame(
        {d: pace_offset(df, SUBJECT, d) for d in DONORS}, index=["offset_s", "laps_used"]
    ).T
    offsets["laps_used"] = offsets["laps_used"].astype(int)
    print(offsets.round(2).to_string())

    print("\n>>> Track status of the donors' laps used as templates <<<")
    w = df[df["driver"].isin(DONORS) & df["lap"].between(DECISION_LAP + 1, horizon)]
    print(w.pivot(index="lap", columns="driver", values="track_status").to_string())

    gaps = counterfactual(df, horizon)
    print(f"\n>>> Gap to {REF} (positive = behind): actual vs counterfactual <<<")
    print(summary(df, gaps, horizon).round(2).to_string())
    print(f"\n>>> Counterfactual gap to {REF} by donor <<<")
    print(gaps.round(2).to_string())

    print("\n>>> Leave-one-out: predicted minus actual lap time, by donor <<<")
    print(leave_one_out(df, horizon).round(2).to_string())

    if not Path(TRAFFIC).exists():
        print(f"\n{TRAFFIC} not found: run build_traffic.py to add the wet shift bracket")
    else:
        traffic = pd.read_parquet(TRAFFIC)
        pairs = wet_shift_pairs(df, traffic)
        print(f"\n>>> Wet shift pairs: laps {CLEAR_LAPS[0]}-{CLEAR_LAPS[1]}, both in clear air <<<")
        print(pairs.round(2).to_string(index=False))
        rows = []
        for last in range(CLEAR_LAPS[0] + 3, CLEAR_LAPS[0] + 8):
            p = wet_shift_pairs(df, traffic, (CLEAR_LAPS[0], last))
            rows.append({"last_lap": last, "pairs": len(p), "median_s": p["shift_s"].median()})
        print("\n>>> Wet shift (median) by the last lap used <<<")
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        print("\n>>> Wet shift (median) by traffic headway and clear-air limit <<<")
        rows = []
        for near in [c for c in traffic.columns if c.startswith("near_")]:
            for limit in (0.05, MAX_NEAR, 0.30):
                p = wet_shift_pairs(df, traffic, near=near, max_near=limit)
                rows.append({
                    "headway": near,
                    "limit": limit,
                    "pairs": len(p),
                    "median_s": p["shift_s"].median()
                })
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        shift = float(pairs["shift_s"].median()) if len(pairs) else 0.0
        print(f"\nShift used for the bracket: {shift:.2f} s per lap, from {len(pairs)} pairs")
        gaps_wet = counterfactual(df, horizon, shift)
        print(f"\n>>> Gap to {REF} (positive = behind): dry offsets plus the wet shift <<<")
        print(summary(df, gaps_wet, horizon).round(2).to_string())