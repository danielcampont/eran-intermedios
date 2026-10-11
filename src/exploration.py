"""Exploration: when the track turned wet, dry pit loss, and donor comparability.

Reads data/monaco_2023_laps.parquet (written by load.py).
Run from the project root:
    uv run python src/exploration.py
"""
import pandas as pd

DATA = "data/monaco_2023_laps.parquet"
DECISION_LAP = 54
ALO = "ALO"
# All donors pitted for inters at the end of lap 54.
# RUS lap 55 is excluded from the model (off at Turn 5, contact with PER).
DONORS = ["HAM", "OCO", "RUS", "GAS"]
REF = "VER"
DRY_REF_LAPS = (40, 47)  # field median flat at about 78 s over these laps (per-lap median of clean laps)
WET_MARGIN_S = 1.0       # field median this far above its dry level implies a wet lap
PACE_LAPS = (46, 53)     # window shown in the pace table; the model's pace offset uses only 46-50


# When did the track really stop being dry?
def first_wet_lap(df: pd.DataFrame) -> int:
    """First lap where the field's median clean lap time rises well above its dry level.

    Uses pace instead of the rain flag, which lags the real conditions by a few laps.
    """
    clean = df[~df["is_in_lap"] & ~df["is_out_lap"] & df["track_status"].eq("1")]
    field = clean.groupby("lap")["lap_time_s"].median()
    ref = field.loc[DRY_REF_LAPS[0]:DRY_REF_LAPS[1]].median()
    later = field.loc[DRY_REF_LAPS[1] + 1:]
    hits = later.index[(later > ref + WET_MARGIN_S).to_numpy()]
    return int(hits.min()) if len(hits) else int(df["lap"].max()) + 1


# Extra seconds a pit stop costs compared to the driver's own baseline.
def dry_pit_losses(df: pd.DataFrame, first_wet: int, max_in_lap: int | None = None) -> pd.DataFrame:
    """Excess time on the in lap and out lap versus the driver's own recent clean laps.

    A lap counts as dry only if it is under green flag, the rain flag is off, and it
    comes before `first_wet`. A stop is kept only if its in lap, out lap, and baseline
    laps are all dry.
    `max_in_lap` optionally drops later stops, for the sensitivity check.
    """
    df = df.copy()
    df["dry"] = (
        df["track_status"].eq("1")
        & ~df["rainfall"].fillna(True).astype(bool)
        & (df["lap"] < first_wet)
    )

    rows = []
    for drv, g in df.groupby("driver"):
        g = g.set_index("lap")
        in_laps = g.index[(g["is_in_lap"] & ~g["is_out_lap"]).to_numpy()]  # skips laps that are both in and out laps
        for lap in in_laps:
            if max_in_lap is not None and lap > max_in_lap:
                continue
            nxt = lap + 1
            if nxt not in g.index or g.loc[nxt, "is_in_lap"]:
                continue
            if not (g.loc[lap, "dry"] and g.loc[nxt, "dry"]):
                continue
            b = g.loc[lap - 4: lap - 1]
            b = b[b["dry"] & ~b["is_in_lap"] & ~b["is_out_lap"]]
            if len(b) < 2:
                continue
            base = b["lap_time_s"].median()
            in_x = g.loc[lap, "lap_time_s"] - base
            out_x = g.loc[nxt, "lap_time_s"] - base  # fresh tyres make this slightly optimistic
            if pd.isna(in_x) or pd.isna(out_x):
                continue
            rows.append({
                "driver": drv,
                "lap": lap,
                "base_s": base,
                "in_excess": in_x,
                "out_excess": out_x,
                "total": in_x + out_x
            })
    return pd.DataFrame(rows)


# Pre-decision pace relative to Verstappen.
# Laps from `first_wet_lap` onwards (51 in this race) are wet, so they show who reacted
# to the rain and when. The pace offset used by the model only uses the dry laps (46-50).
def pace_vs_ref(df: pd.DataFrame, laps=PACE_LAPS) -> pd.DataFrame:
    w = df[df["lap"].between(*laps) & ~df["is_in_lap"] & ~df["is_out_lap"]]
    p = w.pivot(index="lap", columns="driver", values="lap_time_s")
    cols = [ALO, *DONORS]
    return p[cols].sub(p[REF], axis=0)


# State entering the decision.
def decision_context(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["lap"] == DECISION_LAP - 1].set_index("driver")
    return d.loc[[ALO, *DONORS], ["position", "compound", "tyre_life", "gap_to_ref_s"]]


if __name__ == "__main__":
    df = pd.read_parquet(DATA)
    print(f"Loaded {len(df):,} driver-laps from {DATA}")

    first_wet = first_wet_lap(df)
    print(f"\nFirst wet lap (field-pace rule): {first_wet}")

    pl = dry_pit_losses(df, first_wet)
    print("\n>>> Dry pit stops: excess seconds vs own baseline <<<")
    print(pl.round(2).sort_values(["lap", "driver"]).to_string(index=False))
    print("\n>>> Summary: dry pit stops <<<")
    print(pl[["in_excess", "out_excess", "total"]].agg(["count", "median", "mean", "std"]).round(2))

    print("\n>>> Sensitivity: median total pit loss by last allowed in lap <<<")
    for cutoff in (50, 48, 45, 40):  # 50 equals no cutoff, since lap 51 is already excluded
        s = dry_pit_losses(df, first_wet, max_in_lap=cutoff)
        med = f"{s['total'].median():.2f}" if len(s) else "n/a"
        print(f"in lap ≤ {cutoff}: n = {len(s)}, median total = {med}")

    print(f"\n>>> Lap time vs VER, laps {PACE_LAPS[0]}-{PACE_LAPS[1]} (in/out laps blank) <<<")
    print(pace_vs_ref(df).round(2).to_string())

    print(f"\n>>> State at the end of lap {DECISION_LAP - 1} (entering the decision) <<<")
    print(decision_context(df).round(2).to_string())