"""Pull the 2023 Monaco GP from FastF1 and write one clean driver-lap table.

Requires: fastf1, pyarrow.
Run from the project root (paths are relative to it):
    uv run python src/load.py
"""
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

YEAR, EVENT, SESSION = 2023, "Monaco", "R"
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "fastf1_cache"
OUT_PATH = DATA_DIR / "monaco_2023_laps.parquet"


def load_session():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = fastf1.get_session(YEAR, EVENT, SESSION)
    session.load(laps=True, weather=True, telemetry=False, messages=False)
    return session


def build_lap_table(session) -> pd.DataFrame:
    laps = session.laps.copy()
    secs = lambda s: s.dt.total_seconds()

    df = pd.DataFrame({
        "driver": laps["Driver"],
        "team": laps["Team"],
        "lap": laps["LapNumber"].astype(int),
        "position": laps["Position"],
        "lap_time_s": secs(laps["LapTime"]),
        "end_time_s": secs(laps["Time"]),
        "compound": laps["Compound"],
        "tyre_life": laps["TyreLife"],
        "stint": laps["Stint"],
        "is_in_lap": laps["PitInTime"].notna(),
        "is_out_lap": laps["PitOutTime"].notna(),
        "track_status": laps["TrackStatus"],
    })

    # Attach the most recent weather reading at the end of each lap.
    w = session.weather_data
    w_t = secs(w["Time"]).to_numpy()
    idx = np.searchsorted(w_t, df["end_time_s"].fillna(-1).to_numpy(), side="right") - 1
    df["rainfall"] = pd.Series(w["Rainfall"].to_numpy()[idx.clip(0)], index=df.index).astype("boolean")
    df.loc[df["end_time_s"].isna(), "rainfall"] = pd.NA

    return df.sort_values(["driver", "lap"]).reset_index(drop=True)


def add_gaps(df: pd.DataFrame, ref: str = "VER") -> pd.DataFrame:
    """Gap to the race leader and to a reference driver, per lap.

    Only meaningful for cars on the lead lap. Lapped cars will show misleading values
    (see `laps_down`).
    """
    df = df.copy()
    df["gap_to_leader_s"] = df["end_time_s"] - df.groupby("lap")["end_time_s"].transform("min")
    ref_t = df[df["driver"] == ref].set_index("lap")["end_time_s"]
    df["gap_to_ref_s"] = df["end_time_s"] - df["lap"].map(ref_t)

    # laps_down: laps the leader had completed when this car crossed the line, minus the
    # car's own lap number. A count at the line, not an on-track state, so it can flip
    # back to 0 when the leader has a long pit lap (e.g. TSU on laps 53-54).
    leader_end = df.groupby("lap")["end_time_s"].min().sort_index().to_numpy()
    t = df["end_time_s"]
    done = np.searchsorted(leader_end, t.fillna(-1).to_numpy(), side="right")
    df["laps_down"] = (done - df["lap"]).clip(lower=0).astype("Int64")
    df.loc[t.isna(), "laps_down"] = pd.NA

    return df


def sanity_check(df: pd.DataFrame, laps=(50, 58), drivers=("ALO", "VER", "OCO", "HAM", "RUS", "GAS", "TSU")):
    """Print the pit window so to verify who pitted when, and on what."""
    view = df[df["driver"].isin(drivers) & df["lap"].between(*laps)]
    cols = ["lap", "driver", "position", "compound", "tyre_life",
            "is_in_lap", "is_out_lap", "rainfall", "lap_time_s", "gap_to_ref_s", "laps_down"]
    print(view.sort_values(["driver", "lap"])[cols].to_string(index=False))


if __name__ == "__main__":
    session = load_session()
    table = add_gaps(build_lap_table(session), ref="VER")
    DATA_DIR.mkdir(exist_ok=True)
    table.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(table):,} driver-laps to {OUT_PATH}")
    print("\n>>> Sanity check: pit window for ALO, VER, OCO, HAM, RUS, GAS, TSU <<<")
    sanity_check(table)