"""How close each car ran to the car ahead on every lap.

Saves the median headway and the share of the lap within 1, 2 and 3 s of the car ahead.
Uses the on-track reconstruction, so lapped cars count. It takes a minute or two, so the
result is saved to data/traffic.parquet for the other scripts to read.

Run from the project root:
    uv run python src/build_traffic.py
"""
import fastf1
import numpy as np
import pandas as pd

import track

OUT = "data/traffic.parquet"
NEAR_S = (1.0, 2.0, 3.0)  # shares are saved for each of these headways (seconds)
LAST_DRY_LAP = 50         # the reference loop comes from the fastest dry lap up to here
GRID_S = 0.5              # time step of the common grid


def near_share_table(laps: pd.DataFrame, drivers: list[str], t_grid, gap) -> pd.DataFrame:
    """Median headway and the share of each lap within each NEAR_S of the car ahead.

    `gap` is the (n_cars, n_times) headway from `track.on_track_gaps`, with the cars in
    the same order as `drivers`. Laps with no start or end time are skipped.
    """
    rows = []
    for i, drv in enumerate(drivers):
        d = laps[(laps["Driver"] == drv) & laps["LapStartTime"].notna() & laps["Time"].notna()]
        for r in d.itertuples():
            start, end = r.LapStartTime.total_seconds(), r.Time.total_seconds()
            k0, k1 = np.searchsorted(t_grid, [start, end])
            if k1 > k0:
                g = gap[i, k0:k1]
                known = np.isfinite(g)
                row = {
                    "driver": drv,
                    "lap": int(r.LapNumber),
                    "gap_median_s": float(np.median(g[known])) if known.any() else np.nan,
                }
                for thr in NEAR_S:
                    row[f"near_{thr:g}s"] = float(np.mean(known & (g < thr)))
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    fastf1.Cache.enable_cache("data/fastf1_cache")
    s = fastf1.get_session(2023, "Monaco", "R")
    s.load(laps=True, telemetry=True, weather=False, messages=False)
    laps = s.laps

    # Reference loop from the fastest dry & green-flag lap that is not a pit lap.
    ok = laps[
        (laps.LapNumber <= LAST_DRY_LAP)
        & laps.PitInTime.isna()
        & laps.PitOutTime.isna()
        & laps.TrackStatus.eq("1")
        & laps.LapTime.notna()
    ]
    best = ok.loc[ok.LapTime.idxmin()]
    pos = s.pos_data[best.DriverNumber]
    sec = pos.SessionTime.dt.total_seconds()
    t0, t1 = best.LapStartTime.total_seconds(), best.Time.total_seconds()
    pw = (sec >= t0 - 2) & (sec <= t1 + 2)
    ref = track.reference_from_lap(
        sec[pw].to_numpy(), pos.loc[pw, ["X", "Y"]].to_numpy(float), t0, t1
    )

    def progress_of(drv):
        """Race progress of one car, with the pit lane bridged from its own speed."""
        d = laps[laps.Driver == drv]
        num = d.DriverNumber.iloc[0]
        pv = s.pos_data[num]
        t = pv.SessionTime.dt.total_seconds().to_numpy()
        xy = pv[["X", "Y"]].to_numpy(float)
        w = (t >= d.LapStartTime.min().total_seconds()) & (t <= d.Time.max().total_seconds())
        ends = d.dropna(subset=["Time"])
        tw, prog, _, _ = track.reconstruct(
            ref, t[w], xy[w], ends.Time.dt.total_seconds().to_numpy(), ends.LapNumber.to_numpy()
        )
        wins = track.pit_windows(
            d.PitInTime.dt.total_seconds().to_numpy(), d.PitOutTime.dt.total_seconds().to_numpy()
        )
        cd = s.car_data[num]
        tc = cd.SessionTime.dt.total_seconds().to_numpy()
        v = cd.Speed.to_numpy(float) / 3.6
        return tw, track.speed_bridge_progress(tw, prog, wins, tc, v), wins

    drivers = list(laps.Driver.unique())
    cars = {drv: progress_of(drv) for drv in drivers}
    t_all = np.concatenate([cars[d][0] for d in drivers])
    t_grid = np.arange(t_all.min(), t_all.max(), GRID_S)
    progress = np.full((len(drivers), len(t_grid)), np.nan)
    in_pit = np.zeros(progress.shape, bool)
    for i, drv in enumerate(drivers):
        tw, sp, wins = cars[drv]
        progress[i] = track.to_grid(tw, sp, t_grid)
        in_pit[i] = track.in_pit_mask(t_grid, wins)
    _, _, gap = track.on_track_gaps(progress, in_pit, t_grid)

    table = near_share_table(laps, drivers, t_grid, gap)
    table.to_parquet(OUT, index=False)
    print(f"Saved {len(table)} driver-laps to {OUT}")
    cars_shown = ["ALO", "HAM", "OCO", "RUS", "GAS"]
    shown = table[table.driver.isin(cars_shown) & table.lap.between(55, 61)]
    print("\n>>> Share of each lap within 2 s of the car ahead <<<")
    print(shown.pivot(index="lap", columns="driver", values="near_2s").round(2).to_string())


if __name__ == "__main__":
    main()