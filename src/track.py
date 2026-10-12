"""On-track position reconstruction from telemetry.

Projects each car's X/Y samples onto one reference lap to get its progress round the
circuit, bridges the pit lane (where that projection fails), and measures on-track gaps
between cars, including lapped ones.

Pure numpy/scipy with no FastF1 dependency, so it can be tested on synthetic data.
X/Y are used in whatever units they come in (FastF1 documents tenths of a metre), and
progress is measured in laps, so units only matter for `spacing` and the residuals.

Tests: uv run python src/test_track.py
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Reference:
    """A path resampled at even spacing, with a nearest-neighbour index.

    For a closed loop (`build_reference`) the end point is not repeated and `s` runs
    from 0 up to just under `length`. For an open path (`build_open_reference`) both
    ends are kept and `s` runs from 0 to `length`.
    """
    xy: np.ndarray  # (m, 2) evenly spaced points along the path
    s: np.ndarray   # (m,) arc length from the start, same units as xy
    length: float   # total path length
    tree: cKDTree   # nearest-neighbour index over xy


def build_reference(xy: np.ndarray, spacing: float = 5.0) -> Reference:
    """Closed and evenly spaced reference loop from one lap's X/Y samples, in time order.

    The lap starts and ends at the timing line, so the loop is closed by joining the
    last sample back to the first. `spacing` is the gap between reference points (5 is
    about half a metre if X/Y are in tenths of a metre).
    """
    pts = np.vstack([xy, xy[:1]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(s[-1])
    n = round(length / spacing)
    s_new = np.arange(n) * (length / n)  # excludes the end point, which equals the start
    dense = np.column_stack([np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1])])
    return Reference(dense, s_new, length, cKDTree(dense))


def reference_from_lap(t, xy, t_start: float, t_end: float, spacing: float = 5.0) -> Reference:
    """Reference loop from one lap, interpolated at exactly the lap's start and end times.

    Building straight from the raw samples starts the loop at the first sample after the
    timing line, which put every car about 27 m off on the 2023 Monaco data.
    `t` (seconds, increasing) and `xy` must reach slightly before `t_start` and after
    `t_end`, so the ends can be interpolated.
    """
    t, xy = np.asarray(t, float), np.asarray(xy, float)
    if t[0] > t_start or t[-1] < t_end:
        raise ValueError("samples must cover the whole lap, including both timing instants")
    inner = (t > t_start) & (t < t_end)
    grid = np.concatenate([[t_start], t[inner], [t_end]])
    pts = np.column_stack([np.interp(grid, t, xy[:, 0]), np.interp(grid, t, xy[:, 1])])
    return build_reference(pts, spacing)


def project(ref: Reference, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest reference point for each sample.

    Returns the fraction of the way along the path and the distance to it (the residual).
    The fraction is in [0, 1) for a closed loop (the lap fraction) and in [0, 1],
    clamped at the ends, for an open path.
    Samples must be finite, so NaNs have to be dropped before calling.
    """
    dist, idx = ref.tree.query(xy)
    return ref.s[idx] / ref.length, dist


def unwrap_progress(u: np.ndarray) -> np.ndarray:
    """Turn lap fractions into cumulative progress (in laps) by counting crossings of the line.

    A drop of more than half a lap is a forward crossing (+1 lap). A rise of more than
    half a lap is a small backward wobble across the line (-1 lap). The starting offset
    is arbitrary here, and `calibrate_progress` fixes it.
    """
    du = np.diff(u)
    laps = np.cumsum(np.where(du < -0.5, 1.0, 0.0) - np.where(du > 0.5, 1.0, 0.0))
    return u + np.concatenate([[0.0], laps])


def calibrate_progress(t, progress, lap_end_t, lap_n):
    """Shift progress by a whole number of laps so it equals lap_n at each recorded lap end.

    This absorbs the grid start behind the line and any formation lap in the data.
    Returns (shifted progress, largest error in laps at the recorded lap ends).
    """
    lap_end_t, lap_n = np.asarray(lap_end_t, float), np.asarray(lap_n, float)
    p_end = np.interp(lap_end_t, t, progress)
    offset = np.round(np.median(lap_n - p_end))
    return progress + offset, float(np.abs(p_end + offset - lap_n).max())


def monotonic_violations(progress: np.ndarray, tol: float = 0.003) -> int:
    """Number of steps where progress goes backwards by more than `tol` laps (about 10 m).

    Cars only move forward, so a violation means a projection onto the wrong part of the
    track, such as where two sections run close together (a hairpin, or the pit lane
    beside the main straight).
    """
    return int((np.diff(progress) < -tol).sum())


def reconstruct(ref: Reference, t, xy, lap_end_t, lap_n):
    """Whole-race progress for one car: project, count laps, and calibrate to the recorded lap ends.

    Drops samples with non-finite coordinates.
    Returns (t, progress, residual, lap-end error), with the first three being aligned
    to the kept samples and the last being a single number.
    """
    t, xy = np.asarray(t, float), np.asarray(xy, float)
    ok = np.isfinite(xy).all(axis=1)
    t, xy = t[ok], xy[ok]
    u, resid = project(ref, xy)
    lap_end_t, lap_n = np.asarray(lap_end_t, float), np.asarray(lap_n, float)
    inside = (lap_end_t > t[0]) & (lap_end_t <= t[-1])
    prog, err = calibrate_progress(t, unwrap_progress(u), lap_end_t[inside], lap_n[inside])
    return t, prog, resid, err


def pit_windows(pit_in_t, pit_out_t) -> list[tuple[float, float]]:
    """(t_in, t_out) pairs for one car: each pit entry with the next pit exit after it.

    An entry with no later exit (a car that stopped in the pits to DNF) gets t_out = inf.
    An exit with no earlier entry (a start from the pit lane) is ignored.
    Times are session seconds, and NaNs are dropped.
    """
    ins = np.asarray(pit_in_t, float)
    outs = np.asarray(pit_out_t, float)
    ins, outs = np.sort(ins[np.isfinite(ins)]), np.sort(outs[np.isfinite(outs)])
    windows = []
    for t_in in ins:
        later = outs[outs > t_in]
        windows.append((float(t_in), float(later[0]) if len(later) else np.inf))
    return windows


def in_pit_mask(t, windows) -> np.ndarray:
    """True for samples inside any pit window (both ends included)."""
    t = np.asarray(t, float)
    mask = np.zeros(len(t), dtype=bool)
    for t_in, t_out in windows:
        mask |= (t >= t_in) & (t <= t_out)
    return mask


def bridge_pit_progress(t, progress, windows) -> np.ndarray:
    """Progress with pit lane samples replaced by a straight line between entry and exit.

    This assumes constant speed through the stop, so it is wrong while the car is
    stationary in its box. That error must not be measured against the raw progress:
    inside the pit lane the raw projection onto the main line is unreliable (see
    `speed_bridge_progress` and `lane_bridge_progress`).
    """
    t, progress = np.asarray(t, float), np.asarray(progress, float)
    keep = ~in_pit_mask(t, windows)
    return np.interp(t, t[keep], progress[keep])


def _edge_progress(t_k, p_k, t_edge: float, before: bool) -> float:
    """Progress at a pit entry/exit instant, extrapolated from the two nearest kept samples.

    `t_k`/`p_k` hold only samples outside every pit window, so the nearest ones sit just
    before the entry (`before=True`) or just after the exit (`before=False`).
    """
    if before:
        i = int(np.searchsorted(t_k, t_edge, side="left"))
        tt, pp = t_k[max(i - 2, 0):i], p_k[max(i - 2, 0):i]
        anchor = -1
    else:
        i = int(np.searchsorted(t_k, t_edge, side="right"))
        tt, pp = t_k[i:i + 2], p_k[i:i + 2]
        anchor = 0
    if len(tt) == 0:
        raise ValueError("no samples outside the pit window on that side")
    if len(tt) == 1:
        return float(pp[0])
    slope = (pp[1] - pp[0]) / (tt[1] - tt[0])
    return float(pp[anchor] + slope * (t_edge - tt[anchor]))


def speed_bridge_progress(t, progress, windows, t_speed, speed_ms) -> np.ndarray:
    """Progress with pit lane samples placed using the car's own speed.

    Inside each pit window, the distance travelled is the integral of speed (m/s) from
    pit entry, rescaled so the car arrives at the progress it has at pit exit. Entry and
    exit progress are extrapolated from the neighbouring samples outside the window.
    Unlike `bridge_pit_progress`, it respects the time spent stationary in the box.
    A window with no speed information, or one that never ends, falls back to the
    straight-line bridge.
    """
    t, progress = np.asarray(t, float), np.asarray(progress, float)
    t_speed, speed_ms = np.asarray(t_speed, float), np.asarray(speed_ms, float)
    out = bridge_pit_progress(t, progress, windows)
    keep = ~in_pit_mask(t, windows)
    t_k, p_k = t[keep], progress[keep]
    for t_in, t_out in windows:
        if not np.isfinite(t_out):
            continue
        inner = (t_speed > t_in) & (t_speed < t_out)
        grid = np.concatenate([[t_in], t_speed[inner], [t_out]])
        v = np.interp(grid, t_speed, speed_ms)
        dist = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(grid))])
        if dist[-1] <= 0:
            continue
        p_in = _edge_progress(t_k, p_k, t_in, before=True)
        p_out = _edge_progress(t_k, p_k, t_out, before=False)
        m = (t >= t_in) & (t <= t_out)
        out[m] = p_in + (p_out - p_in) * np.interp(t[m], grid, dist) / dist[-1]
    return out


def build_open_reference(xy: np.ndarray, spacing: float = 5.0) -> Reference:
    """Evenly spaced reference for an open path such as the pit lane, from the X/Y of one pass.

    Unlike `build_reference`, the path is not closed. Samples closer than `spacing` to
    the last kept one are dropped, so a car standing still does not add length through
    GPS jitter. Both ends are kept, so `project` on this reference returns the fraction
    (0 to 1) of the way along the path, clamped at the ends.

    Limitation: only jitter is handled. A sample that jumps backwards along the path
    (seen in a few real pit stops) adds length, so the reference must be built from a
    clean pass: no backward steps, and a path length close to the straight-line distance
    between its ends.
    """
    xy = np.asarray(xy, float)
    kept = [0]
    for i in range(1, len(xy)):
        if np.hypot(*(xy[i] - xy[kept[-1]])) >= spacing:
            kept.append(i)
    pts = xy[kept]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(s[-1])
    n = max(round(length / spacing), 1)
    s_new = np.linspace(0.0, length, n + 1)
    dense = np.column_stack([np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1])])
    return Reference(dense, s_new, length, cKDTree(dense))


def lane_bridge_progress(t, progress, windows, xy, lane_ref: Reference) -> np.ndarray:
    """Progress with pit lane samples placed by where the car is along the pit lane.

    Each sample inside a pit window is projected onto `lane_ref`
    (see `build_open_reference`), and the fraction of the way along the lane is mapped
    linearly onto the progress between pit entry and pit exit, which are extrapolated
    from the neighbouring samples outside the window, as in `speed_bridge_progress`. `xy`
    must be aligned with `t`. A window that never ends keeps the straight-line bridge.
    """
    t, progress, xy = np.asarray(t, float), np.asarray(progress, float), np.asarray(xy, float)
    out = bridge_pit_progress(t, progress, windows)
    keep = ~in_pit_mask(t, windows)
    t_k, p_k = t[keep], progress[keep]
    for t_in, t_out in windows:
        if not np.isfinite(t_out):
            continue
        m = (t >= t_in) & (t <= t_out)
        if not m.any():
            continue
        frac, _ = project(lane_ref, xy[m])
        p_in = _edge_progress(t_k, p_k, t_in, before=True)
        p_out = _edge_progress(t_k, p_k, t_out, before=False)
        out[m] = p_in + (p_out - p_in) * frac
    return out


def to_grid(t, progress, t_grid) -> np.ndarray:
    """A car's progress on a common time grid, with NaNs outside the span of its own data."""
    return np.interp(t_grid, t, progress, left=np.nan, right=np.nan)


def _monotone(progress, t_grid) -> np.ndarray:
    """Progress forced to be strictly increasing so it can be inverted.

    It can wobble backwards by a few metres, and inversion (the time at which a given
    progress was reached) needs it to increase.
    """
    progress = np.asarray(progress, float)
    mono = np.full_like(progress, np.nan)
    for i in range(progress.shape[0]):
        ok = np.isfinite(progress[i])
        if ok.any():
            tt = t_grid[ok]
            mono[i, ok] = np.maximum.accumulate(progress[i, ok]) + 1e-9 * (tt - tt[0])
    return mono


def on_track_gaps(progress, in_pit, t_grid):
    """For every car and time: the car physically ahead, how far, and the time headway to it.

    `progress` is (n_cars, n_times) in laps, with NaN where a car has no data. `in_pit`
    is a boolean array of the same shape. "Ahead" means the nearest car in front along
    the circuit whatever lap it is on, so a lapped car counts if it is physically in
    front. Cars in the pit lane, or without data, are never ahead. A car that is itself
    in the pit lane still gets a result, based on its bridged position.

    Returns (ahead, dist, gap_s), each shaped (n_cars, n_times):
      ahead : index of the car ahead, -1 if there is none
      dist  : distance to it along the circuit, in laps
      gap_s : seconds since that car was at the spot this car is at now (its own recent
              pace, not a fixed speed); NaN when that moment is before its data starts
    """
    progress = np.asarray(progress, float)
    in_pit = np.asarray(in_pit, bool)
    t_grid = np.asarray(t_grid, float)
    n, m = progress.shape

    mono = _monotone(progress, t_grid)
    frac = np.mod(mono, 1.0)

    # d[i, j, k]: distance forward along the circuit from car i to car j at time k.
    d = np.mod(frac[None, :, :] - frac[:, None, :], 1.0)
    usable = np.isfinite(frac) & ~in_pit
    d = np.where(usable[None, :, :], d, np.inf)
    d[np.arange(n), np.arange(n), :] = np.inf  # a car is never ahead of itself
    ahead = d.argmin(axis=1)
    dist = np.take_along_axis(d, ahead[:, None, :], axis=1)[:, 0, :]
    none = ~np.isfinite(dist) | ~np.isfinite(frac)
    ahead[none] = -1
    dist[none] = np.nan

    # Headway: when did the car ahead last pass the spot where this car is now?
    gap = np.full((n, m), np.nan)
    for j in range(n):
        ii, kk = np.nonzero(ahead == j)
        if len(ii) == 0:
            continue
        ok = np.isfinite(mono[j])
        target = mono[j, kk] - dist[ii, kk]
        t_star = np.interp(target, mono[j, ok], t_grid[ok], left=np.nan, right=np.nan)
        gap[ii, kk] = t_grid[kk] - t_star
    return ahead, dist, gap


def headway_to(progress, t_grid, i: int, j: int):
    """Distance and time headway from car i to one chosen car j, wherever each of them is.

    Same definition as `on_track_gaps`, but for a specific car instead of the nearest
    one, and pit lane cars are not excluded. Returns (dist in laps, gap in seconds).
    Just after car i passes car j, `dist` jumps from near 0 to near 1 lap, which marks
    the pass.
    """
    t_grid = np.asarray(t_grid, float)
    mono = _monotone(np.asarray(progress, float)[[i, j]], t_grid)
    frac = np.mod(mono, 1.0)
    dist = np.mod(frac[1] - frac[0], 1.0)
    ok = np.isfinite(mono[1])
    t_star = np.interp(mono[1] - dist, mono[1, ok], t_grid[ok], left=np.nan, right=np.nan)
    return dist, t_grid - t_star