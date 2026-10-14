"""Tests for track.py on a synthetic circuit with known car positions.

Run from the project root:
    uv run python src/test_track.py
(pytest also works, if installed)
"""
import numpy as np

from track import (
    bridge_pit_progress,
    build_open_reference,
    build_reference,
    calibrate_progress,
    headway_to,
    in_pit_mask,
    lane_bridge_progress,
    monotonic_violations,
    on_track_gaps,
    pit_windows,
    project,
    reconstruct,
    reference_from_lap,
    speed_bridge_progress,
    to_grid,
    unwrap_progress,
)

# A simple closed circuit: r = R + A*cos(3*theta). It never crosses itself.
R, A, K = 1000.0, 150.0, 3


def curve(theta):
    """Points on the test circuit for angles theta (radians)."""
    r = R + A * np.cos(K * theta)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


_th = np.linspace(0, 2 * np.pi, 200_001)
_s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(curve(_th), axis=0).T))])
LENGTH = _s[-1]


def position(progress):
    """True X/Y for a lap progress. The fractional part is the distance round the lap."""
    theta = np.interp(np.mod(progress, 1.0) * LENGTH, _s, _th)
    return curve(theta)


def make_car(p0, speed, t_end, noise=0.5, seed=0, hz=5.0):
    """Samples of a car with slightly varying speed (still always moving forward)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_end, 1 / hz)
    p = p0 + speed * t + 0.01 * np.sin(2 * np.pi * t / 37)
    return t, p, position(p) + rng.normal(0, noise, (len(t), 2))


def reference():
    """Reference loop built from one clean lap of a synthetic car."""
    t, p, xy = make_car(0.0, 0.0125, 80.0)
    return build_reference(xy[p < 1.0], spacing=5.0)


def recovered(p0, speed, t_end, first_lap_end=1):
    """Run a synthetic car through project, unwrap, and calibrate.

    Returns (t, true progress, recovered progress, lap-end error, residuals).
    """
    ref = reference()
    t, p, xy = make_car(p0, speed, t_end)
    u, resid = project(ref, xy)
    n = np.arange(first_lap_end, int(p.max()) + 1)
    t_end_n = np.interp(n, p, t)  # when the truth crosses each lap boundary
    prog, err = calibrate_progress(t, unwrap_progress(u), t_end_n, n)
    return t, p, prog, err, resid


def test_progress_matches_truth():
    _, p, prog, err, _ = recovered(-0.01, 0.0125, 420.0)  # starts just behind the line
    assert err < 0.005, err
    assert np.abs(prog - p).max() < 0.003


def test_formation_lap_is_absorbed():
    _, p, prog, err, _ = recovered(-1.01, 0.0125, 420.0)  # one extra lap before the start
    assert err < 0.005, err
    assert np.abs(prog - p).max() < 0.003


def test_lapped_car_keeps_correct_gap():
    _, pa, proga, _, _ = recovered(-0.01, 0.0125, 400.0)
    _, pb, progb, _, _ = recovered(-0.02, 0.0095, 400.0)
    assert (pa - pb).max() > 1.0  # A really has lapped B
    assert np.abs((proga - progb) - (pa - pb)).max() < 0.005


def test_residual_measures_distance_from_line():
    ref = reference()
    t, p, xy = make_car(0.0, 0.0125, 80.0)
    theta = np.interp(np.mod(p, 1.0) * LENGTH, _s, _th)
    off = xy + 5.0 * np.column_stack([np.cos(theta), np.sin(theta)])  # about 5 units off the line
    _, resid = project(ref, off)
    assert 4.0 < np.median(resid) < 5.1


def test_monotonic_check():
    _, _, prog, _, _ = recovered(-0.01, 0.0125, 420.0)
    assert monotonic_violations(prog) == 0
    bad = prog.copy()
    bad[200:260] -= 0.05  # a stretch projected onto the wrong part of the track
    assert monotonic_violations(bad) >= 1


def test_reference_starts_at_the_timing_instant():
    t, p, xy = make_car(0.0, 0.0125, 120.0)
    t_start = 0.1  # between two samples, like a real timing line
    p_start = np.interp(t_start, t, p)
    t_end = np.interp(p_start + 1.0, p, t)
    true_start = position(p_start)[0]

    exact = reference_from_lap(t, xy, t_start, t_end)
    naive_first = xy[t > t_start][0]  # what building from the raw samples would start at
    assert np.hypot(*(exact.xy[0] - true_start)) < 2.0
    assert np.hypot(*(naive_first - true_start)) > 5.0  # the old behaviour really was off
    assert abs(exact.length - LENGTH) / LENGTH < 0.01


def test_reconstruct_drops_nan_samples():
    ref = reference()
    t, p, xy = make_car(-0.01, 0.0125, 420.0)
    xy[100:103] = np.nan  # missing position samples
    n = np.arange(1, int(p.max()) + 1)
    tt, prog, resid, err = reconstruct(ref, t, xy, np.interp(n, p, t), n)
    assert len(tt) == len(t) - 3
    assert err < 0.005, err
    assert np.abs(prog - np.interp(tt, t, p)).max() < 0.003


def test_pit_windows():
    assert pit_windows([100.0, 200.0], [110.0, 205.0]) == [(100.0, 110.0), (200.0, 205.0)]
    # A double stop, like ALO on laps 54-55: in, out, in, out.
    assert pit_windows([100.0, 130.0], [115.0, 145.0]) == [(100.0, 115.0), (130.0, 145.0)]
    # Retired in the pits: no exit after the second entry.
    assert pit_windows([100.0, 200.0], [110.0]) == [(100.0, 110.0), (200.0, np.inf)]
    # NaNs dropped; an exit before any entry (pit lane start) is ignored.
    assert pit_windows([np.nan, 100.0], [50.0, 110.0, np.nan]) == [(100.0, 110.0)]


def test_in_pit_mask_includes_both_ends():
    m = in_pit_mask([9.0, 10.0, 15.0, 20.0, 21.0], [(10.0, 20.0)])
    assert m.tolist() == [False, True, True, True, False]


def _stop_scenario():
    """A car that slows through a pit lane and sits still for 4 s, plus its speed signal.

    Returns (t, true progress, speed, kept sample times, recovered progress).
    """
    ref = reference()
    dt, v0 = 0.2, 0.0125
    t = np.arange(0.0, 300.0, dt)
    v = np.full(len(t), v0)
    v[(t >= 100) & (t < 120)] = 0.005
    v[(t >= 108) & (t < 112)] = 0.0
    p = -0.01 + np.cumsum(v) * dt
    xy = position(p) + np.random.default_rng(1).normal(0, 0.5, (len(t), 2))
    n = np.arange(1, int(p.max()) + 1)
    tt, prog, _, _ = reconstruct(ref, t, xy, np.interp(n, p, t), n)
    return t, p, v, tt, prog


def test_constant_speed_bridge_is_a_straight_line_between_entry_and_exit():
    t, p, v, tt, prog = _stop_scenario()
    windows = [(100.0, 120.0)]
    mask = in_pit_mask(tt, windows)
    bridged = bridge_pit_progress(tt, prog, windows)
    assert np.allclose(bridged[~mask], prog[~mask])  # nothing changes outside the stop
    speed = np.diff(bridged[mask]) / np.diff(tt[mask])
    assert np.allclose(speed, speed[0], rtol=1e-6)   # constant speed through the stop


def test_speed_bridge_beats_constant_speed_bridge():
    t, p, v, tt, prog = _stop_scenario()
    windows = [(100.0, 120.0)]
    truth = np.interp(tt, t, p)
    mask = in_pit_mask(tt, windows)
    speed_ms = v * LENGTH  # any consistent unit works: the result is rescaled to the exit
    sb = speed_bridge_progress(tt, prog, windows, t, speed_ms)
    cb = bridge_pit_progress(tt, prog, windows)
    assert np.allclose(sb[~mask], prog[~mask])     # nothing changes outside the stop
    assert np.abs(sb - truth)[mask].max() < 0.002  # tracks the stationary stop
    assert np.abs(cb - truth)[mask].max() > 0.005  # the constant-speed bridge does not
    assert np.all(np.diff(sb) > -0.003)            # still moves forward


def test_speed_bridge_falls_back_without_speed_or_exit():
    t, p, v, tt, prog = _stop_scenario()
    zero = np.zeros_like(t)
    for windows, speed in (([(100.0, 120.0)], zero), ([(100.0, np.inf)], v * LENGTH)):
        sb = speed_bridge_progress(tt, prog, windows, t, speed)
        assert np.allclose(sb, bridge_pit_progress(tt, prog, windows))


def lane_curve(u):
    """A gently curved 'pit lane' about 3200 units long. u runs from 0 to 1 along it."""
    u = np.asarray(u, float)
    return np.column_stack([3200.0 * u, 200.0 * np.sin(np.pi * u)])


def test_open_reference_ignores_standing_still():
    path = lane_curve(np.linspace(0, 1, 80))
    jittery = np.vstack([path[:40], np.repeat(path[40:41], 25, axis=0), path[41:]])
    clean = build_open_reference(path)
    ref = build_open_reference(jittery)
    assert abs(ref.length - clean.length) / clean.length < 0.001
    assert np.all(np.diff(ref.s) > 0)
    assert ref.s[0] == 0.0 and ref.s[-1] == ref.length


def test_open_reference_fraction_is_clamped_at_the_ends():
    ref = build_open_reference(lane_curve(np.linspace(0, 1, 80)))
    frac, _ = project(ref, np.array([[-500.0, 0.0], [1600.0, 200.0], [3800.0, 0.0]]))
    assert frac[0] == 0.0 and frac[2] == 1.0
    assert abs(frac[1] - 0.5) < 0.01


def test_lane_bridge_follows_the_car_through_the_stop():
    t, p, v, tt, prog = _stop_scenario()
    windows = [(100.0, 120.0)]
    truth = np.interp(tt, t, p)
    p_in, p_out = np.interp([100.0, 120.0], t, p)
    frac = np.clip((truth - p_in) / (p_out - p_in), 0.0, 1.0)
    rng = np.random.default_rng(2)
    xy = lane_curve(frac) + rng.normal(0, 0.5, (len(tt), 2))        # only in-window rows are used
    lane = build_open_reference(lane_curve(np.linspace(0, 1, 90)))  # a separate pass of the lane
    mask = in_pit_mask(tt, windows)

    lb = lane_bridge_progress(tt, prog, windows, xy, lane)
    cb = bridge_pit_progress(tt, prog, windows)
    assert np.allclose(lb[~mask], prog[~mask])     # nothing changes outside the stop
    assert np.abs(lb - truth)[mask].max() < 0.003  # follows the stationary stop
    assert np.abs(cb - truth)[mask].max() > 0.005  # the constant-speed bridge does not

    sb = speed_bridge_progress(tt, prog, windows, t, v * LENGTH)
    assert np.abs(lb - sb)[mask].max() < 0.004  # the two methods agree


T_GRID = np.arange(0.0, 400.0, 0.5)


def _cars(speeds, p0):
    """Progress on T_GRID for cars with constant speeds (laps/s) and start offsets (laps)."""
    speeds, p0 = np.asarray(speeds, float), np.asarray(p0, float)
    return p0[:, None] + speeds[:, None] * T_GRID[None, :]


def test_to_grid_is_nan_outside_the_data():
    g = to_grid(np.array([10.0, 20.0]), np.array([1.0, 2.0]), np.array([5.0, 15.0, 25.0]))
    assert np.isnan(g[0]) and np.isnan(g[2]) and abs(g[1] - 1.5) < 1e-12


def test_ahead_car_and_headway_match_brute_force():
    speeds = np.array([0.0125, 0.0100, 0.0110])
    prog = _cars(speeds, [0.00, 0.30, 0.55])
    ahead, dist, gap = on_track_gaps(prog, np.zeros(prog.shape, bool), T_GRID)
    for k in (0, 40, 200, 500, 790):
        frac = np.mod(prog[:, k], 1.0)
        for i in range(3):
            d = np.mod(frac - frac[i], 1.0)
            d[i] = np.inf
            j = int(np.argmin(d))
            assert ahead[i, k] == j
            assert abs(dist[i, k] - d[j]) < 1e-9
            if T_GRID[k] >= d[j] / speeds[j]:
                assert abs(gap[i, k] - d[j] / speeds[j]) < 1e-4  # constant speed: headway = d / v
            else:
                assert np.isnan(gap[i, k])  # the car ahead had not yet started


def test_lapped_car_in_front_counts_as_ahead():
    prog = _cars([0.0125, 0.0100], [1.05, 0.15])  # A is a lap up on B but B is just in front of A
    ahead, dist, gap = on_track_gaps(prog, np.zeros(prog.shape, bool), T_GRID)
    k = 20  # t = 10 s: A at 1.175 laps, B at 0.25 laps, so B is 0.075 laps ahead on the road
    assert ahead[0, k] == 1
    assert abs(dist[0, k] - 0.075) < 1e-9
    assert abs(gap[0, k] - 0.075 / 0.0100) < 1e-4
    assert ahead[1, k] == 0


def test_cars_in_the_pit_or_without_data_are_never_ahead():
    prog = _cars([0.0125, 0.0110, 0.0100], [0.0, 0.10, 0.20])
    in_pit = np.zeros(prog.shape, bool)
    in_pit[1, 100:200] = True  # car 1 in the pit lane
    prog[1, 300:] = np.nan     # car 1 retired
    ahead, dist, gap = on_track_gaps(prog, in_pit, T_GRID)
    assert np.all(ahead[0, 100:200] != 1)
    assert np.all(ahead[0, 300:] != 1)
    assert np.all(ahead[1, 300:] == -1) and np.isnan(gap[1, 300:]).all()
    assert np.all(ahead[0, 300:] >= 0)  # car 2 is still there


def test_headway_to_a_chosen_car_and_the_pass():
    # B is just ahead of A and slower, so A closes in, passes, and B ends up just behind.
    prog = _cars([0.0125, 0.0100], [0.00, 0.10])
    dist, gap = headway_to(prog, T_GRID, 0, 1)
    k = 40  # t = 20 s: A at 0.25, B at 0.30
    assert abs(dist[k] - 0.05) < 1e-9
    assert abs(gap[k] - 0.05 / 0.0100) < 1e-4
    jump = np.where(np.diff(dist) > 0.5)[0]
    assert len(jump) == 1  # exactly one pass in this window
    t_pass = T_GRID[jump[0]]
    assert abs(t_pass - 0.10 / (0.0125 - 0.0100)) < 1.0  # when A has made up the 0.10 lap


def test_headway_to_ignores_the_pit_lane_unlike_on_track_gaps():
    prog = _cars([0.0125, 0.0100], [0.00, 0.10])
    in_pit = np.zeros(prog.shape, bool)
    in_pit[1, :] = True
    assert np.all(on_track_gaps(prog, in_pit, T_GRID)[0][0] == -1)  # nobody usable ahead of car 0
    assert np.isfinite(headway_to(prog, T_GRID, 0, 1)[1][40])       # but a chosen car is measured


def test_unwrap_counts_forward_crossings_and_backward_wobbles():
    u = np.array([0.95, 0.99, 0.01, 0.99, 0.02])  # cross the line, wobble back over it, cross again
    assert np.allclose(unwrap_progress(u), [0.95, 0.99, 1.01, 0.99, 1.02])


def test_open_reference_is_inflated_by_a_backward_step():
    """Known limitation (see `build_open_reference`): a sample that jumps back adds length.

    If the function is ever made robust to this, flip the assertion.
    """
    path = lane_curve(np.linspace(0, 1, 80))
    stepped = np.vstack([path[:30], path[26:27], path[30:]])  # one sample jumps four steps back
    clean = build_open_reference(path)
    ref = build_open_reference(stepped)
    assert ref.length > 1.05 * clean.length


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)