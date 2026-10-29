"""Independent, truth-based single-jump measurements; never actor inputs.

Rows must be sampled after every 2 ms physics step, beginning at t=0 or
t=0.002. Geometric clearance and load-bearing contact are separate tests.
"""
import mujoco
import numpy as np

DT = 0.002
REQUIREMENTS = {
    "sampling_hz": 500, "episode_seconds": 20.0,
    "prestand_seconds": 0.5, "takeoff_window_s": [1.0, 3.0],
    "minimum_clearance_m": 0.002, "minimum_flight_s": 0.060,
    "peak_clearance_range_m": [0.015, 0.030], "minimum_com_vz_mps": 0.15,
    "landing_confirmation_s": 0.050, "maximum_postlanding_support_gap_s": 0.020,
    "maximum_prelaunch_single_wheel_transition_s": 0.020,
    "terminal_stable_seconds": 3.0, "maximum_drift_m": 0.2,
    "last_5s_drift_m": 0.05, "contact_threshold_n": 1.0,
    "minimum_upright": 0.95, "maximum_horizontal_speed_mps": 0.15,
    "maximum_gyro_rps": 1.0,
}


def sample(env):
    """Return a JSON-serializable physical sample, with COM velocity in world axes.

    mj_subtreeVel explicitly refreshes the derived subtree velocities. The
    floating base subtree contains the entire robot; its subtree_com and
    subtree_linvel are mass-weighted robot COM position and velocity.
    """
    model, data = env.model, env.data
    cache = getattr(env, "_jump_metrics_geometry_cache", None)
    if cache is None:
        wheel_geoms = [[], []]
        wheel_bodies = []
        for side in ("left", "right"):
            wheel_bodies.append(model.body(side + "_wheel").id)
        for geom in range(model.ngeom):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            body = int(model.geom_bodyid[geom])
            if body in wheel_bodies:
                if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CYLINDER:
                    raise ValueError("Single-jump metrics require cylindrical wheel collision geoms")
                wheel_geoms[wheel_bodies.index(body)].append(geom)
        if not all(wheel_geoms):
            raise ValueError("Missing wheel collision geometry")
        # Reject a partial subtree rather than silently calling it whole-robot COM.
        if not np.isclose(model.body_subtreemass[env.base_id], model.body_mass.sum()):
            raise ValueError("Base subtree does not contain the entire robot")
        cache = (wheel_geoms, wheel_bodies)
        env._jump_metrics_geometry_cache = cache
    wheel_geoms, wheel_bodies = cache
    bottoms = []
    for geoms in wheel_geoms:
        values = []
        for geom in geoms:
            axis_z = float(data.geom_xmat[geom].reshape(3, 3)[2, 2])
            radius, halfwidth = model.geom_size[geom, :2]
            support = radius * np.sqrt(max(0.0, 1.0 - axis_z**2)) + halfwidth * abs(axis_z)
            values.append(float(data.geom_xpos[geom, 2] - support))
        bottoms.append(min(values))
    normal = [0.0, 0.0]
    other_normal = 0.0
    wrench = np.zeros(6)
    for index, contact in enumerate(data.contact):
        if env.floor_id not in (contact.geom1, contact.geom2):
            continue
        geom = contact.geom2 if contact.geom1 == env.floor_id else contact.geom1
        body = int(model.geom_bodyid[geom])
        mujoco.mj_contactForce(model, data, index, wrench)
        force = max(0.0, float(wrench[0]))
        if body in wheel_bodies:
            normal[wheel_bodies.index(body)] += force
        else:
            other_normal += force
    mujoco.mj_subtreeVel(model, data)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, env.base_id, velocity, 0)
    return {
        "time": float(data.time), "wheel_bottoms": bottoms,
        "wheel_normal_n": normal, "other_normal_n": other_normal,
        "com_z": float(data.subtree_com[env.base_id, 2]),
        "com_vz": float(data.subtree_linvel[env.base_id, 2]),
        "xy": data.xpos[env.base_id, :2].tolist(),
        "world_velocity": velocity[3:].tolist(),
        "gyro": data.sensor("gyro").data.tolist(),
        "upright": float(data.xmat[env.base_id].reshape(3, 3)[2, 2]),
        "joint_power_w": float(np.dot(data.actuator_force, data.actuator_velocity)),
    }


def _runs(mask):
    """Half-open runs of true samples."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _longest(mask):
    return float(max((end - start for start, end in _runs(mask)), default=0) * DT)


def summarize(rows):
    """Recompute strict qualification from raw rows; no historical success latch.

    Durations use occupied 2 ms sample bins. A t=0 row establishes the origin
    but occupies no elapsed-time bin. Invalid/missing/irregular data cannot pass.
    """
    rows = list(rows)
    if not rows:
        return {"passed": False, "failures": ["empty_trace"], "requirements": REQUIREMENTS.copy()}
    times = np.asarray([r["time"] for r in rows], dtype=float)
    xy_all = np.asarray([r["xy"] for r in rows], dtype=float)
    origin = xy_all[0].copy()
    valid_sampling = bool(np.isfinite(times).all() and times[0] >= -1e-9
                          and times[0] <= DT + 1e-8
                          and np.allclose(np.diff(times), DT, rtol=0, atol=1e-8))
    # The initial state is not an elapsed interval or an additional flight bin.
    active = times > 1e-9
    samples = [r for r, keep in zip(rows, active) if keep]
    if not samples:
        return {"passed": False, "failures": ["no_elapsed_samples"], "requirements": REQUIREMENTS.copy()}
    t = times[active]
    xy = xy_all[active]
    normal = np.asarray([r["wheel_normal_n"] for r in samples], dtype=float)
    clearance = np.min(np.asarray([r["wheel_bottoms"] for r in samples], dtype=float), axis=1)
    other = np.asarray([r["other_normal_n"] for r in samples], dtype=float)
    upright = np.asarray([r["upright"] for r in samples], dtype=float)
    velocity = np.asarray([r["world_velocity"] for r in samples], dtype=float)
    gyro = np.asarray([r["gyro"] for r in samples], dtype=float)
    vz = np.asarray([r["com_vz"] for r in samples], dtype=float)
    finite = all(np.isfinite(a).all() for a in (xy, normal, clearance, other, upright, velocity, gyro, vz))
    dual = np.all(normal > 1.0, axis=1)
    stable = dual & (other <= 1.0) & (upright >= .95) & (np.linalg.norm(velocity[:, :2], axis=1) <= .15) & (np.linalg.norm(gyro, axis=1) <= 1.)
    airborne = np.all(normal <= 1.0, axis=1) & (clearance > .002)
    unsupported = np.all(normal <= 1.0, axis=1)
    flights = [(int(a), int(b)) for a, b in _runs(airborne) if (b-a)*DT >= .060-1e-9]
    prestand = _longest(stable & (t < 1.0-1e-9))
    terminal = float(next(((b-a)*DT for a, b in reversed(_runs(stable)) if b == len(stable)), 0.))
    max_drift = float(np.max(np.linalg.norm(xy-origin, axis=1)))
    tail = xy[t > 15.0+1e-9]
    tail_drift = float(np.linalg.norm(np.ptp(tail, axis=0))) if len(tail) else float("inf")
    flight_reports = []
    landing_index = None
    for a, b in flights:
        # Follow the measured flight back through the <=2 mm clearance region.
        # An old stand event cannot certify a later unsupported release as an
        # active ground takeoff. Allow up to 20 ms of single-wheel transition.
        launch = a
        while launch > 0 and unsupported[launch-1]:
            launch -= 1
        previous_dual = np.flatnonzero(dual[:launch])
        transition_s = None if not len(previous_dual) else float((launch-int(previous_dual[-1])-1)*DT)
        preceding_support = bool(launch > 0 and not unsupported[launch-1]
                                 and transition_s is not None and transition_s <= .020+1e-9)
        # Time is the sample-bin endpoint, so its interval began one DT earlier.
        start_time = max(0., float(t[launch]-DT))
        landing = next(((c, d) for c, d in _runs(dual & (np.arange(len(t)) >= b)) if (d-c)*DT >= .050-1e-9), None)
        if len(flights) == 1 and landing is not None:
            landing_index = int(landing[0])
        flight_reports.append({"start_s": start_time, "end_s": float(t[b-1]),
                               "clearance_start_s": max(0., float(t[a]-DT)),
                               "duration_s": (b-a)*DT,
                               "peak_clearance_m": float(np.max(clearance[a:b])),
                               "peak_com_vz_mps": float(np.max(vz[launch:b])),
                               "preceded_by_ground_support": preceding_support,
                               "single_wheel_transition_s": transition_s,
                               "landing_s": None if landing is None else float(t[landing[0]]-DT)})
    gap = None if landing_index is None else _longest(~dual[landing_index:])
    failures = []
    if not valid_sampling: failures.append("sampling_not_contiguous_500hz")
    if not finite: failures.append("nonfinite_state")
    if abs(float(t[-1])-20.) > 1e-7 or len(samples) != 10000: failures.append("incomplete_20s_episode")
    if prestand < .5-1e-9: failures.append("insufficient_prestand_before_1s")
    if np.any(upright < .5): failures.append("fell")
    if np.any(other > 1.): failures.append("nonwheel_support")
    if len(flights) != 1: failures.append("requires_exactly_one_flight")
    if len(flights) == 1:
        f = flight_reports[0]
        if not 1.-1e-9 <= f["start_s"] <= 3.+1e-9: failures.append("takeoff_outside_window")
        if not f["preceded_by_ground_support"]: failures.append("no_ground_support_before_takeoff")
        if not .015 <= f["peak_clearance_m"] <= .030 or float(np.max(clearance)) > .030:
            failures.append("clearance_outside_target")
        if not f["peak_com_vz_mps"] > .15: failures.append("no_active_com_ascent")
    if landing_index is None: failures.append("no_confirmed_dual_wheel_landing")
    elif gap > .020+1e-9: failures.append("postlanding_support_loss")
    if terminal < 3.-1e-9: failures.append("insufficient_terminal_stability")
    if max_drift > .2: failures.append("excessive_global_drift")
    if tail_drift > .05: failures.append("excessive_last5s_drift")
    return {"passed": not failures, "failures": failures,
            "duration_s": float(t[-1]), "sample_count": len(samples),
            "prestand_seconds": prestand, "terminal_stable_seconds": terminal,
            "max_drift_m": max_drift, "last_5s_drift_m": tail_drift,
            "postlanding_max_support_gap_s": gap,
            "peak_clearance_m": float(np.max(clearance)),
            "flight_count": len(flights), "flights": flight_reports,
            "requirements": REQUIREMENTS.copy()}
