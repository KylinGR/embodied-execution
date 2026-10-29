"""Rerunnable positive/negative checks of the independent jump qualifier.

Run with PYTHONPATH=examples python examples/check_jump_metrics.py.
Synthetic traces test event logic, not dynamic feasibility or robot performance.
"""
from copy import deepcopy
import json

import mujoco
import numpy as np

from diablo_training.env import DiabloEnv
from jump_metrics import sample, summarize


def fixture():
    rows = []
    for i in range(10001):
        time = i * .002
        flight = 1.1 <= time < 1.2
        rows.append(dict(time=time, xy=[0., 0.],
                         wheel_bottoms=[.02 if flight else 0.] * 2,
                         wheel_normal_n=[0. if flight else 100.] * 2,
                         other_normal_n=0., upright=1., world_velocity=[0., 0., 0.],
                         gyro=[0., 0., 0.], com_z=.25, com_vz=.2 if flight else 0.,
                         joint_power_w=10. if flight else 0.))
    return rows


def main():
    reports = []

    def check(name, rows, expected, reason=None):
        result = summarize(rows)
        assert result["passed"] is expected, (name, result)
        if reason:
            assert reason in result["failures"], (name, result)
        json.dumps(result, allow_nan=False)
        reports.append({"case": name, "passed_as_expected": True,
                        "qualification_passed": result["passed"], "failures": result["failures"]})

    baseline = fixture()
    check("positive_complete_jump", baseline, True)
    check("positive_without_t0_row", baseline[1:], True)

    rows = deepcopy(baseline)
    for row in rows:
        row.update(wheel_bottoms=[0., 0.], wheel_normal_n=[100., 100.], com_vz=0.)
    check("no_jump", rows, False, "requires_exactly_one_flight")

    rows = deepcopy(baseline)
    for row in rows:
        row["wheel_bottoms"] = [0., 0.]
        row["com_z"] = .5 if 1.1 <= row["time"] < 1.2 else .25
    check("only_body_raised", rows, False, "requires_exactly_one_flight")

    rows = deepcopy(baseline)
    for row in rows:
        flight = row["time"] < .1
        row.update(wheel_bottoms=[.02 if flight else 0.] * 2,
                   wheel_normal_n=[0. if flight else 100.] * 2, com_vz=-.2 if flight else 0.)
    check("initial_free_release", rows, False, "takeoff_outside_window")

    rows = deepcopy(baseline)
    for row in rows:
        if .8 <= row["time"] < 1.1:
            row.update(wheel_normal_n=[0., 0.], wheel_bottoms=[.001, .001])
    check("old_stand_then_unsupported_below_clearance", rows, False, "takeoff_outside_window")

    for transition, expected in ((.020, True), (.022, False)):
        rows = deepcopy(baseline)
        for row in rows:
            if 1.1-transition-1e-9 <= row["time"] < 1.1-1e-9:
                row["wheel_normal_n"] = [100., 0.]
        check(f"single_wheel_transition_{transition:.3f}s", rows, expected,
              None if expected else "no_ground_support_before_takeoff")

    for gap, expected in ((.020, True), (.022, False)):
        rows = deepcopy(baseline)
        for row in rows:
            if 2.-1e-9 <= row["time"] < 2.+gap-1e-9:
                row["wheel_normal_n"] = [0., 0.]
        check(f"postlanding_support_gap_{gap:.3f}s", rows, expected,
              None if expected else "postlanding_support_loss")

    rows = deepcopy(baseline)
    for row in rows:
        row["com_vz"] = 0.
    check("retract_wheels_without_com_ascent", rows, False, "no_active_com_ascent")

    rows = deepcopy(baseline)
    for row in rows:
        if 2.1 <= row["time"] < 2.2:
            row.update(wheel_bottoms=[.02, .02], wheel_normal_n=[0., 0.], com_vz=.2)
    check("two_jumps", rows, False, "requires_exactly_one_flight")

    rows = deepcopy(baseline)
    rows[500]["other_normal_n"] = 1.01
    check("nonwheel_support", rows, False, "nonwheel_support")
    rows = deepcopy(baseline)
    rows[-1]["upright"] = .9
    check("terminal_stability_not_latched", rows, False, "insufficient_terminal_stability")
    check("short_episode", baseline[:-1], False, "incomplete_20s_episode")
    check("missing_physics_sample", baseline[:500]+baseline[501:], False, "sampling_not_contiguous_500hz")

    env = DiabloEnv()
    try:
        env.reset(seed=0)
        env.data.qvel[2] = .3
        mujoco.mj_forward(env.model, env.data)
        measured = sample(env)
        assert abs(measured["com_vz"]-.3) < 1e-12, measured
        assert np.allclose(measured["world_velocity"], [0., 0., .3]), measured
        assert np.allclose(measured["wheel_bottoms"], [.005, .005], atol=1e-6), measured
        json.dumps(measured, allow_nan=False)
        reports.append({"case": "physical_world_com_velocity_and_wheel_clearance", "passed_as_expected": True})
    finally:
        env.close()
    print(json.dumps({"all_checks_passed": True, "checks": reports}, indent=2))


if __name__ == "__main__":
    main()
