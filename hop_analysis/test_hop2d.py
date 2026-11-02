"""hop2d の物理的な整合性チェック。"""

import math

import pytest

from hop2d import G, Params, motor_free_speed, simulate


def test_motor_matches_bam_figures_at_7v4():
    p = Params(friction_torque=0.0)
    # 無負荷回転数 V/kt ≈ 20 rad/s、停動トルク kt V / R ≈ 0.96 N·m
    assert motor_free_speed(0.0, 7.4, p) == pytest.approx(7.4 / 0.366)
    assert motor_free_speed(0.366 * 7.4 / 2.81, 7.4, p) == pytest.approx(0.0, abs=1e-9)


def test_air_time_and_rise_follow_ballistics():
    res = simulate(7.4)
    assert res.lifted_off
    # h = g T² / 8
    assert res.flight_rise == pytest.approx(G * res.air_time**2 / 8)


def test_higher_voltage_jumps_higher():
    rises = [simulate(v).flight_rise for v in (6.5, 7.4, 8.2)]
    assert rises[0] < rises[1] < rises[2]


def test_friction_and_backlash_reduce_jump():
    base = simulate(7.4).rise_above_extended
    assert simulate(7.4, Params(friction_torque=0.0)).rise_above_extended > base
    assert simulate(7.4, Params(backlash_deg=1.0)).flight_rise <= simulate(7.4).flight_rise + 1e-6


@pytest.mark.parametrize("voltage", [6.5, 7.4, 8.2])
def test_optimal_release_dominates_full_push_and_backlash(voltage):
    # 押し切りは特異点付近で接地したまま減速するので、最適解放以上にはならない。
    assert simulate(voltage).flight_rise >= simulate(voltage, release="full").flight_rise - 1e-9
    # ストロークを削るバックラッシュで跳躍が伸びてはいけない（旧実装の 6.5 V で逆転していた）。
    assert simulate(voltage, Params(backlash_deg=1.0)).flight_rise <= simulate(voltage).flight_rise + 1e-9


def test_too_heavy_robot_cannot_lift_off():
    res = simulate(7.4, Params(mass=5.0))
    assert not res.lifted_off
    assert res.takeoff_velocity == 0.0
    assert math.isfinite(res.push_time)
