"""microduck の真上片足ジャンプを xz 2 次元・折れ線脚で見積もる。

モデル（簡略化の仮定）:
- 全質量 m を股（hip）位置の質点に集中させる。脚は質量なし。
- 脚は 太もも L1・すね L2 の 2 リンク＋足（足首高さ h_a、足首→つま先 l_t）の折れ線。
- 立ち足は真上にだけ蹴る（水平方向はバランス制御で完全に拘束されると仮定）。
- 駆動は 膝 と 足首 の 2 関節（直列）。股ピッチは質点を股に置くので上下に寄与しない。
- 膝: 股–足首距離 d = 2L cos(κ/2)（L1 = L2 = L、κ は膝の曲げ角、0 で伸展）。
- 足首: 足がつま先まわりに回転し、足首点の高さ y_a = r sin β（r = |足首→つま先|）。
- サーボは DC モータ近似（出力軸換算）: τ = kt (V − kt ω) / R − τ_f。
  kt, R は BAM m6 同定値、τ_f はクーロン摩擦（停動 0.96 → 正味 0.70 N·m から 0.26 N·m）。
- 脚が質量なしなので、直列チェーンには同じ鉛直力 F が伝わる。
  各関節は F·J_i のトルクを出す速度で回り、ẏ = Σ J_i ω_i から F を解く。
- F ≤ 0（地面は引けない）または全関節が可動端に達したら離地。以後は放物運動。
- 伸展端付近では J → 0 の特異点で F < mg となり、接地したまま減速する。
  方策は伸展を止めれば即座に抜重できる（脚が質量なし）ので、既定では
  蹴り出し中の鉛直速度が最大の瞬間に離地する「最適解放」で評価する。
  ``release="full"`` で最大トルクのまま押し切る開ループ版も出せる。
- バックラッシュ: 押し方向に既に負荷がかかっている（重力で予圧）とみなし、
  各関節の可動端（ストローク）を 1° 減らす最悪ケースとして扱う。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

G = 9.81


@dataclass(frozen=True)
class Params:
    mass: float = 0.737  # [kg] robot_groundcontact.xml の合計
    leg_link: float = 0.042  # [m] 太もも・すね
    ankle_height: float = 0.0258  # [m] 足首の高さ
    toe_length: float = 0.0342  # [m] 足首→つま先（水平）
    kt: float = 0.366  # [N·m/A] 出力軸換算トルク定数（BAM m6）
    resistance: float = 2.81  # [Ω]（BAM m6）
    friction_torque: float = 0.26  # [N·m] クーロン摩擦（停動 0.96 − 正味 0.70）
    knee_start_deg: float = 90.0  # しゃがみ姿勢の膝曲げ角
    knee_end_deg: float = 0.0  # 膝の伸展端
    backlash_deg: float = 0.0  # 各関節のストローク損失（バックラッシュ）
    dt: float = 1e-5  # [s] 積分刻み


@dataclass(frozen=True)
class Result:
    voltage: float
    lifted_off: bool
    push_time: float  # [s] 蹴り出し時間
    takeoff_velocity: float  # [m/s]
    stroke: float  # [m] 蹴り出し中の股の上昇量
    rise_above_extended: float  # [m] 伸展立位の股高さから頂点までの上昇
    flight_rise: float  # [m] 離地点から頂点までの上昇 = v²/2g
    air_time: float  # [s] 離地高さに戻るまで = 2v/g
    peak_force: float  # [N] 地面反力のピーク


def motor_free_speed(torque: float, voltage: float, p: Params) -> float:
    """出力トルク torque を出しているときの関節速度 [rad/s]（摩擦込み、負なら押し戻される）。"""
    stall = p.kt * voltage / p.resistance
    no_load = voltage / p.kt
    return no_load * (1.0 - (torque + p.friction_torque) / stall)


def simulate(
    voltage: float, p: Params = Params(), release: str = "optimal", trace: list | None = None
) -> Result:
    """release: "optimal"（速度最大で離地）または "full"（最大トルクで押し切る）。

    trace にリストを渡すと、蹴り出し中の (t, κ, β, y, v) を毎ステップ追記する。
    """
    if release not in ("optimal", "full"):
        raise ValueError(f"unknown release: {release!r}")
    L = p.leg_link
    r = math.hypot(p.ankle_height, p.toe_length)
    beta0 = math.atan2(p.ankle_height, p.toe_length)
    beta_end = math.pi / 2 - math.radians(p.backlash_deg)
    kappa_end = math.radians(p.knee_end_deg + p.backlash_deg)
    kappa = math.radians(p.knee_start_deg)
    beta = beta0

    stall = p.kt * voltage / p.resistance
    no_load = voltage / p.kt

    def hip_height(kappa: float, beta: float) -> float:
        return r * math.sin(beta) + 2 * L * math.cos(kappa / 2)

    y0 = hip_height(kappa, beta)
    y_extended = hip_height(kappa_end, beta_end)
    v = 0.0
    t = 0.0
    peak_force = 0.0
    lifted_off = False
    while t < 1.0:
        if trace is not None:
            trace.append((t, kappa, beta, hip_height(kappa, beta), v))
        # 鉛直速度へのヤコビアン（関節が伸展方向に回るときの dy/dq）。可動端の関節は 0。
        j_knee = L * math.sin(kappa / 2) if kappa > kappa_end else 0.0
        j_ankle = r * math.cos(beta) if beta < beta_end else 0.0
        js = [j for j in (j_knee, j_ankle) if j > 1e-9]
        if not js:
            lifted_off = v > 0
            break
        # ẏ = Σ J_i ω0 (1 − (F J_i + τ_f)/τ_s) を F について解く。
        free = sum(j * no_load * (1 - p.friction_torque / stall) for j in js)
        gain = sum(j * j * no_load / stall for j in js)
        force = (free - v) / gain
        if force <= 0.0:
            lifted_off = True
            break
        peak_force = max(peak_force, force)
        if j_knee > 1e-9:
            kappa -= motor_free_speed(force * j_knee, voltage, p) * p.dt
            kappa = max(kappa, kappa_end)
        if j_ankle > 1e-9:
            beta += motor_free_speed(force * j_ankle, voltage, p) * p.dt
            beta = min(beta, beta_end)
        v_next = v + (force / p.mass - G) * p.dt
        if release == "optimal" and v > 0.0 and v_next < v:
            # 速度が減り始めた = F < mg。ここで伸展を止めて離地する。
            lifted_off = True
            break
        v = v_next
        t += p.dt
        if hip_height(kappa, beta) < y0 - 0.01:  # 支えきれず沈み込んだ
            break

    y_lo = hip_height(kappa, beta)
    v_lo = max(v, 0.0) if lifted_off else 0.0
    flight_rise = v_lo**2 / (2 * G)
    return Result(
        voltage=voltage,
        lifted_off=lifted_off and v_lo > 0,
        push_time=t,
        takeoff_velocity=v_lo,
        stroke=y_lo - y0,
        rise_above_extended=y_lo + flight_rise - y_extended,
        flight_rise=flight_rise,
        air_time=2 * v_lo / G,
        peak_force=peak_force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voltages", type=float, nargs="+", default=[6.5, 7.4, 8.2])
    args = parser.parse_args()

    cases = [
        ("最適解放・摩擦あり・BLなし", Params(), "optimal"),
        ("最適解放・摩擦あり・BL1°", Params(backlash_deg=1.0), "optimal"),
        ("最適解放・摩擦なし（上限）", Params(friction_torque=0.0), "optimal"),
        ("押し切り・摩擦あり・BLなし", Params(), "full"),
    ]
    header = "| 条件 | V | 離地 | 蹴り出し[ms] | 離地速度[m/s] | 浮き上がり v²/2g [mm] | 伸展立位比 [mm] | 滞空[ms] | 反力ピーク[N] |"
    print(header)
    print("|" + "---|" * 9)
    for name, p, release in cases:
        for v in args.voltages:
            res = simulate(v, p, release)
            print(
                f"| {name} | {v:.1f} | {'○' if res.lifted_off else '×'} | {res.push_time * 1e3:.1f} | "
                f"{res.takeoff_velocity:.3f} | {res.flight_rise * 1e3:.1f} | {res.rise_above_extended * 1e3:.1f} | "
                f"{res.air_time * 1e3:.1f} | {res.peak_force:.1f} |"
            )


if __name__ == "__main__":
    main()
