"""hop2d の真上片足ジャンプを、xz 平面の折れ線脚として条件ごとに GIF に描く。

描画の約束（hop2d のモデルに合わせる）:
- 蹴り出し中はつま先を原点に固定し、足はつま先まわりに回転する。
- 股（質点）は足首の真上にあり、股–足首距離は d = 2L cos(κ/2)。膝は前方（+x）に出す。
- 離地後は離地時の関節角のまま剛体として放物運動し、離地高さに戻ったところで着地とする。
GIF 内の文字は、コンテナに日本語フォントがないため英語にしている。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from hop2d import G, Params, simulate  # noqa: E402

FRAME_DT = 0.002  # [s] 1 フレームあたりのシミュレーション時間
FPS = 25  # 0.002 s / 0.04 s → 20 倍スロー
HOLD_FRAMES = 12  # 開始・終了の静止フレーム数

CASES = [
    ("optimal", "optimal release, friction, no backlash", Params(), "optimal"),
    ("optimal_backlash", "optimal release, friction, backlash 1deg", Params(backlash_deg=1.0), "optimal"),
    ("optimal_nofriction", "optimal release, no friction (upper bound)", Params(friction_torque=0.0), "optimal"),
    ("full_push", "full push (open loop), friction, no backlash", Params(), "full"),
]


def leg_points(kappa: float, beta: float, p: Params, lift: float = 0.0) -> list[tuple[float, float]]:
    """つま先→足裏（足首直下）→足首→膝→股 の折れ線座標 [m]。lift は全体の鉛直オフセット。"""
    L = p.leg_link
    r = math.hypot(p.ankle_height, p.toe_length)
    beta0 = math.atan2(p.ankle_height, p.toe_length)
    phi = beta - beta0  # 足の回転角（踵が上がる向きが正）
    toe = (0.0, 0.0)
    sole = (-p.toe_length * math.cos(phi), p.toe_length * math.sin(phi))
    ankle = (-r * math.cos(beta), r * math.sin(beta))
    d = 2 * L * math.cos(kappa / 2)
    hip = (ankle[0], ankle[1] + d)
    knee = (ankle[0] + L * math.sin(kappa / 2), ankle[1] + d / 2)
    return [(x, y + lift) for x, y in (toe, sole, ankle, knee, hip)]


def build_frames(voltage: float, p: Params, release: str) -> tuple[list[dict], object]:
    trace: list = []
    res = simulate(voltage, p, release, trace=trace)
    step = max(1, round(FRAME_DT / p.dt))
    stance = trace[::step] + [trace[-1]]
    y_start = trace[0][3]
    frames = []
    for _ in range(HOLD_FRAMES):
        t, kappa, beta, y, _v = trace[0]
        frames.append(dict(t=0.0, kappa=kappa, beta=beta, lift=0.0, phase="crouch", rise=0.0))
    for t, kappa, beta, y, _v in stance:
        frames.append(dict(t=t, kappa=kappa, beta=beta, lift=0.0, phase="push-off", rise=0.0))
    _t_lo, kappa_lo, beta_lo, _y_lo, _ = trace[-1]
    v_lo = res.takeoff_velocity
    air_time = res.air_time
    n_flight = int(air_time / FRAME_DT)
    for i in range(1, n_flight + 1):
        tf = i * FRAME_DT
        lift = max(v_lo * tf - 0.5 * G * tf * tf, 0.0)
        frames.append(dict(t=res.push_time + tf, kappa=kappa_lo, beta=beta_lo, lift=lift, phase="flight", rise=lift))
    for _ in range(HOLD_FRAMES):
        frames.append(dict(t=res.push_time + air_time, kappa=kappa_lo, beta=beta_lo, lift=0.0, phase="landed", rise=0.0))
    return frames, (res, y_start)


def render(voltage: float, key: str, title: str, p: Params, release: str, out_dir: Path) -> Path:
    frames, (res, y_start) = build_frames(voltage, p, release)
    mm = 1e3
    fig, ax = plt.subplots(figsize=(4.2, 5.0), dpi=90)
    ax.set_xlim(-70, 50)
    ax.set_ylim(-8, 185)  # 上部は情報欄（頂点の破線と重ならないように）
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.axhline(0, color="0.3", lw=2)
    hip_lo = leg_points(frames[-1]["kappa"], frames[-1]["beta"], p)[-1]
    ax.axhline((hip_lo[1] + res.flight_rise) * mm, color="tab:red", ls="--", lw=0.8)
    ax.axhline(y_start * mm, color="0.6", ls=":", lw=0.8)
    (leg_line,) = ax.plot([], [], "-o", color="tab:blue", lw=2.5, ms=4)
    (trail,) = ax.plot([], [], "-", color="tab:orange", lw=1)
    (mass,) = ax.plot([], [], "o", color="tab:orange", ms=14)
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=7, family="monospace")
    ax.set_title(f"{title}\nVin = {voltage:.1f} V", fontsize=8)
    trail_x: list[float] = []
    trail_y: list[float] = []

    def update(i: int):
        f = frames[i]
        pts = leg_points(f["kappa"], f["beta"], p, f["lift"])
        xs = [x * mm for x, _ in pts]
        ys = [y * mm for _, y in pts]
        leg_line.set_data(xs, ys)
        mass.set_data([xs[-1]], [ys[-1]])
        if f["phase"] in ("push-off", "flight"):
            trail_x.append(xs[-1])
            trail_y.append(ys[-1])
        trail.set_data(trail_x, trail_y)
        info.set_text(
            f"t = {f['t'] * 1e3:6.1f} ms  [{f['phase']}]\n"
            f"rise now = {f['rise'] * mm:5.1f} mm\n"
            f"push {res.push_time * 1e3:.1f} ms, v_lo {res.takeoff_velocity:.3f} m/s\n"
            f"rise v^2/2g {res.flight_rise * mm:.1f} mm, air {res.air_time * 1e3:.1f} ms\n"
            f"red dashed: COM apex, grey dotted: crouch\n"
            f"slow motion x{1 / (FRAME_DT * FPS):.0f}"
        )
        return leg_line, mass, trail, info

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    out = out_dir / f"hop2d_{key}_{voltage:.1f}V.gif"
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voltages", type=float, nargs="+", default=[6.5, 7.4, 8.2])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "gifs")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key, title, p, release in CASES:
        for v in args.voltages:
            print(render(v, key, title, p, release, args.out_dir))


if __name__ == "__main__":
    main()
