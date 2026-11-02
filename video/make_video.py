"""YouTube 提出用の動画を組み立てる（タイトル → xz 2 次元の解析 → 片足立ち → 報酬ハッキング → 続く）。

各区間は 1280x720・25 fps に揃える。シミュレーションの録画（50 fps）は 1 フレームおきに使って等倍、
スロー区間は全フレームを使って 0.5 倍速にする。字幕は IPAex ゴシックで描く。
"""

import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 25
BG = (18, 22, 30)
FG = (240, 240, 240)
ACCENT = (255, 196, 0)


def font(path, size):
    return ImageFont.truetype(path, size)


def text_block(draw, lines, fnt, top, color=FG, center=True, spacing=12):
    y = top
    for line in lines:
        w = draw.textlength(line, font=fnt)
        x = (W - w) / 2 if center else 60
        draw.text((x, y), line, font=fnt, fill=color)
        y += fnt.size + spacing
    return y


def card(args, title, lines, seconds, title_color=ACCENT):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = text_block(d, [title], font(args.font, 56), 200, color=title_color)
    text_block(d, lines, font(args.font, 30), y + 40)
    frame = np.asarray(img)
    return [frame] * int(seconds * FPS)


def compose(args, frame, header, notes):
    """フレームを左に、見出しと説明を上と右に置く。"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 24), header, font=font(args.font, 34), fill=ACCENT)
    src = Image.fromarray(frame)
    area_w, area_h = 760, 600
    scale = min(area_w / src.width, area_h / src.height)
    src = src.resize((int(src.width * scale), int(src.height * scale)), Image.LANCZOS)
    img.paste(src, (40 + (area_w - src.width) // 2, 90 + (area_h - src.height) // 2))
    fnt = font(args.font, 24)
    y = 110
    for line in notes:
        d.text((830, y), line, font=fnt, fill=FG)
        y += fnt.size + 14
    return np.asarray(img)


def gif_frames(path, seconds):
    im = Image.open(path)
    frames = []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append(np.asarray(im.convert("RGB")))
    n = int(seconds * FPS)
    return [frames[int(k * len(frames) / n)] for k in range(n)]


def video_frames(path, stride=2, start=0.0, end=None, src_fps=50):
    frames = [f for f in imageio.get_reader(path)]
    a = int(start * src_fps)
    b = len(frames) if end is None else int(end * src_fps)
    return frames[a:b:stride]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--font", required=True)
    p.add_argument("--gif", required=True)
    p.add_argument("--stage1", required=True)
    p.add_argument("--hack", required=True)
    p.add_argument("--hop", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = []
    out += card(
        args,
        "microduck にケンケンさせてみた",
        [
            "Physical AI 応用1講座 最終課題",
            "Pollen Robotics の二足ロボット microduck（約 0.74 kg）",
            "Isaac Lab + Newton / MuJoCo Warp（シミュレーションのみ・実機なし）",
        ],
        6,
    )

    notes = [
        "まず、そもそも跳べるかを確認",
        "",
        "・xz 2 次元・折れ線脚モデル",
        "・サーボは BAM で同定した XL330 の",
        "  DC モータモデル（摩擦あり）",
        "・7.4 V で最適なタイミングで蹴ると",
        "  滞空 約 105 ms、重心が 約 14 mm 浮く",
        "  （6.5〜8.2 V で 84〜124 ms）",
        "",
        "→ 小さいながらケンケンは物理的に可能",
    ]
    out += [compose(args, f, "1. xz 2 次元モデルでジャンプできるか確認", notes) for f in gif_frames(args.gif, 12)]

    notes = [
        "先行研究 microduck_rl（flamingo）の",
        "片足立ちを、報酬・カリキュラム・出現",
        "まで合わせて Isaac Lab に移植し学習",
        "",
        "・PPO、4096 環境、3000 iteration",
        "・押し 0.15 m/s",
        "・ドメインランダム化あり",
        "・成功率 約 70%",
        "  （5 s 以上の片足支持）",
        "",
        "この動画：押しあり・等倍・成功例",
    ]
    out += [compose(args, f, "2. microduck_rl をトレースして片足立ちを学習", notes) for f in video_frames(args.stage1)]

    notes = [
        "片足立ちモデルから追加学習",
        "・立ち足の滞空時間に応じた報酬",
        "・上げた足の接地は罰",
        "",
        "結果：上げた足で床を蹴って跳ぶ",
        "  → 報酬ハッキング",
        "  この例：上げた足の接地 5 回",
        "          跳躍 3 回",
        "",
        "0〜2 s 静止 → 2〜5 s ケンケン → 静止",
    ]
    header = "3. 跳躍時間の報酬でケンケン → 報酬ハッキング"
    out += [compose(args, f, header, notes + ["", "等倍"]) for f in video_frames(args.hack)]
    out += [
        compose(args, f, header, notes + ["", "ケンケン区間を 0.5 倍速で再生"])
        for f in video_frames(args.hack, stride=1, start=2.0, end=5.0)
    ]

    notes = [
        "報酬を改良して再学習",
        "・上げた足が離陸 0.1 s 前から浮いている",
        "  跳躍だけに報酬",
        "・立ち続けると時間に比例して増える罰",
        "・ケンケン中は静止を促す報酬を外す",
        "",
        "結果：倒れ込んだまま足を動かして",
        "  「跳躍」を稼ぐ、2 つ目の報酬ハッキング",
        "・胴体の傾きの終了条件（70°）に",
        "  掛からない倒れ方だった",
        "→ 次は倒れ込みを終了条件・罰に入れる",
    ]
    header = "4. 報酬を改良 → 2 つ目の報酬ハッキング"
    out += [compose(args, f, header, notes + ["", "等倍"]) for f in video_frames(args.hop)]
    out += [
        compose(args, f, header, notes + ["", "ケンケン区間を 0.5 倍速で再生"])
        for f in video_frames(args.hop, stride=1, start=2.0, end=5.0)
    ]

    out += card(
        args,
        "まとめ",
        [
            "物理モデルで可否を確認 → 先行研究をトレースして片足立ち",
            "→ 報酬ハッキングを分析して報酬を改良（2 回）→ ケンケンは未達",
            "すべてシミュレーションのみ（Isaac Lab + Newton / MuJoCo Warp）",
            "※ 0.5 倍速の区間は画面内に表記",
        ],
        9,
        title_color=FG,
    )

    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    for f in out:
        writer.append_data(f)
    writer.close()
    print(f"frames {len(out)} duration {len(out) / FPS:.1f} s")


if __name__ == "__main__":
    main()
