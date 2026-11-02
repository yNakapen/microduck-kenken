# microduck にケンケンさせてみた（sim only）

Physical AI 応用1講座 最終課題のコード一式です。
Pollen Robotics の二足ロボット microduck に、強化学習で片足立ちとケンケン（片足の連続ホップ）をさせることに挑戦しました。

**すべてシミュレーションのみ（sim only）です。実機では一切動かしていません。**

- シミュレータ：Isaac Lab 3.0（Newton / MuJoCo Warp、BAM サーボモデル）
- 学習：PPO（rsl_rl）、RTX 3090 1 枚
- 先行研究：[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) の `flamingo` ブランチ（片足立ち）

## 構成

| ディレクトリ | 内容 |
|---|---|
| `isaaclab_patches/` | Isaac Lab（コミット `210ee799c`）に当てるパッチ一式（`git format-patch`） |
| `isaaclab_files/` | 上のパッチで追加・変更したファイル（閲覧用） |
| `hop_analysis/` | xz 2 次元・折れ線脚モデルによる真上ジャンプの見積もり（`hop2d.py`）、テスト、GIF |
| `video/` | 提出動画を組み立てるスクリプト |

主なファイル（`isaaclab_files/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/`）：

| ファイル | 内容 |
|---|---|
| `flamingo_env_cfg.py` | 片足立ち `IsaacContrib-Flamingo-Flat-MicroDuck`（microduck_rl の段階 1 の移植） |
| `flamingo_hop_env_cfg.py` | ケンケン `IsaacContrib-FlamingoHop-Flat-MicroDuck`（片足立ちモデルから追加学習） |
| `flamingo_cycle_env_cfg.py` | 両足⇄片足の Cycle 版（移植のみ。今回は学習を途中で中止） |
| `mdp/flamingo.py`, `mdp/flamingo_hop.py`, `mdp/flamingo_cycle.py` | 報酬・指令・リセット |
| `flamingo_checks.py`, `flamingo_metrics.py`, `flamingo_eval.py` | 静的な姿勢チェック、評価指標、評価用のロールアウト |

スクリプト（`isaaclab_files/scripts/reinforcement_learning/microduck/`）：`check_flamingo_pose.py`、`eval_flamingo.py`、`eval_flamingo_hop.py`、`record_flamingo.py`（Newton の Warp レイトレーシングで録画。OpenGL 不要）。

## 再現手順

```bash
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab
git checkout 210ee799c
git am /path/to/microduck-kenken/isaaclab_patches/*.patch
# Isaac Lab の uv 環境を用意したうえで
# 1) 片足立ち
uv run python scripts/reinforcement_learning/train.py --rl_library rsl_rl \
    --task IsaacContrib-Flamingo-Flat-MicroDuck --num_envs 4096
# 2) ケンケン（片足立ちのチェックポイントから追加学習）
uv run python scripts/reinforcement_learning/train.py --rl_library rsl_rl \
    --task IsaacContrib-FlamingoHop-Flat-MicroDuck --num_envs 4096 \
    --checkpoint logs/rsl_rl/microduck_flamingo_flat/<run>/model_2999.pt
# 評価と録画
uv run python scripts/reinforcement_learning/microduck/eval_flamingo_hop.py --policy <exported>/policy.pt
uv run --with imageio --with imageio-ffmpeg python scripts/reinforcement_learning/microduck/record_flamingo.py \
    --task IsaacContrib-FlamingoHop-Flat-MicroDuck --hop --policy <exported>/policy.pt
```

## 結果（概要）

1. **跳べるかの確認**（`hop_analysis/`）：7.4 V で最適なタイミングで蹴ると、滞空 約 105 ms、重心の浮き上がり 約 14 mm（6.5〜8.2 V で 84〜124 ms）。小さいながらケンケンは物理的に可能。
2. **片足立ち**：microduck_rl の報酬・カリキュラム・出現を合わせて移植し、microduck_rl 自身が後で入れた修正（上げた足の目標高さ、押しと action_rate の上限）を反映。押し 0.15 m/s で成功率 約 70%（5 s 以上の片足支持、3 seed 平均 70.3%）。
3. **ケンケン（報酬ハッキング）**：立ち足の滞空時間に応じた報酬で、跳躍は出たが、上げた足で床を蹴って跳ぶ報酬ハッキングが発生（離陸直前 60 ms に上げた足が接地していた割合 93%）。
4. **ケンケン（報酬を改良）**：上げた足が離陸の 0.1 s 前から浮いている跳躍だけに報酬、立ち続けると時間に比例して増える罰（上限なし）、ケンケン中は静止を促す報酬を外す、の 3 点で再学習。3 s で約 23 回ケンケンし（128 エピソード、最小 19 回）、押し 0.15 m/s でも転倒 0%、その後 2.5 s 片足で静止。上げた足の接地が 1 エピソードに 1 回ほど残るのが課題。

## ライセンス・帰属

- `isaaclab_patches/` と `isaaclab_files/` は Isaac Lab への変更で、Isaac Lab と同じ BSD-3-Clause（各ファイルの SPDX ヘッダのとおり）。
- 片足立ちの姿勢・報酬の重み・カリキュラムの値は microduck_rl（Apache-2.0）の `flamingo` ブランチ（`4757d6ae`）から移植しています。
- `hop_analysis/` と `video/` は Apache-2.0。
- ロボットのメッシュ（STL、CC BY-NC-SA 4.0）などのアセットは含みません。Isaac Lab の公式アセット（USD）を実行時に取得します。
- 動画の字幕フォント（IPAex ゴシック）は含みません。
