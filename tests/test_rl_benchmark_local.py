#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""uav_rl_benchmark 本地测试（不需要 ROS / 仿真器 / torch）.

覆盖：
    A. 参数解析与配置加载（_parse_arch / _parse_seeds / load_config）
    B. 环境定位（能根据源码布局 / devel 布局找到 uav_ppo_nav 的 env.py）
    C. 算法工厂的纯函数（arch_str）与算法名校验
    D. **评估指标计算**：用桩策略驱动真实 NavEnv，验证成功率/超时率/路径效率/
       动作平滑度算得对不对（这是本包最核心的逻辑，不需要训练框架即可验证）
    E. 报告汇总（有 pandas 才算；缺库则跳过）

用法: python3 tests/test_rl_benchmark_local.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

PASS, FAIL = [], []


def check(cond, label):
    (PASS if cond else FAIL).append(label)
    print("  [%s] %s" % ("PASS" if cond else "FAIL", label))


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRIPTS = os.path.join(ROOT, "src", "air", "uav_rl_benchmark", "scripts")
CONFIG = os.path.join(ROOT, "src", "air", "uav_rl_benchmark", "config", "benchmark.yaml")
sys.path.insert(0, SCRIPTS)

import algos as A            # noqa: E402
import benchmark as B        # noqa: E402
import main as M             # noqa: E402
import nav_env as NE         # noqa: E402


# ----------------------------------------------------------------- 桩策略
class HoverPolicy(object):
    """永远悬停：到不了目标，回合必然靠 max_steps 截断 -> 全是"超时"."""

    def predict(self, obs, deterministic=True):
        return np.zeros(4, dtype=np.float32), None


class GoalSeekPolicy(object):
    """最简单的比例控制器（直接读观测里的世界系目标偏移，不依赖任何网络）.

    观测第 16~18 维就是 goal_rel / GOAL_SCALE，所以可以反解出世界系目标方向。
    """

    def predict(self, obs, deterministic=True):
        g = np.asarray(obs, dtype=np.float64).reshape(-1)[16:19] * 8.0
        a = np.clip(g / 3.0, -1.0, 1.0)
        return np.array([a[0], a[1], a[2], 0.0], dtype=np.float32), None


# ================================================================= A. 参数解析
def test_parse():
    print("== A. 参数解析 / 配置加载 ==")
    check(M._parse_arch("64,64") == (64, 64), "_parse_arch('64,64')")
    check(M._parse_arch("64x64") == (64, 64), "_parse_arch('64x64')（x 分隔也支持）")
    check(M._parse_arch([256, 256]) == (256, 256), "_parse_arch(list)")
    check(M._parse_seeds("0,1,2") == (0, 1, 2), "_parse_seeds('0,1,2')")
    check(M._parse_seeds([3]) == (3,), "_parse_seeds(list)")

    cfg = M.load_config(CONFIG)
    if cfg:
        check("algos" in cfg.get("matrix", {}), "配置文件含 matrix.algos")
        check(cfg["matrix"]["steps"] == 100000, "配置文件 matrix.steps = 100000")
        check(cfg["eval"]["episodes"] == 100, "配置文件 eval.episodes = 100")
        check(len(cfg["matrix"]["archs"]) == 3, "配置文件列出 3 种网络结构")
    else:
        print("  [跳过] 未安装 PyYAML 或配置缺失")


# ================================================================= B. 环境定位
def test_nav_env():
    print("== B. 环境定位（复用 uav_ppo_nav 的 NavEnv）==")
    d = NE.find_nav_scripts()
    check(os.path.isfile(os.path.join(d, "env.py")), "定位到含 env.py 的目录")
    check(os.path.isfile(os.path.join(d, "policy.py")), "同目录含 policy.py")
    NavEnv, policy = NE.load_nav_env()
    env = NavEnv(num_obstacles=4, seed=0)
    obs, _ = env.reset()
    check(obs.shape == (22,), "复用到的 NavEnv 观测维度 22")
    check(policy.OBS_DIM == 22 and policy.ACT_DIM == 4, "policy 接口常量一致")
    env.close()


# ================================================================= C. 算法工厂
def test_algos():
    print("== C. 算法工厂（纯函数部分）==")
    check(A.arch_str([64, 64]) == "64x64", "arch_str([64,64]) -> '64x64'")
    check(A.arch_str((256, 256)) == "256x256", "arch_str((256,256)) -> '256x256'")
    check(A.arch_str([64, 64, 64]) == "64x64x64", "arch_str([64,64,64]) -> '64x64x64'")
    check(set(A.ALGOS) == {"ppo", "sac", "td3"}, "ALGOS = ppo / sac / td3")
    try:
        A.make_model("dqn", None)
        check(False, "未知算法应抛 ValueError")
    except ValueError:
        check(True, "未知算法抛 ValueError")
    except ImportError:
        check(True, "无 sb3 时抛 ImportError（可接受）")


# ================================================================= D. 评估指标
def test_evaluate_metrics():
    print("== D. 评估指标计算（桩策略 + 真实 NavEnv）==")
    NavEnv, _ = NE.load_nav_env()

    # D1. 悬停策略：必然超时
    m = B.evaluate(HoverPolicy(), NavEnv, episodes=5, obstacles=0, seed=7)
    check(m["success_rate"] == 0.0, "悬停策略成功率 = 0")
    check(abs(m["timeout_rate"] - 1.0) < 1e-9, "悬停策略超时率 = 100%")
    check(abs(m["success_rate"] + m["crash_rate"] + m["timeout_rate"] - 1.0) < 1e-9,
          "三类结果占比之和 = 1")
    check(m["mean_steps"] > 100, "悬停策略平均步数很大（%0.0f 步）" % m["mean_steps"])
    check(m["action_smoothness"] < 1e-9, "悬停策略动作恒定 -> 平滑度 = 0")
    check(m["path_efficiency"] == 0.0,
          "悬停策略路径效率 = 0（原地不动不能算满分）")

    # D2. 比例控制策略：无障碍环境下应能到目标
    m2 = B.evaluate(GoalSeekPolicy(), NavEnv, episodes=30, obstacles=0, seed=7)
    check(m2["success_rate"] > 0.5,
          "比例控制策略成功率 > 50%%（实测 %.0f%%）" % (m2["success_rate"] * 100))
    check(m2["success_rate"] > m["success_rate"], "比例控制优于悬停（指标有区分度）")
    check(m2["action_smoothness"] >= 0.0, "动作平滑度非负")
    check(0.0 < m2["path_efficiency"] <= 1.0,
          "比例控制路径效率落在 (0,1] 内（实测 %.2f）" % m2["path_efficiency"])
    print("      悬停:  成功率 %.0f%%  步数 %.0f  效率 %.2f  平滑 %.3f"
          % (m["success_rate"] * 100, m["mean_steps"], m["path_efficiency"],
             m["action_smoothness"]))
    print("      比例控: 成功率 %.0f%%  步数 %.0f  效率 %.2f  平滑 %.3f"
          % (m2["success_rate"] * 100, m2["mean_steps"], m2["path_efficiency"],
             m2["action_smoothness"]))

    # D3. 指标字段齐全
    need = set(B.RESULT_FIELDS) - {"algo", "arch", "seed", "steps", "obstacles",
                                   "eval_episodes", "train_time_s", "train_fps"}
    check(need <= set(m2.keys()), "评估返回包含全部指标字段：%s" % sorted(need))


# ================================================================= E. 报告汇总
def test_report(tmp_dir):
    print("== E. 报告汇总与绘图 ==")
    try:
        import pandas  # noqa: F401,WPS433
    except Exception:                                       # noqa: BLE001
        print("  [跳过] 缺少 pandas")
        return
    # 造一份最小的 results.csv
    import csv
    rows = []
    for algo in ("ppo", "sac"):
        for seed in (0, 1):
            r = {k: 0.0 for k in B.RESULT_FIELDS}
            r.update({"algo": algo, "arch": "64x64", "seed": seed, "steps": 1000,
                      "obstacles": 16, "eval_episodes": 5,
                      "train_time_s": 10.0 + seed, "train_fps": 100.0,
                      "success_rate": 0.5 if algo == "ppo" else 0.25,
                      "crash_rate": 0.2, "timeout_rate": 0.3,
                      "mean_steps": 40.0, "mean_reward": 1.0,
                      "mean_path_len": 5.0, "path_efficiency": 0.7,
                      "action_smoothness": 0.1})
            rows.append(r)
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=B.RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    md = B.__dict__  # noqa: F841  (占位，避免 linter 抱怨)
    import report as R  # noqa: E402,WPS433
    path = R.build_report(tmp_dir, make_plots=False)
    check(path is not None and os.path.isfile(path), "生成了 report.md")
    txt = open(path, encoding="utf-8").read()
    low = txt.lower()
    check("ppo" in low and "sac" in low, "报告表格含两种算法")
    check("成功率" in txt, "报告含成功率列")
    check("结论要点" in txt, "报告含结论要点小节")


def main():
    tmp_dir = os.path.join(HERE, "_tmp_bench_report")
    try:
        test_parse()
        test_nav_env()
        test_algos()
        test_evaluate_metrics()
        test_report(tmp_dir)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n==================== 结果 ====================")
    print("通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
    if FAIL:
        for f in FAIL:
            print("  FAIL: %s" % f)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
