# BENI velocity-flat for MotrixLab

> [English>>](README-en.md)

A BENI two-wheel-legged robot velocity-tracking task for MotrixLab,
ported from Mondo-Robotics/open-beni (mjlab, MIT) to the MotrixLab
direct-workflow (FastSAC).

- 上游项目：[https://github.com/Mondo-Robotics/open-beni（MIT）](https://github.com/Mondo-Robotics/open-beni)
- 目标框架：[https://github.com/Motphys/MotrixLab（Apache-2.0）](https://github.com/Motphys/MotrixLab)
- 验证环境：MotrixLab 0.4.0b0 与 main @ 5bd0054、Python 3.10、CUDA 12.x、
  单张消费级 GPU（8 GB，与其他 GPU 任务共存）



https://github.com/user-attachments/assets/a136308f-68c9-4f77-888c-c8308d992022



## 目录内容

```
overlay/          按 MotrixLab 相对路径布局的全部文件，复制到 checkout 根目录
                  （或直接运行 install.py）。
  motrix_envs/src/motrix_envs/robot/assets/beni/
                    MJCF + 烘焙 mesh + 贴图（MIT，来自 open-beni）
  motrix_envs/src/motrix_envs/robot/beni.py
                    RobotCfg（7 个部署执行器，官方默认姿态）
  motrix_envs/src/motrix_envs/locomotion/beni/
                    cfg.py（奖励/终止配置）
                    walk.py（DirectEnv：观测 28/31 维、动作 7 维、膝弹簧）
                    beni_task.py（registry 注册，任务名 "beni-velocity-flat"）
  configs/task/beni-velocity-flat/motrix.fastsac.yaml
  runs/beni-velocity-flat/.../      预训练 run（含 metadata；最佳点
                    model_0008000.pt）
  scripts/_beni_*.py                评估 / 录像 / 绘图 / 烘焙辅助脚本
install.py        把 overlay/ 安装进 MotrixLab checkout，并自动补丁两个
                  注册用 __init__.py（幂等，可重复执行）。
verify_install.py 安装后自检（环境构建 + 50 步随机动作）。
LICENSE           MIT（本移植），含上游署名声明。
NOTICE            溯源与署名详情。
```

## 关于预训练 runs/ 目录

`overlay/runs/` 自带一份训练好的 run，安装后可以立即评估。但 checkpoint
每个约 16 MB，**不应提交进 git 仓库**。发布本移植时：

- 把 `runs/` 加入 `.gitignore`（上游 MotrixLab 自 5bd0054 起也已忽略
  runs/ 目录）；
- 将 `runs/beni-velocity-flat/` 打包（zip/tar.gz）挂到 GitHub Release
  资产上，用户下载后解压到 checkout 根目录即可。
  `verify_install.py` 会自动识别任意时间戳的 run 目录；评估/录像脚本用
  `--run <RUN_DIR>` 指定。

## 安装（无需克隆 open-beni）

1. 前提：一个可用的 MotrixLab checkout，且虚拟环境已按其 README 构建。

2. 在本目录下执行：

       python install.py --motrixlab /path/to/MotrixLab

   该脚本把 overlay/* 拷入 checkout，并向两个注册文件追加 import 行。
   幂等设计：重复执行安全，不会覆盖非本移植的文件。

3. 验证：

       # Linux / macOS
       /path/to/MotrixLab/.venv/bin/python verify_install.py --motrixlab /path/to/MotrixLab

       # Windows
       F:\path\to\MotrixLab\.venv\Scripts\python.exe verify_install.py --motrixlab F:\path\to\MotrixLab

## 从零训练

在 MotrixLab checkout 根目录：

    # Linux / macOS
    .venv/bin/python scripts/train.py task=beni-velocity-flat/motrix.fastsac

    # Windows：上游 fastsac 需要一个小的 sched 兼容垫片（见下方环境说明）
    set TORCH_COMPILE_DISABLE=1
    set PYTHONPATH=overlay\scripts;%PYTHONPATH%
    .venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'overlay/scripts'); import _win_sched_compat; exec(open('scripts/train.py',encoding='utf-8').read())" -- --task=beni-velocity-flat/motrix.fastsac

默认配置：2048 个并行环境、10000 次迭代、每 1000 迭代存一次 checkpoint。
实测（RTX 2080 Ti 级单卡，learner 约 300 MB 显存、物理在 CPU 上、同机另有
GPU 任务并行）：约 4200 env-steps/s，全程约 1 小时 30 分。episode 长度
曲线通常在 4000 迭代左右打满 20 秒上限。多卡/有其他 GPU 任务时用
`CUDA_VISIBLE_DEVICES` 指定空闲卡。

## 评估预训练模型（固定指令电池）

在 MotrixLab checkout 根目录（`RUN_DIR` 为 runs/beni-velocity-flat/
motrix/torch/fastsac/ 下的时间戳目录名）：

    .venv/bin/python scripts/_beni_eval.py \
        --run <RUN_DIR> --ckpt model_0008000.pt --steps 600

    # Windows: .venv\Scripts\python.exe scripts\_beni_eval.py ...

脚本会跑 8 条固定指令（站立 / 前进 / 后退 / 原地转 / 组合），输出存活率、
线速度/角速度跟踪误差、基座高度与倾角。

录制跟随相机演示视频：

    .venv/bin/python scripts/_beni_record.py \
        --run <RUN_DIR> --ckpt model_0008000.pt \
        --out deliverables/demo.mp4

## 任务规格

| 项目 | 内容 |
|---|---|
| 机器人 | BENI，7 个部署执行器（L1 R1 L2 R2 LW RW head_yaw）：腿=位置目标（默认 + 0.25·a），轮=速度目标（20·a rad/s），头=位置目标（0.5·a） |
| 观测 | actor 28 维（角速度 3 + 重力投影 3 + 指令 3 + 关节位置 5 + 关节速度 7 + 上一动作 7）；critic = actor + 基座线速度（31 维） |
| 指令 | vx ∈ [-1,1] m/s，wz ∈ [-2,2] rad/s，25% 概率站立，5 秒重采样 |
| 回合 | 20 秒；仿真 dt 0.002，控制 dt 0.01 |
| 终止 | 非法体节触地；轮速超 350 rad/s |
| 奖励 | exp 核线速度/偏航跟踪（σ=0.5 + 锐 σ=√0.02）、默认姿态（权重 -8）、腿对称、倾角、垂直速度、动作率、接触惩罚、存活奖励、摔倒罚 -10。所有正向项以"直立且无非法接触"门控 |

被动膝弹簧（移植自上游 BeniMotor，按站立区间线性化）：
`τ = 0.3636 · (θ + 0.3848)`，在 walk.py 中以位置执行器 ctrl 偏移注入。
注意：MotrixSim 会**忽略** joint 的 `stiffness/springref` XML 属性，
弹簧必须留在代码里。

## 参考结果（单张消费级 GPU，seed 1，model_0008000.pt）

- 8/8 指令电池 600 步全程 100% 存活
- 线速度跟踪误差 0.025–0.106 m/s，基座高度 0.134–0.137 m
- 站立关节角贴近官方默认（L1/R1 0.37/0.46、L2/R2 0.60/0.55，官方为
  0.3/0.5）
- 已知怪癖：前进+左转的偏航跟踪较差（2.03 rad/s 误差），前进+右转正常
  （0.13）——见"改进方向"

## 改进方向

- 针对 fwd+左转怪癖：加强偏航跟踪奖励或对称数据增强
- 移植精确的非线性膝弹簧（每物理子步计算）替代站立区间线性化
- 移植上游摔倒重置课程（仰卧/俯卧/侧躺重置）

## 已知环境说明

- 已在 MotrixLab main @ 5bd0054（2026-09-25）与 0.4.0b0 时期 checkout 上
  验证。walk.py 从 `sim.model` 导入 ActuatorType，并带旧版 `sim.backend`
  回退，两代代码都能工作。
- 上游提交 c58cf54（NUMA 绑定）引入的 `os.sched_getaffinity` 调用没有
  win32 守卫：Windows 上跑 fastsac 训练目前需要包内附带的
  `overlay/scripts/_win_sched_compat.py` 小垫片（在导入 motrix_rl 之前
  导入），或改在 Linux 上训练。这个守卫值得给上游提 PR。
- MotrixSim 会忽略 joint `stiffness/springref` XML 属性（膝弹簧因此在
  walk.py 代码注入），也不支持多层 `<layer>` PBR 材质（RGB 贴图已改用
  经典属性写法）。

## 许可证

本移植采用 MIT。BENI 资产与奖励设计源自
[Mondo-Robotics/open-beni（MIT）](https://github.com/Mondo-Robotics/open-beni)。[MotrixLab](https://github.com/Motphys/MotrixLab) 本身是 Apache-2.0；overlay
文件只使用了 MotrixLab 为新增机器人/任务提供的扩展点，除两行注册
import 外不修改 MotrixLab 源码。
