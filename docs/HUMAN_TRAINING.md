# Casbot02 独立 human encoder 训练

新增的真实 motion＋human 联合训练入口见 [JOINT_HUMAN_TRAINING.md](JOINT_HUMAN_TRAINING.md)。下文仍是冻结 teacher 的独立后训练，二者入口、日志与 checkpoint 分开。

这个任务参考 [MorphMimic 当前 human copy 接口](https://github.com/lzyang2000/morphmimic/blob/main/src/morphmimic/train/imagination.py)，训练一个新的人体目标编码器，复用现有 motion 策略的解码器。原来的 `sm-train`、训练配置、观测、随机化、奖励、模型文件和普通部署配置均不修改。

## 保留的接口与新增的接口

- 本体观测保持 **930 = 10×93**，其中每帧仍有 6D 参考骨盆相对姿态；历史采用原来的 term-major 顺序和固定尺度。
- motion 目标仍为 **690 = 10×69**，原动作编码器、解码器和世界模型全部冻结。
- 新增 human 目标为 **2254**：10 帧×24 关节×(位置3＋rot6d6)=2160，加 24 个关节 mask、10 个时间 mask，再加 10×6 的人体朝向反馈。它是新编码器的目标输入，不是给原本体观测追加维度。
- 朝向反馈沿用 MorphMimic v10：人体 Spine3（五点模式为 Pelvis）相对机器人 torso/腰部坐标系的朝向。用 IMU 姿态和腰关节即可计算机器人 torso 朝向，不需要实际根部位置。它与原 930 维观测中的骨盆相对姿态是两个独立输入项。
- human encoder 输出 32 维 latent，接原解码器，仍输出 27 维动作。

默认采用 MorphMimic README 推荐的 `copy` 路线：人体编码器在其自身动作驱动的世界模型状态上，匹配 motion teacher 的动作和 latent。两项损失分别做运行尺度归一化，默认权重均为 1。全身/五点各约一半样本，固定 10 帧预览，不启用人体 drift/jitter。

本项目的独立版本与 MorphMimic 的联合训练有明确区别：**它是在现有 checkpoint 上追加 human 编码器的后训练任务**，不继续更新 motion/WM，不采集新的仿真 replay。起点取配对参考状态，然后由冻结的 motion teacher 预热 10 步形成一致的观测历史，再做 human student 的 imagined rollout。`copy` 模式每步断开动态梯度，没有 human critic，也不优化回报目标；不能把它描述为完整移植了 `return` 或 direct-human 训练。这样原策略不会因 human 训练而漂移，后续可另加独立的 replay/在线蒸馏任务。

## 1. 下载人体数据

已有 Casbot02 CSV/NPZ 只有机器人轨迹，不能恢复真实人体骨架。必须使用**同一动作、逐帧对应的 SOMA Uniform BVH**；不使用 `soma_proportional`，也不把机器人 FK 当成人体标签。

官方：[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)。2026-11-14 文件页将人体数据作为约 45.2 GB 的 `soma_uniform.tar.gz` 提供，没有逐条 BVH 下载入口。可以先只训练 20 条配对动作，但需先下载这个人体压缩包。已有机器人数据时，不必再下载 G1 压缩包或比例人体包。

先在网页登录并接受数据条款。本机代理端口 7897 已用 HTTP 方式验证可访问 Hugging Face；`socks://` 会被当前客户端拒绝。安装好的 `hf` 为 2.x，多文件直接列出文件名，不用旧式多参数 `--include`。

```bash
cd /home/casbot/Desktop/SuperMimic-Reproduced
bash scripts/human/download_seed_human.sh login
# 浏览器打开 hf.co/oauth/device，输入终端验证码并确认；等待终端登录成功。
bash scripts/human/download_seed_human.sh download
```

下载目录默认 `/media/casbot/Linux_Drive/supermimic_ws/bones_seed_human`。脚本不会自动登录或接受条款，不会打印/读取令牌。它通过当前进程的环境设置代理，不改系统设置或原训练环境。

可用 `SM_HF_PROXY`、`SM_SEED_DIR` 环境变量更换代理/目标目录。`plan` 模式只检查下载计划：

```bash
bash scripts/human/download_seed_human.sh plan
```

遇到 `Waiting for authorization` 表示代理已通，正在等用户完成浏览器授权；不是下载报错。遇到 401/403 时应检查登录账号及该账号的数据集访问权限。

## 2. 先抽取已有机器人数据对应的 20 条动作

```bash
env -u PYTHONPATH .venv/bin/python scripts/human/extract_seed_subset.py \
  --archive /media/casbot/Linux_Drive/supermimic_ws/bones_seed_human/soma_uniform.tar.gz \
  --robot-dir data/npz/bones_seed_casbot02_3mm \
  --manifest data/npz/bones_seed_casbot02_3mm/manifest_filtered.csv \
  --output-dir data/raw/seed_human_bvh20 \
  --limit 20
```

只抽取已有 keep-list 中能匹配的 SEED 名称，不更改机器人数据。`--names name1 name2 ...` 可显式指定动作，`--limit 0` 处理全部所选名称。压缩 tar 必须顺序扫描，少量抽取也可能耗时；这不等于只下载少量字节。

确认 `extraction_report.json` 中没有缺失动作。不要把标准骨架的 T-pose 文件当作训练动作。

## 3. 转换人体 sidecar，并检查严格配对

```bash
env -u PYTHONPATH .venv/bin/python scripts/human/prepare_pairs.py \
  --robot-dir data/npz/bones_seed_casbot02_3mm \
  --manifest data/npz/bones_seed_casbot02_3mm/manifest_filtered.csv \
  --bvh-root data/raw/seed_human_bvh20 \
  --output-dir data/human/seed_uniform20
```

输出为新的 `human50/`、`pairs.csv`、`preparation_report.json`，不覆盖原 NPZ。转换保留 MorphMimic 的 24 个 SMPL slot、SOMA T-pose 局部轴去除、厘米→米、Y-up→Z-up、120→50 Hz 的线性位置插值与四元数 SLERP。

配对核对 50 Hz、关节顺序、有限值、单位四元数和帧数。短于 52 帧的片段不进入默认 32 步 imagined rollout。机器人和人体开头可能因之前的数据清洗而不一致，**不会自动裁短到共同长度**。不匹配的动作记录为 rejected；需根据已知裁剪记录在独立 `pairs.csv` 中填写 `robot_start_frame`、`human_start_frame`、`length`，再验证。原始源数据保持不变。

如果已有 MorphMimic human50，使用 `--human-root /path/to/human50` 代替 `--bvh-root`。只按唯一文件 stem 配对，重名歧义会拒绝。

## 4. 独立训练

默认 teacher 跟随当前已验证的部署模型，通过 checkpoint SHA256 与导出 bundle 核对；当前为 cpu7 / 96000。先小 batch 验证数据与资源：

```bash
env -u PYTHONPATH .venv/bin/python scripts/human/train_human.py \
  --pairs data/human/seed_uniform20/pairs.csv \
  --output-dir logs/human_casbot02_seed20 \
  --device cuda:0 --batch-size 256 --iterations 60000 \
  --masking mixed --cache-clips 0
```

可显式传 `--teacher-ckpt ... --bundle ...`，两者必须对应同一个已验证的导出。不要拿当前资产默认值替代旧模型训练用的 q0；此入口从匹配 bundle 读取 q0、模型和关节契约，不改电机参数。

网络与批次在 `--device` 上，原始配对轨迹保存在 CPU。`--cache-clips 0` 把全部选中配对保存在内存；大集合内存不足时用正数限制 CPU 缓存，代价是磁盘读入。采样会先按片段分组，避免一个大 batch 中反复解压同一个文件。本入口默认将 OMP/BLAS/MKL 与 Torch 线程限制为 1，仅作用于这个新进程。

按片段划分约 10% 验证集，镜像 `_M` 与其原动作放在同一侧，避免泄漏。日志中的 `action_mse`、`latent_mse`、`val_*` 是蒸馏误差，`imagined_survival` 是冻结世界模型的预测；**不是整段真仿真完成率或归一化回报**。

每 4000 次保存独立 `human_*.pt` 与 `human_latest.pt`；正常结束或 Ctrl+C 也保存。发生数值异常时不覆盖上次正常的 latest。断点恢复保存 human 网络、归一化、优化器、损失尺度、数据指纹与随机状态：

```bash
env -u PYTHONPATH .venv/bin/python scripts/human/train_human.py \
  --pairs data/human/seed_uniform20/pairs.csv \
  --output-dir logs/human_casbot02_seed20 \
  --device cuda:0 --batch-size 256 --iterations 60000 \
  --masking mixed --cache-clips 0 \
  --resume logs/human_casbot02_seed20/human_latest.pt
```

**不要把 human checkpoint 传给原来的 `sm-train --resume`。** 原 motion/WM checkpoint 与原部署模型都保持原样。

## 5. 导出 human actor（独立产物）

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/human/export_human.py \
  --checkpoint logs/human_casbot02_seed20/human_latest.pt \
  --output-dir outputs/human_casbot02_seed20_export
```

导出会比较 full/five 和不同 batch 下的 TorchScript/PyTorch 输出，生成独立 `human_actor.pt` 与 `manifest.json`，不会安装到 `rl_controller`。

这阶段接入的是 human **训练与导出**接口。现有 PICO 三点/GMR 发送器与真机仍走原 motion actor，不会自动切成 human。正式接 PICO human 推理前，还需独立接入 Uniform skeleton 映射、统一人体坐标/时钟，并为原 930 维观测中的 6D 参考骨盆项提供对齐后的参考姿态，再做仿真验证；不能把原始 PICO 四元数直接当成训练用的 SOMA/SMPL 姿态。

代码来源与修改说明见 `src/supermimic/human/NOTICE`，上游适用 Apache-2.0。当前未开始真实 SEED human 训练；测试中的合成人体只用于验证程序接口，不作为真实训练数据。
