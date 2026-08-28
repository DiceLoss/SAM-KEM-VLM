# GPU 实验人员唯一执行说明

## 任务边界

实验人员不再设计协议、选择数据、修改超参数或解释结果，只负责在有 GPU 和完整
原始数据的服务器上依次执行本文件命令，并把完整产物与日志交回。

当前阶段是 **development-only**：BrainWeb 为 16/4 个 source model、HECKTOR
为 179/45 个 patient，`test_locked=0`。所有 244 个独立对象都已在历史实验中
暴露，因此本阶段可以训练和验证 normalization 候选，但不能产生最终测试结论。

BrainWeb 不是每个模型只取一层：每个 source model 中所有有限、非负且总活度大于
零的轴向层都作为二维样本，同时严格继承该模型的 train/validation 归属。HECKTOR
当前使用 NPZ `input[subject_id-1,72,:,:,1]`。代码不会读取 HECKTOR 肿瘤 mask。

## 1. 安装并固定环境

```bash
git clone https://github.com/DiceLoss/SAM-KEM-VLM.git
cd SAM-KEM-VLM
git switch main
git pull --ff-only origin main

conda create -n sam-kem-gpu python=3.11 -y
conda activate sam-kem-gpu

# 根据服务器驱动安装 PyTorch 2.5.1 + torchvision 0.20.1。
# CUDA 12.4 示例：
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -e '.[data,test,evaluation,medsam2]'

git clone https://github.com/bowang-lab/MedSAM2.git external/MedSAM2
git -C external/MedSAM2 checkout 332f30d420f1d1b08e2a79b3ae6a602458808383
test "$(git -C external/MedSAM2 rev-parse HEAD)" = \
  "332f30d420f1d1b08e2a79b3ae6a602458808383"

mkdir -p checkpoints
curl -L --fail \
  -o checkpoints/MedSAM2_latest.pt \
  https://huggingface.co/wanglab/MedSAM2/resolve/c0b302de5e19211e9f1425c6215f6172eb5eca7f/MedSAM2_latest.pt
echo 'c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060  checkpoints/MedSAM2_latest.pt' \
  | sha256sum -c -

make quality
```

禁止切换 MedSAM2 commit、配置文件或 checkpoint。官方配置必须是
`sam2/configs/sam2.1_hiera_t512.yaml`。

## 2. 一次性准备来源清单

将下面两个路径换成服务器上的真实路径。`--hecktor-unit` 必须填写实际 DICOM
导出数值的单位；当前若已确认是 SUV，就写 `SUV`，不能猜测为 scanner counts。

```bash
sam-kem prepare-development-sources \
  --root . \
  --config code/configs/experiment.yaml \
  --brainweb-root /ABSOLUTE/PATH/brainweb_petmr \
  --hecktor-npz /ABSOLUTE/PATH/HECKTOR.npz \
  --hecktor-unit SUV \
  --output-root staging/clean_v1_multislice_sources \
  --manifest inputs/clean_v1/source_activity_manifest.csv

git add inputs/clean_v1/source_activity_manifest.csv \
  inputs/clean_v1/source_activity_manifest.csv.manifest.json
git commit -m 'Record development PET source manifest'
```

该命令会展开 BrainWeb 每个模型的全部有效活度层；同时流式解包
`HECKTOR.npz` 的 `input.npy`，只提取 PET channel，不会打开其他 NPZ member。
若 SHA、数组形状、subject 映射或切片值异常，命令会终止。来源清单应为 3735
行：BrainWeb 3511 层（2815 train / 696 validation）加 HECKTOR 224 层
（179/45），绝不能仍是 244。BrainWeb 每个模型有 172--181 个有效层；这些数量
已由提交的 20 个完整追踪 volume 独立核对，并由配置强制检查。

## 3. 摄入数据并拟合 normalization 候选

```bash
sam-kem ingest-activity \
  --root . --config code/configs/experiment.yaml \
  --manifest inputs/clean_v1/source_activity_manifest.csv \
  --output arrays/clean_v1 --splits train validation

sam-kem build-normalization-manifest \
  --root . --arrays arrays/clean_v1 \
  --output inputs/clean_v1/normalization_activity_manifest.csv

# calibration 会拒绝脏工作树，先固定它即将读取的清单。
git add inputs/clean_v1/normalization_activity_manifest.csv*
git commit -m 'Record development normalization input manifest'

sam-kem calibrate-normalization \
  --root . --config code/configs/experiment.yaml \
  --protocol code/configs/protocols/normalization_calibration.yaml

git add runs/clean_v1/normalization_calibration
git commit -m 'Calibrate development normalization candidates'
```

Step 3A 审计后只生成一个训练配置。两个数据集的训练集最小值均为 0，因而
`train_global_minmax` 与 `train_max_fixed_C` 在所有训练输入上逐像素相同；前者只会
在验证阶段额外裁剪。它保留为诊断项，不重复训练 decoder。禁止用 per-image
min-max 作为主实验：

```bash
sam-kem generate-candidate-config \
  --root . --base-config code/configs/experiment.yaml \
  --candidates runs/clean_v1/normalization_calibration/normalization_candidates.csv \
  --candidate train_max_fixed_C --run-id dev_trainmax_v1 \
  --output code/configs/generated/dev_trainmax_v1.yaml \
  --protocol-output-dir code/configs/generated/dev_trainmax_v1_protocols

git add code/configs/generated
git commit -m 'Freeze development fixed-scale config'
```

生成器只复制 8 个下游协议；已经完成的 `normalization_calibration.yaml` 不属于候选
训练 bundle，不得复制或重跑。`generation_manifest.json` 是配置和协议的权威 SHA
清单。不要使用 Windows 工作树在 `git add` 之前计算的 CRLF 文件哈希；生成器以
LF 原子写入，提交前后哈希必须一致。

## 4. 为固定尺度配置生成训练输入和 Qwen box

本阶段分三道门执行，禁止跨门：4A 生成 case，4B 只渲染提示图，4C 才允许产生
付费 Qwen 调用。每道门结束后先审核数量、哈希、失败率和归档空间。

### 4A. 生成 case 和输入 manifest

```bash
RUN=dev_trainmax_v1
CFG=code/configs/generated/${RUN}.yaml

sam-kem ingest-activity --root . --config "$CFG" \
  --manifest inputs/clean_v1/source_activity_manifest.csv \
  --output "arrays/${RUN}" --splits train validation

sam-kem prepare-cases --root . --config "$CFG" \
  --arrays "arrays/${RUN}" --splits train validation --resume-verified

sam-kem build-manifest --root . --config "$CFG" \
  --arrays "arrays/${RUN}" \
  --output "results/${RUN}/case_manifest_inputs.csv"
```

4A 的 manifest 应为 44,820 行。提交 CSV 及 sidecar 后停止，等待审核。

### 4B. 只渲染提示图

```bash
sam-kem prepare-prompt-images --root . --config "$CFG" \
  --manifest "results/${RUN}/case_manifest_inputs.csv" \
  --output "prompts/${RUN}/train_validation_images" \
  --splits train validation --resume-verified
```

4B 必须得到 14,940 行 prompt manifest，即 3,735 个切片乘 4 个 count level。
PNG 和逐图 JSON sidecar 属于外部不可变归档的 bulk artifacts；Git 只提交
`prompt_manifest.csv` 及其 manifest。此时不得设置 API key，完成后再次停止审核。

### 4C. 经单独批准后调用 Qwen；审核后再构建 box cache

先用 `--pilot-one-per-dataset-count` 固定选择每个 dataset/count 字典序第一条，共8条，
但写入独立的 `qwen_pilot_attempts` 目录。pilot 不写 `_SUCCESS.json`，也绝不复用为
正式response；审核8条响应后，才允许在空的 `train_validation_responses` 中执行完整
批次。禁止跳过pilot直接发出14,940个请求任务。401/403等不可恢复HTTP错误只归档
一次并立即中止整个批次，禁止把认证/配置错误转为质心fallback。

```bash
export DASHSCOPE_API_KEY='服务器环境变量中的真实 key'
QWEN_IMPLEMENTATION_SHA256="$(sha256sum code/sam_kem/qwen.py | awk '{print $1}')"
sam-kem request-boxes --root . --config "$CFG" \
  --manifest "prompts/${RUN}/train_validation_images/prompt_manifest.csv" \
  --responses "prompts/${RUN}/qwen_pilot_attempts" \
  --policy inputs/clean_v1/qwen_request_policy.json \
  --implementation-sha256 "$QWEN_IMPLEMENTATION_SHA256" \
  --resume-verified --pilot-one-per-dataset-count

# pilot 经研究员单独批准后才在独立的正式目录执行完整批次：
sam-kem request-boxes --root . --config "$CFG" \
  --manifest "prompts/${RUN}/train_validation_images/prompt_manifest.csv" \
  --responses "prompts/${RUN}/train_validation_responses" \
  --policy inputs/clean_v1/qwen_request_policy.json \
  --implementation-sha256 "$QWEN_IMPLEMENTATION_SHA256" \
  --resume-verified

# 立即清除当前 shell 中的凭证并停止；此时禁止 build-box-cache。
unset DASHSCOPE_API_KEY
```

正式批次必须从空目录开始，包含全部 14,940 个 cache key；独立 pilot 的 8 条不得
复制或计入正式批次。每条正式成功 attempt 显式记录 HTTP 状态，并钉住实际执行的
`code/sam_kem/qwen.py` SHA-256。完整批次结束后先提交 `_SUCCESS.json`、运行日志和
汇总审计，按照 `inputs/clean_v1/qwen_pilot_visual_audit.json` 的规则完成分层视觉审核。
在研究员再次明确批准之前，不得执行下面的 cache 命令。

`dev_trainmax_v1` 的正式批次曾因单个服务商 HTTP 400 中断，原始目录已作只读取证；
后续按 `inputs/clean_v1/qwen_full_batch_incident_resolution_20260828.json` 透明迁移并在
`train_validation_responses_resolved_v1` 完成。该运行只能读取这个带有效
`_SUCCESS.json` 的派生目录，禁止把原始目录与派生目录混用。语义审核结论固定在
`inputs/clean_v1/qwen_box_semantic_decision.json`：14,928 个成功框原样保留，只有12个
终止请求失败使用预先声明的质心框；不得后验扩框、筛框、改提示词或选择性重试。

```bash
# 仅在完整响应批次通过单独审核后执行：
sam-kem build-box-cache --config "$CFG" \
  --manifest "results/${RUN}/case_manifest_inputs.csv" \
  --responses "prompts/${RUN}/train_validation_responses_resolved_v1" \
  --policy inputs/clean_v1/qwen_request_policy.json \
  --output "prompts/${RUN}/train_validation_box_cache.csv"

# 必须通过逐框审计；它验证成功框坐标完全不变且 fallback 恰好对应12个终止失败键。
python code/experiments/audit_qwen_box_cache.py \
  --cache "prompts/${RUN}/train_validation_box_cache.csv" \
  --diagnostics logs/qwen_box_diagnostics.csv \
  --semantic-summary logs/qwen_box_semantic_summary.json \
  --decision inputs/clean_v1/qwen_box_semantic_decision.json \
  --output logs/qwen_box_cache_audit.json

git add "results/${RUN}" \
  "prompts/${RUN}/train_validation_images/prompt_manifest.csv" \
  "prompts/${RUN}/train_validation_images/prompt_manifest.csv.manifest.json" \
  "prompts/${RUN}/train_validation_responses_resolved_v1/_SUCCESS.json" \
  "prompts/${RUN}/train_validation_box_cache.csv" \
  "prompts/${RUN}/train_validation_box_cache.csv.manifest.json" \
  logs/qwen_box_cache_audit.json
git commit -m "Prepare ${RUN} decoder inputs and boxes"
```

完成 cache 与审计后再次停止。收到独立的 `Go Step 5` 之前，不得执行 GPU preflight
或 decoder 训练。

每个 `(dataset, subject, slice, count)` 只有一个请求任务，使用 r=0 MLEM-10；
r=0/1/2 共用该 box。每个任务严格使用同一份最多三次的固定重试预算，每次 attempt
都归档；禁止按结果挑选性追加重试。合法 no-detection 和完整预算耗尽的 request
failure 分别记录，再使用质心 fallback。

## 5. GPU 开工门和正式训练

只运行下面一条命令。它先执行真实 CUDA 模型构造、512 网格 box、前向、
L1、反向和冻结参数检查；任何一步失败都不会开始长训练。通过后依次训练 BrainWeb
和 HECKTOR，各自一个四剂量联合 decoder。

Step 5 不调用 Qwen，也不需要任何 DashScope 凭证。训练前必须保持
`DASHSCOPE_API_KEY` 未设置；旧 key 的轮换属于独立安全处置，不能把新 key 注入训练
进程。checkpoint 与 resume state 会同时绑定训练 manifest、manifest sidecar、box
cache 和 cache sidecar 的 SHA-256，任一输入变化都会拒绝恢复或复用。

```bash
set -o pipefail
unset DASHSCOPE_API_KEY
mkdir -p logs
python code/experiments/run_gpu_training.py \
  --config code/configs/generated/dev_trainmax_v1.yaml \
  2>&1 | tee logs/gpu_dev_trainmax_v1.log

# 两个数据集均完成后执行；任何 provenance、输入绑定或训练协议不一致都会失败。
python code/experiments/audit_decoder_training.py --root . \
  --config code/configs/generated/dev_trainmax_v1.yaml \
  --output logs/decoder_training_handoff_audit.json

# 固定小型 provenance/metrics；二进制 .pt 仍由 .gitignore 排除。
git add checkpoints/dev_trainmax_v1/*.json \
  checkpoints/dev_trainmax_v1/*.training.csv \
  checkpoints/dev_trainmax_v1/*.training.csv.manifest.json \
  logs/decoder_training_handoff_audit.json
git commit -m 'Record dev_trainmax_v1 decoder training provenance'

```

训练协议固定为 20 epochs、每 epoch 400 steps、AdamW、学习率 `1e-4`、batch 1、
L1-only。image/prompt/memory encoder 全部冻结，只更新 mask decoder 中与所选 single-mask
logit 相连的参数。与该 L1 目标无连接的 IoU/object-score 输出头及另外三个 multimask
hypernetwork 输出头保持冻结；这不会改变 single-mask 前向值，只避免把必然
`grad=None` 的旁路分支误报为训练失败。中断后重新执行同一命令会读取该 checkpoint
对应的 `.resume.pt`，不得另建临时 checkpoint。

## 6. GPU 完成后立即交回

对 `dev_trainmax_v1` 的两个数据集分别交回：

- `medsam2_decoder_*.pt`；
- `.resume.pt`；
- `.json` provenance；
- `.training.csv` 及 manifest；
- `logs/gpu_dev_trainmax_v1.log`；
- `nvidia-smi`、Python、PyTorch、CUDA 版本；
- 所有文件 SHA-256 和不可变归档 URI。

不要另行生成或训练 `dev_globalminmax_v1`，不要运行 test，不要修改 epochs、学习率、
loss、box 或 checkpoint。GPU 训练完成后停止，等待主研究员注册 checkpoint SHA 并
启动 validation sweep。
