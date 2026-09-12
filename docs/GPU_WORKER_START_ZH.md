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
`grad=None` 的旁路分支误报为训练失败。训练图像路径使用官方 `model.forward_image`：
冻结的 image encoder 不建立梯度，而 decoder 的高分辨率 `conv_s0/conv_s1` 投影保留
autograd；不能使用官方 predictor 中带 `@torch.no_grad()` 的 `set_image` 代替该训练
路径。中断后重新执行同一命令会读取该 checkpoint 对应的 `.resume.pt`，不得另建临时
checkpoint。

## 6. GPU 完成后立即交回

两个数据集的 handoff audit 通过并提交 provenance 后，先构建只读本地交付包：

```bash
python code/experiments/package_decoder_checkpoints.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1.yaml \
  --output delivery/dev_trainmax_v1_decoder_bundle.zip \
  --record logs/decoder_checkpoint_archive_build.json

python code/experiments/verify_decoder_checkpoint_bundle.py \
  --archive delivery/dev_trainmax_v1_decoder_bundle.zip \
  --output logs/decoder_checkpoint_archive_verify.json
```

打包器会重新核验 `.pt`、`.resume.pt`、训练曲线及 sidecar、provenance 和 handoff audit
之间的 SHA 绑定；使用固定 ZIP member 元数据并在包内写入 `MANIFEST.sha256`。它拒绝
覆盖已有 archive/record，也拒绝从脏工作树打包。独立 verify 命令再流式计算每个成员
SHA，拒绝重复、缺失、额外或校验不符的成员。对 `dev_trainmax_v1` 的两个数据集
分别交回：

- `medsam2_decoder_*.pt`；
- `.resume.pt`；
- `.json` provenance；
- `.training.csv` 及 manifest；
- `logs/gpu_dev_trainmax_v1.log`；
- `nvidia-smi`、Python、PyTorch、CUDA 版本；
- 所有文件 SHA-256 和不可变归档 URI。

`decoder_checkpoint_archive_build.json` 的初始状态必须是
`local_bundle_complete_pending_immutable_upload`，其中 `immutable_archive_uri` 保持
`null`。本地 ZIP、网盘共享路径或可覆盖 URL 不能冒充不可变 URI。只有上传到带版本、
对象锁或 DOI 的外部位置并复核下载后 SHA 一致，才能另行登记 archive URI。

### 6B. 外部发布后注册下游配置

不得原地修改 `dev_trainmax_v1.yaml`：两个训练 provenance 已绑定其 SHA。外部归档
发布后，必须从给定 URI 重新下载一份 ZIP 到不同路径，再生成独立的 registered config：

```bash
python code/experiments/register_decoder_archive.py \
  --root . \
  --training-config code/configs/generated/dev_trainmax_v1.yaml \
  --build-record logs/decoder_checkpoint_archive_build.json \
  --verify-record logs/decoder_checkpoint_archive_verify.json \
  --verified-download /ABSOLUTE/PATH/REDOWNLOADED/dev_trainmax_v1_decoder_bundle.zip \
  --archive-uri 'VERSIONED_OR_DOI_BACKED_URI' \
  --archive-version 'IMMUTABLE_VERSION_OR_OBJECT_ID' \
  --output-config code/configs/generated/dev_trainmax_v1_registered.yaml \
  --registration-record logs/decoder_checkpoint_archive_registration.json
```

注册器拒绝本地原包冒充重新下载、拒绝无 URI scheme/版本标识、拒绝下载后 SHA/大小
不一致，并保持全局 full-result `immutable_archive_uri: null`。validation prior 和 sweep
只能读取 registered config；原训练配置继续作为不可变训练证据。

当前 development bundle 可通过仓库内精确路径的 Git LFS 规则发布。只允许
`delivery/dev_trainmax_v1_decoder_bundle.zip`；不得把 `*.pt`、其他 ZIP 或患者数组批量
加入 LFS。LFS pointer 的 `oid sha256` 必须等于 bundle SHA
`acacc789c28279b1084167659391d37a51cb6c97e319d96534dc522ca48c6347`，size 必须为
`125563903`。推送后必须在一个不同的新 clone/checkout 中按固定 Git commit 执行
`git lfs pull`，再把下载文件交给注册器。Git commit、LFS OID 和 bundle SHA 三者共同
作为 development archive version；最终论文公开归档仍应迁移到长期 DOI/对象锁存储。

不要另行生成或训练 `dev_globalminmax_v1`，不要运行 test，不要修改 epochs、学习率、
loss、box 或 checkpoint。GPU 训练完成后停止，等待主研究员注册 checkpoint SHA 并
启动 validation sweep。

### 6C. Validation prior GPU smoke gate

注册完成后先执行只读 smoke test。它在 200k count、anchor realization 0 上，按
`subject_id, slice_id` 字典序分别选择 BrainWeb 和 HECKTOR 的第一条 validation case，
每个数据集只运行一次 decoder forward。该命令不得写正式 prior，也不得启动 sweep：

```bash
PYTHONPATH=code python code/experiments/validation_prior_preflight.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_registered.yaml \
  --manifest results/dev_trainmax_v1/case_manifest_inputs.csv \
  --count-level 200000
```

只有输出 `status: PASS`、两个 checkpoint SHA 与注册配置一致、两个 prior 都是有限的
`128x128` 连续数组，才允许另行下达 validation prior 全量生成指令。smoke test 不需要
DashScope key，不得调用 Qwen，不得读取 test。

全量 validation prior 必须使用可校验恢复模式，并写独立完成记录：

```bash
PYTHONPATH=code python code/experiments/generate_priors.py \
  --config code/configs/generated/dev_trainmax_v1_registered.yaml \
  --manifest results/dev_trainmax_v1/case_manifest_inputs.csv \
  --splits validation \
  --output-dir runs/dev_trainmax_v1/validation_priors \
  --resume-verified
```

恢复时只允许复用 `.npy` 与相邻 JSON 都存在且 run、split、输入 MLEM、box cache、
registered config、checkpoint 和 prior array SHA 全部一致的 prior。完成后必须得到
`prior_manifest.csv`、sidecar、`generation_record.json` 和 `_SUCCESS.json`。validation
应为 8,892 个 case/prior；不得同时生成 train 或 test prior。

生成完成后不得直接进入 sweep。先逐个验证 prior、metadata、MLEM、box、checkpoint 和
完成 manifest，并记录 raw continuous output 的越界分布：

```bash
PYTHONPATH=code python code/experiments/audit_validation_priors.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_registered.yaml \
  --case-manifest results/dev_trainmax_v1/case_manifest_inputs.csv \
  --prior-dir runs/dev_trainmax_v1/validation_priors \
  --diagnostics logs/validation_prior_diagnostics.csv \
  --summary logs/validation_prior_audit.json
```

必须输出 `status: PASS` 和 8,892 条 diagnostics。审计会如实列出 validation 中实际
使用 fallback box 的 cache key；不得根据 prior 数值或审计结果筛除样本或重生成 box。

审计与 prior completion manifests 提交后，必须从干净工作树打包并独立流式校验。归档
只允许使用精确 LFS 路径 `delivery/dev_trainmax_v1_validation_priors.zip`：

```bash
PYTHONPATH=code python code/experiments/package_validation_priors.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_registered.yaml \
  --prior-dir runs/dev_trainmax_v1/validation_priors \
  --audit logs/validation_prior_audit.json \
  --diagnostics logs/validation_prior_diagnostics.csv \
  --output delivery/dev_trainmax_v1_validation_priors.zip \
  --record logs/validation_prior_archive_build.json

PYTHONPATH=code python code/experiments/verify_validation_prior_bundle.py \
  --archive delivery/dev_trainmax_v1_validation_priors.zip \
  --output logs/validation_prior_archive_verify.json
```

build/verify 的 archive SHA、manifest SHA、成员数和 8,892 prior 数必须一致。验证前后
不得修改或重新生成 prior；归档上传并从 fresh clone 完整 LFS 下载验证前不得进入 sweep。

LFS 发布并 fresh-clone 验证后生成第三份下游配置；不得修改 decoder registered config：

```bash
PYTHONPATH=code python code/experiments/register_validation_prior_archive.py \
  --root . \
  --source-config code/configs/generated/dev_trainmax_v1_registered.yaml \
  --build-record logs/validation_prior_archive_build.json \
  --verify-record logs/validation_prior_archive_verify.json \
  --verified-download /ABSOLUTE/FRESH_CLONE/delivery/dev_trainmax_v1_validation_priors.zip \
  --archive-uri 'PINNED_GITHUB_COMMIT_BLOB_URI' \
  --archive-version 'GIT_COMMIT_AND_LFS_OID_AND_SIZE' \
  --output-config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --registration-record logs/validation_prior_archive_registration.json
```

注册器再次流式验证下载 ZIP 的 17,799 个成员。sweep 入口只接受显式 `--config` 和
`--protocol`，并强制检查这条 prior archive 注册链；旧的默认 experiment config 不再
能意外启动 validation selection。

### 6H. Anatomy comparison-source inventory

MR/CT 只属于 MR-KEM、CT-KEM 和 Bowsher-KEM 对比方法；proposed method 禁止读取 anatomy。
在生成 `anatomy.npy` 前先做只读清点，BrainWeb 不允许根据文件名自动猜一个 MR：

```bash
PYTHONPATH=code python code/experiments/audit_anatomy_sources.py \
  --root . \
  --source-manifest inputs/clean_v1/source_activity_manifest.csv \
  --output logs/anatomy_source_inventory.json
```

输出必须列出 20 个 BrainWeb subject 各自目录中的全部 NIfTI 候选、SHA、shape 和 spacing；
HECKTOR 必须固定为同一已验证 NPZ 的 `input[subject_id-1,72,:,:,0]` CT/HU 通道。该步骤
不得选择 MR、不得写 anatomy array、不得重建 manifest，也不得启动 sweep。

### 6I. Freeze and ingest anatomy comparison inputs

研究员选择规则固定如下：BrainWeb 只使用每个 subject 根目录的 `t1.nii.gz`；`mu.nii.gz`
和任何 `sim_*` PET 派生文件均不得作为 MR。T1 必须与该 subject 的 `sim_0/true_pet.nii.gz`
具有相同 shape 和 NIfTI affine/axis codes，不能只凭 shape/spacing 判断共注册。HECKTOR 使用
同一 NPZ 中与 PET 同 subject、同 z=72 的 `input[...,0]` CT/HU。存储保持 raw float32：T1
不做逐图归一化，CT 不做 window/clip；仅在 MR-KEM/CT-KEM/Bowsher-KEM 构造 kernel feature
时做既定 per-image min--max。proposed method 禁止读取这些文件。

旧的 schema-v1 inventory 不含 affine，作为 forensic evidence 保留但不得用于 freeze。用新文件
名生成 schema-v2 inventory，并把其实际 SHA 传给 freeze：

```bash
PYTHONPATH=code python code/experiments/audit_anatomy_sources.py \
  --root . \
  --source-manifest inputs/clean_v1/source_activity_manifest.csv \
  --output logs/anatomy_source_inventory_v2.json

PYTHONPATH=code python code/experiments/prepare_anatomy.py freeze \
  --root . \
  --inventory logs/anatomy_source_inventory_v2.json \
  --inventory-sha256 ACTUAL_SHA256 \
  --source-manifest inputs/clean_v1/source_activity_manifest.csv \
  --output inputs/clean_v1/anatomy_source_manifest.csv

PYTHONPATH=code python code/experiments/prepare_anatomy.py ingest \
  --root . \
  --manifest inputs/clean_v1/anatomy_source_manifest.csv \
  --arrays arrays/dev_trainmax_v1 \
  --splits train validation --resume-verified

PYTHONPATH=code python code/experiments/prepare_anatomy.py audit \
  --root . \
  --manifest inputs/clean_v1/anatomy_source_manifest.csv \
  --arrays arrays/dev_trainmax_v1 \
  --splits train validation \
  --output logs/anatomy_ingestion_audit.json
```

首次完整摄入必须报告 3,735 个数组（BrainWeb 3,511、HECKTOR 224）；audit 必须为 `PASS`。
随后才可构建只含 validation 的 8,892-row case manifest：

```bash
sam-kem build-manifest --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --arrays arrays/dev_trainmax_v1 \
  --splits validation \
  --output results/dev_trainmax_v1/validation_case_manifest.csv
```

在 anatomy audit 与 validation manifest sidecar 均提交审核前，不得启动 sweep。

### 6J. Validation sweep exact-grid runtime preflight

完整 validation grid 对每个 case 产生 580 条结果；8,892 个 case 共 5,157,360 条，并执行
41,347,800 次 MLEM/OSEM/KEM operator update。当前 reference backend 是 SciPy sparse CSR
float64 CPU，RTX 3090 不参与。因此在决定分片/恢复或 CUDA 等价实现前，只允许对每个 dataset
一个固定 case 跑完整网格：

```bash
PYTHONPATH=code python code/experiments/validation_sweep_runtime_preflight.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --output logs/validation_sweep_runtime_preflight.json
```

固定样本选择规则为每个 dataset 在 validation、c=200000、r=0 下按 subject/slice 字典序第一项；
预期是 BrainWeb `subject04/z038` 与 HECKTOR `10/z072`。每项必须实际完成 580 个 metric
evaluation，而不是缩小网格。该步骤不得创建 `runs/.../hyperparameter_sweep`、不得冻结参数、
不得读取 test，也不得把两样本结果当作科学比较结果。

### 6K. Eight-process CPU scaling gate

正式 backend 固定为 reference SciPy CSR float64 的 CPU 多进程分片；不得用 PyTorch thread
数量推断 SciPy 性能，也不得同时运行 36 个 process 和每进程 36 个 BLAS thread。CUDA
sparse 重写在 validation selection 前不采用。先以 8 个独立 process 并发运行 8 个完整
580-evaluation case，验证 Windows spawn、内存和实际吞吐：

```bash
PYTHONPATH=code python code/experiments/validation_sweep_parallel_preflight.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --runtime-preflight logs/validation_sweep_runtime_preflight.json \
  --output logs/validation_sweep_parallel_preflight.json \
  --workers 8
```

允许 serial runtime JSON 在运行前保持唯一 untracked evidence；其他 Git status 必须为空。
程序固定选择 BrainWeb 与 HECKTOR 各前 4 个不同 validation slice、c=200000、r=0。不得更改
worker 数，不得设置 OMP/MKL/OpenBLAS threads，不得生成正式 sweep 或 frozen 参数。根据
`measured_speedup`、`parallel_efficiency`、峰值内存和投影总时长再固定正式并发数。

### 6L-pre. Shared evaluator and shard-writer smoke

8-process 实测固定正式 backend 为 SciPy CSR float64、8 个长驻 process。正式运行前先验证重构
evaluator 的单一实现及 fragment 写入/sidecar/恢复：对 Step 6J 的两个原样本各写一个 580-row
staging fragment，并要求每种方法的 `(NRMSE, SSIM)` float64 byte hash 与 Step 6J 完全一致。

```bash
PYTHONPATH=code python code/experiments/validation_sweep_shard_smoke.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --runtime-preflight logs/validation_sweep_runtime_preflight.json \
  --parallel-preflight logs/validation_sweep_parallel_preflight.json \
  --output-dir staging/dev_trainmax_v1_validation_sweep_shard_smoke \
  --audit logs/validation_sweep_shard_smoke.json
```

输出必须是 2 个 fragment、共 1,160 行，两个 fragment 都通过内部 resume verification；
BrainWeb 与 HECKTOR 的所有方法 `all_methods_exact: true`。staging 不是正式 sweep，不得合并、
不得冻结参数。三个 preflight/smoke JSON 在审核前均不提交。

### 6L-plan. Immutable slice-group shard plan

正式执行前将完整 validation manifest 固化为 741 个不可变 slice-group shard：BrainWeb 696、
HECKTOR 45。每个 shard 唯一对应 `(dataset, subject_id, slice_id)`，包含固定 4 个 count × 3 个
realization = 12 cases、6,960 result rows。计划必须通过 SHA 链绑定 6J/6K/6L-pre 三份递进
证据、registered config、protocol 和 case manifest，并记录本次正式执行计划的 Git commit。

```bash
PYTHONPATH=code python code/experiments/build_validation_sweep_shard_plan.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --runtime-preflight logs/validation_sweep_runtime_preflight.json \
  --parallel-preflight logs/validation_sweep_parallel_preflight.json \
  --smoke-audit logs/validation_sweep_shard_smoke.json \
  --output logs/validation_sweep_shard_plan.json
```

计划本身不运行 reconstruction、不创建正式 sweep 输出、不合并 fragment、不冻结超参数。恢复时
只能跳过同一 Git commit 下 CSV、manifest、SUCCESS marker、主键、行数和 SHA-256 全部验证通过的
完整 shard；禁止覆盖 partial 或失败 shard。merge 与 hyperparameter freeze 均为独立的后续审核门。

### 6L-full. Formal resumable validation sweep shards

正式 runner 固定使用 8 个长驻 process 和 SciPy CSR float64 backend。研究员最终 release 前先
运行一次零写入 `--release-check-only`，完整验证 plan/config/protocol/manifest、741-shard membership、
thread environment 及正式 namespace 尚不存在。PASS 后才按计划运行 741 个 slice-group shard。
每个 shard 在隐藏临时目录写齐 CSV、CSV manifest、SUCCESS marker 后原子发布整个目录；正式
namespace 被 Git 忽略。首次运行不得使用 resume 参数：

```bash
PYTHONPATH=code python code/experiments/run_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --plan logs/validation_sweep_shard_plan.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_sweep_shards \
  --workers 8 \
  --release-check-only
```

正式 release 时使用同一命令但移除 `--release-check-only`。

只有进程中断后才可在完全相同的 Git commit 和输入绑定上追加 `--resume-verified`。任何 shard
三件套不完整或 SHA、主键、行数、case membership、执行 commit 不匹配都会停止；不得删除、覆盖
或手工补写。741 个 shard 全部验证后只写 `shard_batch_SUCCESS.json`，仍不合并结果、不冻结参数。

### 6M-pre. Read-only completed-shard integrity audit

全量 shard 完成后先做只读审计，不合并、不冻结。审计逐一验证 741 组 CSV/manifest/SUCCESS、
5,157,360 个结果行的分片主键与方法网格、execution/plan/input SHA 链、首末 shard 和时间线。
中断遗留的隐藏事务目录只做 SHA inventory，保留原状并从正式 shard set 排除。

```bash
PYTHONPATH=code python code/experiments/audit_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --plan logs/validation_sweep_shard_plan.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_sweep_shards \
  --audit logs/validation_sweep_shard_integrity_audit.json
```

审计 PASS 前禁止 merge。审计输出在研究员审核前不提交；正式 shard 及隐藏事务目录始终不提交、
不删除、不移动、不覆盖。

### 6M-merge. Transactional merge integrity gate

已审核并提交 shard audit 后，按冻结 shard 顺序流式合并 741 个 CSV。合并前重新计算完整
shard-set SHA，并逐项确认 8 个中断事务目录未变；合并目录包含 CSV、sidecar、merge record、
SUCCESS 四个文件。整个目录一次原子发布，不读取原始图像、不运行 reconstruction、不冻结参数。

```bash
PYTHONPATH=code python code/experiments/merge_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --plan logs/validation_sweep_shard_plan.json \
  --shard-dir runs/dev_trainmax_v1/hyperparameter_sweep_shards \
  --shard-audit logs/validation_sweep_shard_integrity_audit.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_sweep \
  --audit logs/validation_sweep_merge_integrity_audit.json
```

输出必须为 5,157,360 行且 `PASS_complete_pending_hyperparameter_freeze`。merge audit 在审核前
不提交；741 个 shard、8 个中断事务目录和四个 merge 文件均不得删除、移动、覆盖或提交。

### 6T. Transactional incremental-merge integrity gate

增量 shard audit 经研究员审核并提交后，按 shard ID 顺序流式合并 741 个增量 CSV。该 gate
重新验证全部 shard 三件套、已审核 shard-set SHA、原始 merged CSV 和 expansion plan/smoke/
selection audit 绑定，只生成独立的 80,568 行增量 CSV；不与原始 5,157,360 行 CSV 拼接，
不执行 union selection，也不冻结参数。

```bash
PYTHONPATH=code python code/experiments/merge_incremental_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --selection-audit logs/validation_hyperparameter_selection_audit.json \
  --plan logs/validation_hyperparameter_expansion_plan.json \
  --smoke-audit logs/validation_hyperparameter_expansion_shard_smoke.json \
  --shard-dir runs/dev_trainmax_v1/hyperparameter_expansion_shards \
  --shard-audit logs/validation_hyperparameter_expansion_shard_integrity_audit.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_expansion \
  --audit logs/validation_hyperparameter_expansion_merge_integrity_audit.json
```

输出必须为 80,568 行且状态为 `PASS_complete_pending_union_selection_audit`。正式目录只含
incremental CSV、sidecar、merge record 和 SUCCESS 四个文件。审核前不提交 merge audit；
原始 merged CSV、两组 shards、旧事务目录和 smoke staging 均不得删除、移动或覆盖。

### 6U. Streaming union-selection audit

增量 merge audit 经研究员审核并提交后，分别流读原始 5,157,360 行 CSV 和增量 80,568 行
CSV，在同一 subject-balanced 聚合器中选择；不创建物理 union CSV。30 个 reviewed boundary
group 的赢家只能是原赢家或预声明 local shell。仅当赢家占据本轮新增的外扩值时才再次触发扩边；
18 个原内部选择必须保持不变。

```bash
PYTHONPATH=code python code/experiments/audit_validation_hyperparameter_union_selection.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --original-merge-audit logs/validation_sweep_merge_integrity_audit.json \
  --original-selection-audit logs/validation_hyperparameter_selection_audit.json \
  --plan logs/validation_hyperparameter_expansion_plan.json \
  --incremental-merge-audit logs/validation_hyperparameter_expansion_merge_integrity_audit.json \
  --output logs/validation_hyperparameter_union_selection_audit.json
```

`union_rows_streamed` 必须为 5,237,928，选择必须为 BrainWeb 24 + HECKTOR 24。只有
`expanded_boundary_selection_count == 0` 时状态才是 `PASS_ready_for_hyperparameter_freeze`；
否则状态为 `BLOCKED_expanded_boundary_requires_additional_shell`，必须先审核下一轮增量扩边，禁止冻结。

### 6V-plan. Second incremental local-shell plan

若第一轮 union audit 仍为 BLOCKED，只对占据第一轮新增外边界的 25 个组继续扩展。第二轮规则为
sigma `0.2 -> 0.4`、普通 radius `4 -> 5`、Bowsher neighbours `16 -> 20`、
Bowsher radius `4 -> 5`、iterations `140 -> 160`；没有新的 iterations lower 扩展。
18 个单轴组和 7 个双轴组产生 39 个交互完整候选（BrainWeb 22、HECKTOR 17），共 48,231 行。

```bash
PYTHONPATH=code python code/experiments/plan_round2_validation_hyperparameter_expansion.py \
  --root . \
  --union-audit logs/validation_hyperparameter_union_selection_audit.json \
  --previous-plan logs/validation_hyperparameter_expansion_plan.json \
  --incremental-merge-audit logs/validation_hyperparameter_expansion_merge_integrity_audit.json \
  --output logs/validation_hyperparameter_expansion_round2_plan.json
```

该命令只写计划 JSON，不运行 reconstruction、不创建 round-2 shard namespace、不修改第一轮结果，
也不生成 frozen hyperparameters。

### 6X-release-check. Final round-2 shard execution identity

第二轮计划经研究员审核并提交后，先运行零写入 release check。该 gate 精确绑定第二轮计划、
第一轮计划、第一轮 union selection audit、第一轮 incremental merge audit，以及原始和第一轮
incremental merged CSV；同时验证 741 个 shard 的完整 membership。BrainWeb 每 shard 66 行，
HECKTOR 每 shard 51 行，总计 48,231 行。

```bash
PYTHONPATH=code python code/experiments/run_round2_incremental_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --previous-plan logs/validation_hyperparameter_expansion_plan.json \
  --union-audit logs/validation_hyperparameter_union_selection_audit.json \
  --incremental-merge-audit logs/validation_hyperparameter_expansion_merge_integrity_audit.json \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --smoke-audit logs/validation_hyperparameter_expansion_round2_shard_smoke.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_expansion_round2_shards \
  --workers 8 \
  --release-check-only
```

smoke audit 提交并由 runner 绑定后，状态必须为
`PASS_ready_for_round2_incremental_shard_execution`，正式输出目录必须仍不存在。研究员审核该次
zero-write 输出后，才可使用同一命令移除 `--release-check-only`；中断后只可增加
`--resume-verified`，不得覆盖既有 shard。

### 6W-smoke. Round-2 fixed-slice numerical gate

初始 identity release gate 通过后，只运行固定的 BrainWeb `subject04/z038` 与 HECKTOR `10/z072` 两个
slice shard。两个进程分别产生 66 与 51 行，共 117 行；staging fragment 必须通过同进程写入和
严格 resume verification，且所有 NRMSE/SSIM 必须有限。

```bash
PYTHONPATH=code python code/experiments/smoke_round2_incremental_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --previous-plan logs/validation_hyperparameter_expansion_plan.json \
  --union-audit logs/validation_hyperparameter_union_selection_audit.json \
  --incremental-merge-audit logs/validation_hyperparameter_expansion_merge_integrity_audit.json \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --output-dir staging/dev_trainmax_v1_hyperparameter_expansion_round2_shard_smoke \
  --audit logs/validation_hyperparameter_expansion_round2_shard_smoke.json
```

smoke audit 经研究员审核并提交、其 SHA 被正式 runner 绑定之前，不得运行 741-shard round-2
计算。smoke staging 与正式 namespace 必须分离，且不得合并或生成 frozen hyperparameters。

### 6X-pre. Round-2 shard integrity audit

741 个 round-2 shard 完成后，先执行只读完整性审计。该 gate 逐 shard 验证 CSV、manifest、
SUCCESS、主键、候选集合、行数、SHA、有限 NRMSE/SSIM 与 execution identity，并核对恰好 8 个
worker PID。它只写外部 audit JSON，不合并 fragment，也不改动 shard namespace。

```bash
PYTHONPATH=code python code/experiments/audit_round2_incremental_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --previous-plan logs/validation_hyperparameter_expansion_plan.json \
  --union-audit logs/validation_hyperparameter_union_selection_audit.json \
  --incremental-merge-audit logs/validation_hyperparameter_expansion_merge_integrity_audit.json \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --smoke-audit logs/validation_hyperparameter_expansion_round2_shard_smoke.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_expansion_round2_shards \
  --audit logs/validation_hyperparameter_expansion_round2_shard_integrity_audit.json
```

状态必须为 `PASS_complete_pending_round2_incremental_merge`，共 741 套三件套、48,231 行，
BrainWeb/HECKTOR 分别为 45,936/2,295 行。审核并单独提交 audit 前不得 merge。

### 6X-merge. Round-2 incremental integrity merge

round-2 shard audit 经研究员审核并提交后，将 741 个 fragment 按 shard ID 与原行序事务性合并为
独立的 48,231 行 CSV。该步骤不物理联合原始或 round-1 CSV，只在记录中声明下一次 streaming
union 应读取 5,286,159 行。

```bash
PYTHONPATH=code python code/experiments/merge_round2_incremental_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --previous-plan logs/validation_hyperparameter_expansion_plan.json \
  --union-audit logs/validation_hyperparameter_union_selection_audit.json \
  --incremental-merge-audit logs/validation_hyperparameter_expansion_merge_integrity_audit.json \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --smoke-audit logs/validation_hyperparameter_expansion_round2_shard_smoke.json \
  --shard-dir runs/dev_trainmax_v1/hyperparameter_expansion_round2_shards \
  --shard-audit logs/validation_hyperparameter_expansion_round2_shard_integrity_audit.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_expansion_round2 \
  --audit logs/validation_hyperparameter_expansion_round2_merge_integrity_audit.json
```

状态必须为 `PASS_complete_pending_round2_union_selection_audit`；正式目录只含 round-2 CSV、
manifest、merge record 与 SUCCESS 四个文件。审核前不提交 merge audit，不运行 union selection，
不生成 frozen hyperparameters。

### 6Y. Three-source streaming union-selection audit

round-2 merge audit 经研究员审核并单独提交后，分别流读原始 5,157,360 行、round-1 80,568 行和
round-2 48,231 行 CSV，在同一 subject-balanced 聚合器中对 5,286,159 行做选择。该 gate 不创建
物理 union CSV；23 个未进入 round-2 的选择必须与第一轮 union audit 字节级一致，另外 25 个组的
赢家只能是第一轮 union 赢家或预声明的 round-2 local shell candidate。

```bash
PYTHONPATH=code python code/experiments/audit_round2_validation_hyperparameter_union_selection.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --original-merge-audit logs/validation_sweep_merge_integrity_audit.json \
  --original-selection-audit logs/validation_hyperparameter_selection_audit.json \
  --round1-plan logs/validation_hyperparameter_expansion_plan.json \
  --round1-merge-audit logs/validation_hyperparameter_expansion_merge_integrity_audit.json \
  --round1-union-audit logs/validation_hyperparameter_union_selection_audit.json \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --round2-merge-audit logs/validation_hyperparameter_expansion_round2_merge_integrity_audit.json \
  --output logs/validation_hyperparameter_round2_union_selection_audit.json
```

仅当 `expanded_boundary_selection_count == 0` 时状态才是
`PASS_ready_for_hyperparameter_freeze`；否则状态为
`BLOCKED_expanded_boundary_requires_additional_shell`。审核前不提交 audit；无论哪种状态，
本步骤均不得生成 frozen hyperparameters、物理 union CSV，或删除任何既有 shard、merge 与 staging
产物。

### 6Z-plan. Third incremental local-shell plan

若三源 union audit 仍为 BLOCKED，只对占据 round-2 新外边界的 12 个组继续扩展。第三轮规则为
sigma `0.4 -> 0.8`、普通 radius `5 -> 6`、Bowsher neighbours `20 -> 24`、
Bowsher radius `5 -> 6`、iterations `160 -> 180`。8 个单轴组与 4 个双轴组产生 20 个
交互完整候选（BrainWeb 15、HECKTOR 5），新增 31,995 行和 2,647,080 operator updates。

```bash
PYTHONPATH=code python code/experiments/plan_round3_validation_hyperparameter_expansion.py \
  --root . \
  --union-audit logs/validation_hyperparameter_round2_union_selection_audit.json \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --round2-merge-audit logs/validation_hyperparameter_expansion_round2_merge_integrity_audit.json \
  --output logs/validation_hyperparameter_expansion_round3_plan.json
```

该命令只写 SHA-bound 计划 JSON，不运行 reconstruction、不创建 round-3 shard namespace，也不修改
前三份 merged CSV。计划必须声明 cumulative incremental candidates/rows 为 135/160,794，下一次
四源 streaming union 为 5,318,154 行。审核前不提交计划，不生成 frozen hyperparameters。

### 6AA-release-check. Round-3 smoke execution identity

第三轮计划审核并提交后，先运行零写入 release check。该 gate 绑定 round-2 plan、三源 union audit、
round-2 merge audit、round-3 plan 和前三份 merged CSV，验证 741-shard membership。BrainWeb 每
shard 45 行，HECKTOR 每 shard 15 行，总计 31,995 行；当前实现只允许 `--release-check-only`，在
smoke audit 审核前从代码层禁止正式执行。

```bash
PYTHONPATH=code python code/experiments/run_round3_incremental_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --union-audit logs/validation_hyperparameter_round2_union_selection_audit.json \
  --round2-merge-audit logs/validation_hyperparameter_expansion_round2_merge_integrity_audit.json \
  --round3-plan logs/validation_hyperparameter_expansion_round3_plan.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_expansion_round3_shards \
  --workers 8 \
  --release-check-only
```

状态必须为 `PASS_ready_for_round3_incremental_shard_smoke`，正式 namespace 必须仍不存在。

### 6AA-smoke. Round-3 fixed-slice numerical gate

release check 通过后，只运行 BrainWeb `subject04/z038` 与 HECKTOR `10/z072` 两个固定 slice shard。
两个独立进程分别产生 45 与 15 行，共 60 行，并在 staging 内严格 resume-verify。

```bash
PYTHONPATH=code python code/experiments/smoke_round3_incremental_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --union-audit logs/validation_hyperparameter_round2_union_selection_audit.json \
  --round2-merge-audit logs/validation_hyperparameter_expansion_round2_merge_integrity_audit.json \
  --round3-plan logs/validation_hyperparameter_expansion_round3_plan.json \
  --output-dir staging/dev_trainmax_v1_hyperparameter_expansion_round3_shard_smoke \
  --audit logs/validation_hyperparameter_expansion_round3_shard_smoke.json
```

smoke audit 审核并提交、其 SHA 被正式 runner 绑定前，不得运行 741-shard round-3 sweep。审核前
不提交 smoke audit 或 staging，不生成 frozen hyperparameters。

### 6AB-final-release-check. Bind the reviewed Round-3 smoke

smoke audit 提交后，正式 runner 必须绑定其 SHA，再做最后一次零写入 release check：

```bash
PYTHONPATH=code python code/experiments/run_round3_incremental_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --union-audit logs/validation_hyperparameter_round2_union_selection_audit.json \
  --round2-merge-audit logs/validation_hyperparameter_expansion_round2_merge_integrity_audit.json \
  --round3-plan logs/validation_hyperparameter_expansion_round3_plan.json \
  --smoke-audit logs/validation_hyperparameter_expansion_round3_shard_smoke.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_expansion_round3_shards \
  --workers 8 \
  --release-check-only
```

状态必须为 `PASS_ready_for_round3_incremental_shard_execution`，并绑定 smoke audit SHA
`a0bf660f2b1452c68e312e5519ac102c9ec937e748172299e4f434ece53efb19`。正式 namespace 必须
仍不存在；移除 `--release-check-only` 前必须再次取得 researcher 明确授权。

### 6AF-pre. Round-3 shard integrity audit

741 个 Round-3 shard 全部完成后，先运行只读完整性审计。该 gate 用 runner 的规范
`_shard_set_sha256` 算法复核 completion marker，并逐 shard 验证 candidate domain、有限指标、
manifest、success marker、执行身份与文件 inventory；审计阶段不 merge。

```bash
PYTHONPATH=code python code/experiments/audit_round3_incremental_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --union-audit logs/validation_hyperparameter_round2_union_selection_audit.json \
  --round2-merge-audit logs/validation_hyperparameter_expansion_round2_merge_integrity_audit.json \
  --round3-plan logs/validation_hyperparameter_expansion_round3_plan.json \
  --smoke-audit logs/validation_hyperparameter_expansion_round3_shard_smoke.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_expansion_round3_shards \
  --audit logs/validation_hyperparameter_expansion_round3_shard_integrity_audit.json
```

状态必须为 `PASS_complete_pending_round3_incremental_merge`，规范 shard-set SHA 必须与 batch
completion marker 一致。审核前不提交 audit，不 merge、不运行 union selection、不冻结参数。

### 6AG-merge. Round-3 incremental integrity merge

Round-3 shard audit 经研究员审核并单独提交后，将 741 个 fragment 按 shard ID 与原行序事务性
合并为独立的 31,995 行 CSV。该步骤不物理联合原始、Round-1 或 Round-2 CSV；只在记录中声明
下一次 four-source streaming union 应读取 5,318,154 行。

```bash
PYTHONPATH=code python code/experiments/merge_round3_incremental_validation_sweep_shards.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --union-audit logs/validation_hyperparameter_round2_union_selection_audit.json \
  --round2-merge-audit logs/validation_hyperparameter_expansion_round2_merge_integrity_audit.json \
  --round3-plan logs/validation_hyperparameter_expansion_round3_plan.json \
  --smoke-audit logs/validation_hyperparameter_expansion_round3_shard_smoke.json \
  --shard-dir runs/dev_trainmax_v1/hyperparameter_expansion_round3_shards \
  --shard-audit logs/validation_hyperparameter_expansion_round3_shard_integrity_audit.json \
  --output-dir runs/dev_trainmax_v1/hyperparameter_expansion_round3 \
  --audit logs/validation_hyperparameter_expansion_round3_merge_integrity_audit.json
```

状态必须为 `PASS_complete_pending_round3_union_selection_audit`；正式目录只含 Round-3 CSV、
manifest、merge record 与 SUCCESS 四个文件。审核前不提交 merge audit，不运行 four-source union
selection，不生成 frozen hyperparameters。

### 6AH. Four-source streaming union-selection audit

Round-3 merge audit 经研究员审核并单独提交后，分别流读原始 5,157,360 行、Round-1 80,568 行、
Round-2 48,231 行和 Round-3 31,995 行 CSV，在同一 subject-balanced 聚合器中对 5,318,154 行
做选择。该 gate 不创建物理 union CSV；36 个未进入 Round-3 的选择必须与上一轮 union audit
逐参数一致，另外 12 个组的赢家只能是上一轮 union 赢家或预声明的 Round-3 local-shell candidate。

```bash
PYTHONPATH=code python code/experiments/audit_round3_validation_hyperparameter_union_selection.py \
  --root . \
  --config code/configs/generated/dev_trainmax_v1_validation_registered.yaml \
  --protocol code/configs/generated/dev_trainmax_v1_protocols/hyperparameter_sweep.yaml \
  --original-merge-audit logs/validation_sweep_merge_integrity_audit.json \
  --original-selection-audit logs/validation_hyperparameter_selection_audit.json \
  --round1-plan logs/validation_hyperparameter_expansion_plan.json \
  --round1-merge-audit logs/validation_hyperparameter_expansion_merge_integrity_audit.json \
  --round1-union-audit logs/validation_hyperparameter_union_selection_audit.json \
  --round2-plan logs/validation_hyperparameter_expansion_round2_plan.json \
  --round2-merge-audit logs/validation_hyperparameter_expansion_round2_merge_integrity_audit.json \
  --round2-union-audit logs/validation_hyperparameter_round2_union_selection_audit.json \
  --round3-plan logs/validation_hyperparameter_expansion_round3_plan.json \
  --round3-merge-audit logs/validation_hyperparameter_expansion_round3_merge_integrity_audit.json \
  --output logs/validation_hyperparameter_round3_union_selection_audit.json
```

仅当 `expanded_boundary_selection_count == 0` 时状态才是
`PASS_ready_for_hyperparameter_freeze`；否则状态为
`BLOCKED_expanded_boundary_requires_additional_shell`。审核前不提交 audit；无论哪种状态，本步骤
均不得生成 frozen hyperparameters、物理 union CSV，或删除任何既有 shard、merge 与 staging 产物。
