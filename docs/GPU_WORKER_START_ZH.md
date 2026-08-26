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

必须生成两个训练候选，不允许用 per-image min-max 作为主实验：

```bash
sam-kem generate-candidate-config \
  --root . --base-config code/configs/experiment.yaml \
  --candidates runs/clean_v1/normalization_calibration/normalization_candidates.csv \
  --candidate train_max_fixed_C --run-id dev_trainmax_v1 \
  --output code/configs/generated/dev_trainmax_v1.yaml \
  --protocol-output-dir code/configs/generated/dev_trainmax_v1_protocols

sam-kem generate-candidate-config \
  --root . --base-config code/configs/experiment.yaml \
  --candidates runs/clean_v1/normalization_calibration/normalization_candidates.csv \
  --candidate train_global_minmax --run-id dev_globalminmax_v1 \
  --output code/configs/generated/dev_globalminmax_v1.yaml \
  --protocol-output-dir code/configs/generated/dev_globalminmax_v1_protocols

git add code/configs/generated
git commit -m 'Freeze development normalization candidate configs'
```

## 4. 为每个候选生成训练输入和 Qwen box

下面代码块分别执行两次，把 `RUN` 依次设为 `dev_trainmax_v1` 和
`dev_globalminmax_v1`。

```bash
RUN=dev_trainmax_v1
CFG=code/configs/generated/${RUN}.yaml

sam-kem ingest-activity --root . --config "$CFG" \
  --manifest inputs/clean_v1/source_activity_manifest.csv \
  --output "arrays/${RUN}" --splits train validation

sam-kem prepare-cases --root . --config "$CFG" \
  --arrays "arrays/${RUN}" --splits train validation

sam-kem build-manifest --root . --config "$CFG" \
  --arrays "arrays/${RUN}" \
  --output "results/${RUN}/case_manifest_inputs.csv"

export DASHSCOPE_API_KEY='服务器环境变量中的真实 key'
sam-kem prepare-prompt-images --root . --config "$CFG" \
  --manifest "results/${RUN}/case_manifest_inputs.csv" \
  --output "prompts/${RUN}/train_validation_images" \
  --splits train validation
sam-kem request-boxes --root . --config "$CFG" \
  --manifest "prompts/${RUN}/train_validation_images/prompt_manifest.csv" \
  --responses "prompts/${RUN}/train_validation_responses"
sam-kem build-box-cache --config "$CFG" \
  --manifest "results/${RUN}/case_manifest_inputs.csv" \
  --responses "prompts/${RUN}/train_validation_responses" \
  --output "prompts/${RUN}/train_validation_box_cache.csv"

git add "results/${RUN}" "prompts/${RUN}"
git commit -m "Prepare ${RUN} decoder inputs and boxes"
```

每个 `(dataset, subject, slice, count)` 只调用一次 Qwen，使用 r=0 MLEM-10；
r=0/1/2 共用该 box。失败调用必须保留，并使用已实现的质心 fallback，禁止挑选性
重试。

## 5. GPU 开工门和正式训练

每个候选只运行下面一条命令。它先执行真实 CUDA 模型构造、512 网格 box、前向、
L1、反向和冻结参数检查；任何一步失败都不会开始长训练。通过后依次训练 BrainWeb
和 HECKTOR，各自一个四剂量联合 decoder。

```bash
mkdir -p logs
python code/experiments/run_gpu_training.py \
  --config code/configs/generated/dev_trainmax_v1.yaml \
  2>&1 | tee logs/gpu_dev_trainmax_v1.log

# 固定第一候选的小型 provenance/metrics；二进制 .pt 仍由 .gitignore 排除。
git add checkpoints/dev_trainmax_v1/*.json \
  checkpoints/dev_trainmax_v1/*.training.csv \
  checkpoints/dev_trainmax_v1/*.training.csv.manifest.json
git commit -m 'Record dev_trainmax_v1 decoder training provenance'

python code/experiments/run_gpu_training.py \
  --config code/configs/generated/dev_globalminmax_v1.yaml \
  2>&1 | tee logs/gpu_dev_globalminmax_v1.log

git add checkpoints/dev_globalminmax_v1/*.json \
  checkpoints/dev_globalminmax_v1/*.training.csv \
  checkpoints/dev_globalminmax_v1/*.training.csv.manifest.json
git commit -m 'Record dev_globalminmax_v1 decoder training provenance'
```

训练协议固定为 20 epochs、每 epoch 400 steps、AdamW、学习率 `1e-4`、batch 1、
L1-only。image/prompt/memory encoder 全部冻结，只更新 mask decoder。中断后重新执行
同一命令会读取该 checkpoint 对应的 `.resume.pt`，不得另建临时 checkpoint。

## 6. GPU 完成后立即交回

对两个候选、两个数据集分别交回：

- `medsam2_decoder_*.pt`；
- `.resume.pt`；
- `.json` provenance；
- `.training.csv` 及 manifest；
- 两份 `logs/gpu_*.log`；
- `nvidia-smi`、Python、PyTorch、CUDA 版本；
- 所有文件 SHA-256 和不可变归档 URI。

不要自行选择 normalization 胜者，不要运行 test，不要修改 epochs、学习率、loss、
box 或 checkpoint。GPU 训练完成后停止，等待主研究员注册 checkpoint SHA 并启动
validation sweep。
