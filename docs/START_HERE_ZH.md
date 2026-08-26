# 实验人员开工说明：先完成 clean_v1 第一阶段

> **2026-08-26 更新：** provenance-only Phase 1 已结束，GPU 实验人员现在应
> 直接执行 [`GPU_WORKER_START_ZH.md`](GPU_WORKER_START_ZH.md)。本文件保留
> 背景规则；其中旧的 14/3/3、156/34/34 和 HE scanner-FOV 阻塞说明均已被
> development-only 16/4/0、179/45/0 方案与完整方形像素域算子取代。

请从这份说明开始，不要直接运行旧实验脚本，也不要使用旧 v3 的结果、split、
checkpoint、box cache 或超参数。

本阶段的目标不是立即得到论文数字，而是先建立一套可以审计的数据来源、划分、
物理尺度和模型来源。
在这些内容经过检查前，禁止接触新的 `test_locked` 数据。

## 一、取得最新版代码

```bash
git clone https://github.com/DiceLoss/SAM-KEM-VLM.git
cd SAM-KEM-VLM
git checkout main
git pull --ff-only origin main
git rev-parse HEAD
```

开始时的 commit 应至少包含：

```text
adfd121aaaaf259bd4cd5ad7741af55565f0ff19
```

如果仓库已经存在，只运行 `git pull --ff-only origin main`，不要覆盖本地尚未
提交的文件。

## 二、创建运行环境并检查代码

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate sam-kem
python -m pip install -e '.[test,evaluation]'
make quality
```

正确结果应为：

- Ruff 检查通过；
- Python compileall 通过；
- 51 项或更多测试全部通过；
- 所有 protocol YAML 解析通过。

然后运行一次发布审计：

```bash
make audit
```

此时 `make audit` 应当失败。
它会列出尚未提供的 split、固定尺度、MedSAM2、checkpoint、Qwen cache、sweep
和归档信息。
不要为了消除报错而填入旧 v3 的值或虚构内容。

## 三、创建自己的工作分支

```bash
git switch -c clean-v1-data-provenance
mkdir -p inputs/clean_v1
cp inputs/subject_inventory.template.csv inputs/clean_v1/subject_inventory.csv
cp inputs/source_activity_manifest.template.csv inputs/clean_v1/source_activity_manifest.csv
```

模板里的示例行和 `REPLACE_*` 都必须替换，不能原样保留。

## 四、第一批必须整理的信息

### 1. BrainWeb

请建立完整 subject/source-model 清单，并为每个实际使用的源文件记录：

- 稳定的 `subject_id`；
- `true_pet.nii.gz` 的准确路径和 SHA-256；
- simulation 与 source model 的对应关系；
- 使用的轴向 slice index 和预先确定的切片选择规则；
- 原始 PET 活度单位；
- `C_BrainWeb` 的数值、单位和可引用来源；
- MR 文件的对应关系，仅用于 MR-KEM/Bowsher-KEM 对比方法。

同一个 BrainWeb source model 的所有 simulation 和切片必须属于同一个 split。

### 2. HECKTOR

请记录：

- 全部病人的稳定 `subject_id`；
- `subject_id` 到 `HECKTOR.npz` axis-0 index 的准确映射；
- 完整 NPZ 文件 SHA-256；
- PET channel index、slice axis、slice index 及预先确定的切片选择规则；
- PET 数值单位；
- DICOM 直接导出为重建 PET 数组的流程及其软件/参数；
- 若仍可取得，记录 PixelSpacing、方向、位置和 rescale/SUV 标签用于来源追踪；
- CT 对应关系，仅用于 CT-KEM/Bowsher-KEM 对比方法。

当前上传的 HECKTOR 样本标记为 SUV，最大值约为 4.5--15.2。
因此不能把大约 30000 的 scanner-count 常数直接用于这些 SUV 数组。
如果 30000 来自机器的另一种原始输出，必须同时提供那种数据表示、单位和来源。
这些 DICOM 标签不用于拟合 scanner FOV；本研究正投影明确在完整方形像素网格上定义。

### 3. MedSAM2

请提供：

- 官方 MedSAM2 Git 仓库地址；
- 固定的 Git commit；
- 实际 source checkout 路径；
- released `MedSAM2.pt` 的下载/来源说明；
- checkpoint 文件 SHA-256；
- 用于保存新 checkpoint 和完整实验产物的不可变归档 URI。

BrainWeb 和 HECKTOR decoder 之后必须分别重训。
不能把 BrainWeb checkpoint 用于 HECKTOR，也不能把 released backbone 伪装成
训练后的 decoder。

## 五、先完成 subject-level split

在 `inputs/clean_v1/subject_inventory.csv` 中，每个 subject 只能出现一次。
`stratum` 只能是预先定义的 simulation family、scanner 或 site，不能根据实验
表现事后分组。

`previous_exposure_status` 必须如实填写：

- `development_allowed`：以前已经用于训练、验证、调参或结果查看；
- `never_used_for_training_validation_tuning_or_reporting`：从未被打开用于这些
  目的，可以进入新的 locked test 候选池。

旧 v3 test 已经在调参中暴露，不能继续称为 untouched test。

清单完成后运行：

```bash
sam-kem create-split --root . \
  --config code/configs/experiment.yaml \
  --inventory inputs/clean_v1/subject_inventory.csv \
  --output inputs/clean_v1/subject_split.csv

sam-kem file-sha --input inputs/clean_v1/subject_split.csv
```

人工检查：

- BrainWeb 为 train/validation/test = 14/3/3；
- HECKTOR 为 train/validation/test = 156/34/34；
- 同一 subject/source model 没有跨 split；
- 所有 `test_locked` 都声明为从未暴露；
- 各预设 stratum 尽可能均衡。

把输出 SHA 填入 `code/configs/experiment.yaml` 的 `data.split_sha256`，然后提交：

```bash
git add inputs/clean_v1/subject_inventory.csv \
  inputs/clean_v1/subject_split.csv \
  inputs/clean_v1/subject_split.csv.manifest.json \
  code/configs/experiment.yaml
git commit -m "Record clean subject inventory and split"
```

如果没有足够的从未暴露 subject，请立即停止并报告，不能把旧测试病例重新命名
为新测试集。

## 六、填写 source activity manifest

使用 `inputs/clean_v1/source_activity_manifest.csv`，每一行对应一个明确的
source/subject/slice。
必须填写完整 SHA、原始单位、source selector 和空间变换。

文件 SHA 可以用：

```bash
sha256sum /absolute/path/to/source_file
```

BrainWeb 只允许：

```text
mean_pool_2x2_full_square
```

HECKTOR 使用完整方形切片变换：

```text
resize_full_square_to_128
```

这里的 144→128 是预先声明的像素中心对齐双线性缩放，必须保留完整方形；禁止
144→90 中心裁剪、圆形角落掩膜、逐图 min-max 或根据图像内容移动视野。

## 七、第一阶段的停止点

完成上述材料后先不要训练模型、不要调用 Qwen、不要生成 test 数据，也不要跑
comparison。

请提交并发送以下内容供审查：

1. 当前 Git commit；
2. `inputs/clean_v1/subject_inventory.csv`；
3. `inputs/clean_v1/subject_split.csv` 及 SHA-256；
4. `inputs/clean_v1/source_activity_manifest.csv`；
5. BrainWeb 和 HECKTOR 的固定 C 数值、单位、推导/机器来源；
6. HECKTOR 的 DICOM 直接导出流程、PET 单位以及切片/通道映射；
7. MedSAM2 仓库、commit、base checkpoint SHA；
8. 外部不可变归档位置；
9. `make quality` 的完整输出；
10. `make audit` 的完整 blocker 输出。

如果 source manifest 含患者标识或受限绝对路径，请通过项目批准的安全渠道提交
原件，并为 GitHub 制作保留 SHA、单位、selector 和几何信息的去标识版本。
不要因去标识而删除复现实验所需的技术字段。

只有这一批内容通过审查后，才进入 train/validation ingestion、规范化敏感性
实验、Qwen box cache 和 decoder 训练。

## 八、禁止事项

- 禁止运行 `code/scripts/_*.py`、`code/src/v3_canonical.py` 或
  `code/src/original/*.py`；
- 禁止使用 `subject_split_v3*.csv`、`subject_split_v4.csv`；
- 禁止使用 `results/*_v3.csv` 生成论文表格；
- 禁止从旧结果复制 alpha、iteration、sigma、radius 等超参数；
- 禁止训练或调参阶段读取 `test_locked`；
- 禁止逐图 min-max 作为主实验输入/target 归一化；
- 禁止使用 CT、MR 或肿瘤 mask 作为我们方法的输入或评价 mask；
- 禁止手改生成的 CSV/JSON 数字，或手写 `_SUCCESS.json`；
- 禁止把 API key、患者信息、checkpoint 或大体积原始数据提交到 GitHub。

完整后续命令见 [`EXECUTION.md`](EXECUTION.md)，异常恢复和归档规则见
[`OPERATIONS_RUNBOOK.md`](OPERATIONS_RUNBOOK.md)。
