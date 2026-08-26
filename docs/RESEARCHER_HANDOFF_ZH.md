# 给实验人员的 clean_v1 交接说明

## 先说结论

当前仓库已经允许 development-only GPU 训练，具体命令以
[`GPU_WORKER_START_ZH.md`](GPU_WORKER_START_ZH.md) 为准。发布审计仍应失败，
因为没有未暴露测试集、训练后 decoder、验证 sweep 和不可变归档。

旧 `results/*_v3.csv`、旧 checkpoint 哈希、旧 box 记录和旧 split 都不能写入
新论文结果。
不要为了让 validator 变绿而把旧值复制到 `experiment.yaml`。

## 你需要交付的原始信息

### BrainWeb

- 全部源模型/subject 的稳定 ID；
- 每个 `true_pet.nii.gz` 的绝对来源说明和 SHA-256；
- simulation/source-model 的关系，确保同一源模型不会跨 train/validation/test；
- 原始活度单位及固定尺度 `C_BrainWeb` 的物理或仿真定义来源；
- 使用哪些轴向切片，以及切片选择规则是否在看结果前确定；
- MR 解剖图像的对应来源，仅供 MR-KEM/Bowsher-KEM 对比方法使用。

### HECKTOR

- 全部病人的稳定 ID 与 NPZ 轴 0 映射；
- 完整 `HECKTOR.npz` 的 SHA-256、PET 通道和轴向切片选择规则；
- PET 数值单位。当前上传样本标记为 SUV，不能直接使用约 30000 的
  scanner-count 常数；
- 若可取得，保留 DICOM spacing/orientation/rescale 标签用于来源和单位追踪；
- CT 对应来源，仅供 CT-KEM/Bowsher-KEM 对比方法使用；
- 证明 PET/CT 来自一体机配准流程的资料可用于论文说明，但不要用 CT 或肿瘤
  mask 定义评价区域。

### MedSAM2 与外部归档

- 官方 MedSAM2 源仓库、固定 Git commit、本地 checkout；
- released `MedSAM2.pt` 的准确来源、SHA-256 和保存位置；
- 可以长期保存数组、PNG、checkpoint、日志和结果的只读/版本化归档 URI；
- DashScope key 只通过环境变量提供，禁止写入仓库。

## 必须按这个顺序执行

1. 创建环境并运行 `make quality`。
2. 使用已固定的 development-only subject split：BrainWeb 16/4/0，HECKTOR
   179/45/0；其中没有 `test_locked`。
3. 完成 `source_activity_manifest.csv`，只导入 train 和 validation。
4. 明确两个数据集的固定 C；如果物理 C 仍不清楚，只能在 train/validation 上
   做规范化敏感性研究，不能查看 locked test。
5. 生成匹配的 180x182 projection、独立 Poisson sinogram 和 MLEM-10。
6. 对每个 `(dataset, subject, slice, count)` 只调用一次 Qwen，输入是 r=0
   的 MLEM-10；同一 box 供 r=0/1/2 使用。失败调用保留记录并用质心框 fallback。
7. BrainWeb 和 HECKTOR 各训练一个四剂量联合模型；冻结 image/prompt/memory
   encoder，只训练 mask decoder，loss 只用 L1。
8. 仅在 validation 上跑完整 sweep，包括 alpha=0.0--0.5；边界最优必须扩展
   网格。然后生成 frozen hyperparameters。
9. 锁定配置并通过 test-release gate 后，才导入和处理 test_locked。
10. 依次跑 comparison、从 comparison 派生 primary、真实 ablation、
    convergence、cross-count 和 validation runtime。
11. 完整归档，填写 archive URI，运行 `make quality` 和 `make audit`。

所有具体命令见 [`EXECUTION.md`](EXECUTION.md)。

## 关键科学口径

- 模拟为独立二维环位置的理想平行束低计数模型，不是 3D total-body PET。
- `y ~ Poisson(s A x)` 表示衰变/探测的计数统计；不要写成热噪声。
- count 越低意味着统计噪声相对越强，但这里只是 expected detected-count
  proxy，不等价于校准后的注射剂量。
- HECKTOR 从临床重建 PET 做 Radon 后得到的是 pseudo-sinogram，不是机器原始
  list-mode/sinogram，也不是真正无噪声活度真值。
- 相邻切片可以作为独立二维模拟案例，但划分和统计汇总必须在 subject 层面，
  不能把同一病人的切片当成独立病人。
- 评价 mask 使用 PET 算子正灵敏度的完整方形支持；禁止使用 CT、MR 或肿瘤
  mask 改变评价区域。
- 我们的方法不使用 CT/MR；MR-KEM、CT-KEM、Bowsher-KEM 是明确命名的
  解剖对比方法。
- `Bowsher-KEM` 是解剖邻域选择的 coefficient-space KEM，不要声称它是经典
  MAP-Bowsher。

## 规范化怎么处理

主实验目标是统一固定物理尺度：每个数据集使用一个有单位和来源的 C，输入与
target 同时除以它，不做逐图 min-max。
SAM prior 作为 kernel feature 时才按每个 prior 做 min-max。

如果 C 暂时拿不到，允许研究以下候选，但必须分开训练和验证：

- 有来源的物理 C；
- 只用训练集求出的全局最大值作为固定 C；
- 只用训练集拟合的 global min-max，并在验证阶段记录 clipping；
- per-image min-max 只作为诊断，不得作为主实验。

每种候选必须有不同 normalization ID、数组、checkpoint、sweep 和输出目录。
不能只看强度分布就宣布某种方法最好，也不能用 test 选择。

## 每个阶段必须保存什么

- 源数据 SHA、单位、选择器和空间几何；
- split CSV 及 SHA；
- 每个 NPY 的 JSON sidecar 和数组 SHA；
- 每次 Qwen 的 request ID、输入图 SHA、prompt SHA、解析状态和 fallback；
- decoder inference checkpoint、resume state、training CSV、参数冻结清单和
  provenance JSON；
- validation 全 sweep、frozen hyperparameters、边界扩展记录；
- 每例结果、subject-level 配对统计、环境、GPU/CUDA 和 run metadata；
- `_SUCCESS.json` 与外部不可变归档 URI。

目录存在不表示完成；只有 sidecar、SHA 和 `_SUCCESS.json` 全部验证通过才算完成。

## 遇到失败时

- 立即停止下游实验，不要手写 `_SUCCESS.json`；
- 保存最早失败 case、报错、commit、config/protocol SHA 和部分目录；
- 只有支持 `--resume-verified` 的命令才能续跑；
- Qwen 的失败记录不能因为结果不好而选择性重试；
- 科学设置有变化时换新的 run ID，并重跑所有受影响下游步骤；
- 禁止手改 CSV 数字、复制旧 checkpoint 哈希或把 partial 文件夹改名为 final。

## 最终交付检查

```bash
git status --short
git rev-parse HEAD
make quality
make audit
```

最终请同时提交：Git commit、锁定配置、所有输入 manifest、split SHA、两个
checkpoint bundle、Qwen cache、validation sweep/frozen parameters、各协议
`_SUCCESS.json`、完整归档 URI 和 deviation/fallback 清单。
