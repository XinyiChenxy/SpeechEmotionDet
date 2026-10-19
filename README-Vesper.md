# 基于官方 Vesper 实现的 CREMA-D 情绪分类课程项目

复用 [HappyColor/Vesper](https://github.com/HappyColor/Vesper) 的真实编码器及下游表示加权方式，对 CREMA-D 做六分类。仅包含演员隔离的训练/验证/测试、最佳模型评估及 WAV/MP3 命令行预测，不涉及模型比较、交叉验证或从头预训练，也不宣称严格复现[论文](https://arxiv.org/abs/2307.10757)分数。

## 当前状态

2026-10-16 在当前容器实测：Python 3.10.11，PyTorch 2.0.1 / CUDA 11.7，RTX 3090 Ti 24 GB；torchaudio、soundfile、ffmpeg 等依赖可用。已有数据、压缩包和 Docker 文件均保留。

数据已完整解码检查并生成 `data/manifest.csv`，共 7,442 条、91 位演员，无损坏音频或 LFS 指针；`1040_ITH_SAD_X.wav` 是有效 SAD 音频，保留原名。seed=42 的划分：

| 划分 | 演员 | 样本 | ANG | DIS | FEA | HAP | NEU | SAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 64 | 5229 | 893 | 893 | 893 | 893 | 764 | 893 |
| val | 14 | 1148 | 196 | 196 | 196 | 196 | 168 | 196 |
| test | 13 | 1065 | 182 | 182 | 182 | 182 | 155 | 182 |

三组演员无重叠且覆盖六类。**目前缺真实的 Vesper 预训练权重，因此没有启动正式训练，也没有生成模型分数。** 官方 README 只有权重路径占位符，[Releases](https://github.com/HappyColor/Vesper/releases) 和 tags 为空，[issue #2](https://github.com/HappyColor/Vesper/issues/2)、[issue #3](https://github.com/HappyColor/Vesper/issues/3) 及其评论未给出可用权重下载链接；未能核实任何作者发布的 Vesper-4 或 Vesper-12 文件。公开页面访问正常，但没有权重 URL 可测试下载。本地也没有相应权重。

缺少的文件是作者预训练完成的 **Vesper-4 checkpoint**，预期为 `checkpoints/Vesper-4.pt`，或通过 `--checkpoint` 指定。项目不会用随机编码器、WavLM 权重替代它。若后续只有可验证的 Vesper-12 权重可用，将配置中的 `model.version` 改为 `Vesper-12`，同时修改权重路径与输出目录；层数/形状不匹配会报错。结构兼容检查本身不能证明来源，权重应来自作者或可核验的作者发布渠道。

## 安装与数据

在当前容器无需重新安装依赖。从其他兼容环境安装：

```bash
cd /workspace
python -m pip install -r requirements.txt
```

非容器环境还需安装 `ffmpeg`、`libsndfile`、`git`，下载官方数据时需 `git-lfs`。现有 Docker 构建说明见 [README-Docker.md](README-Docker.md)。

现有音频在 `data/AudioWAV/`，不要重复下载。更换数据位置时修改 `configs/cremad.yaml` 的 `data.root`。没有数据时可获取[官方 AudioWAV](https://github.com/CheyneyComputerScience/CREMA-D)，只取 WAV：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CheyneyComputerScience/CREMA-D.git data/CREMA-D
git -C data/CREMA-D lfs pull --include='AudioWAV/*.wav' --exclude=''
# 然后将 data.root 改为 data/CREMA-D/AudioWAV
```

也可自行下载 [Kaggle CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) 后将音频放入 `data/AudioWAV/`，不需要向项目提供凭据。官方 ZIP 内的音频可能只是 Git LFS 指针，必须下载真实文件。数据集的使用条款见其[独立许可证](https://github.com/CheyneyComputerScience/CREMA-D/blob/master/LICENSE.txt)。

```bash
python -m ser.vesper prepare --config configs/cremad.yaml
```

manifest 包含绝对音频路径、演员 ID、情绪代码/ID/名称、原始时长（秒）、split。优先使用 WAV；同一条只有 MP3 时正常解码。以文件名 intended emotion 为标签，固定映射为：`ANG=0 angry, DIS=1 disgust, FEA=2 fear, HAP=3 happy, NEU=4 neutral, SAD=5 sad`，没有 surprise。按演员固定种子随机分组，先划分再增强，无需拟合全局预处理统计量。发现损坏、过短或 LFS 指针时默认拒绝生成 manifest，明细写入 `data/audio_audit.json`；只有显式 `--skip-invalid` 才跳过。

已有 manifest 默认保留；有意更换数据/种子/划分配置时使用 `prepare --force`。迁移项目到其他路径也应重新生成 manifest。重新生成后不能与旧训练断点混用；评估和恢复训练会核对 manifest SHA256。

## 权重核对与训练

上游源码固定于 `2b62af3e15fbd526fac6d4f1d6fea279c306244d`，完整放在 `third_party/Vesper/`，附原许可证和 Git 来源。[THIRD_PARTY.md](THIRD_PARTY.md) 记录实际结构、复用范围及 padding 修正。

所有超参数位于 `configs/cremad.yaml`，相对路径按项目根目录解析。默认 Vesper-4、batch size 16、首 3 秒、SGD 1e-4、50 epochs，适配本容器 24 GB 显存。与官方下游默认方式一致，冻结编码器并学习各层权重、投影和六分类头；冻结编码器保持 eval 模式。`model.freeze_upstream: false` 可更新 Transformer，此时需自行减小 batch size。可选白噪声增强仅用于 train，默认关闭。

取得真实权重后依次执行：

```bash
python -m ser.vesper check-checkpoint --config configs/cremad.yaml --checkpoint /path/to/Vesper-4.pt
python -m ser.vesper smoke --config configs/cremad.yaml --checkpoint /path/to/Vesper-4.pt
python -m ser.vesper train --config configs/cremad.yaml --checkpoint /path/to/Vesper-4.pt
```

`check-checkpoint` 核对 `model` 下的 `vesper.*` 参数、层数、形状，严格加载全部编码器参数并记录权重 SHA256；仅去掉官方下游同样忽略的 mask embedding 和输出 layer norm 参数。`smoke` 使用真实训练样本完成前向、反向、分类头更新、保存/重新加载及单文件预测，并检查不同 padding 长度下的结果一致性；缺真实权重时立即失败，不运行替代模型。

按 **val Macro-F1 最高值**保存 `best.pt`，按验证集进展 early stopping。测试集只在训练结束后评估，不参与最佳模型选择。`last.pt` 保存最近完整 epoch 的模型、优化器、余弦调度器、AMP scaler、随机数状态和 early stopping 状态；崩溃时当前未完成 epoch 需重新运行。保持原配置与原目录恢复：

```bash
python -m ser.vesper train --config configs/cremad.yaml --resume outputs/cremad_vesper4/last.pt
```

新训练必须使用空输出目录，避免覆盖已有结果。训练结束自动用 `best.pt` 评估 val/test。断点包含完整编码器、分类头、层权重、标签、预处理及训练配置，预测不依赖原始预训练文件。

后台训练（先校验权重和 manifest，日志持久保存；容器必须保持运行）：

```bash
bash scripts/train_background.sh configs/cremad.yaml /path/to/Vesper-4.pt
tail -f outputs/cremad_vesper4/train.log
# 启动脚本会同时给出包含 tqdm 输出的 outputs/train_<UTC时间>.log 和 PID
```

## 评估和单文件预测

```bash
python -m ser.vesper evaluate --checkpoint outputs/cremad_vesper4/best.pt --split test
python -m ser.vesper evaluate --checkpoint outputs/cremad_vesper4/best.pt --split val
python -m ser.vesper predict --checkpoint outputs/cremad_vesper4/best.pt --audio data/AudioWAV/1001_DFA_ANG_XX.wav
python -m ser.vesper predict --checkpoint outputs/cremad_vesper4/best.pt --audio /path/to/sample.mp3 --output outputs/prediction.json
```

预测返回预测情绪、六类概率和实际处理秒数。训练、评估、预测共同使用单声道、16 kHz、取首 3 秒、有效样本 layer_norm 的预处理。变长批次在有效样本之外补零，用精确卷积长度屏蔽 attention 和汇聚；短于 400 个 16 kHz 样本的音频拒绝输入。长音频只识别开头 3 秒，与训练一致。

评估输出 Accuracy、Macro-F1、UAR（六类 recall 的算术平均）、每类 precision/recall/F1/support、混淆矩阵，全部分数为 0–1。输出目录：

- `data/manifest.csv`、`data/manifest.summary.json`、`data/audio_audit.json`：数据及划分审计。
- `outputs/environment.json`、`outputs/weight_availability*.json`：环境及权重可用性证据。
- `outputs/unit_tests.log`、`outputs/validation_status.json`：已执行验证与阻塞项。
- `outputs/cremad_vesper4/{best.pt,last.pt,config.json,labels.json,train.log,history.jsonl}`：正式训练后生成。
- `outputs/cremad_vesper4/{val,test}_metrics.json`、`*_predictions.csv`、`*_confusion_matrix.csv/png`：真实评估后生成。
- `outputs/smoke/{smoke.pt,report.json}`：真实权重 smoke test 通过后生成，不能当作正式训练结果。

## 验证

```bash
python -m unittest discover -s tests -v
python scripts/check_weight_availability.py
```

离线测试覆盖声道/重采样/MP3、LFS/损坏/过短音频、演员隔离及固定划分、卷积有效长度、带 bias 投影后的 padding 汇聚与梯度、拒绝错误 checkpoint、指标/图像导出。测试使用临时音频和手工特征张量，不训练随机编码器，也不向 `outputs` 写伪造评估分数。当前真实模型的严格加载、前向/反向、保存重载和正式训练尚未执行，唯一必要输入为可验证的预训练 Vesper 权重。
