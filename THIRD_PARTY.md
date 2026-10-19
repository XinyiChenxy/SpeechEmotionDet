# emotion2vec（当前项目）

- 论文：https://aclanthology.org/2024.findings-acl.931/
- 作者仓库：https://github.com/ddlBoJack/emotion2vec
- 完整保留于 `third_party/emotion2vec`，commit `a0e606c668227ab0c01527e196a4acc42b1efad9`。该快照根目录没有单独的 LICENSE 文件，未替作者添加或推定许可证。
- 使用的预训练模型：https://huggingface.co/emotion2vec/emotion2vec_base ，revision `0c9a3152734f9d7a7a05b4ee6bfb3c109d288664`。原模型卡、配置及下载校验记录保存在 `checkpoints/emotion2vec_base`。模型卡标记为 `model-license`，链接 FunASR；许可独立于项目代码。
- 实际执行的编码器来自官方推荐的 FunASR 接口：固定 PyPI `funasr==1.2.6`，直接调用其 `funasr.models.emotion2vec.model.Emotion2vec` 类。MIT 代码许可和模型许可原文保存在 `third_party/FunASR_licenses/`；其中许可文本下载自 FunASR 主分支，具体 URL 见其仓库，不将该文本快照冒称 1.2.6 版许可。发行 wheel SHA256 和实际执行文件 SHA256 见该目录 `provenance.json`。

`ser/emotion2vec.py` 仅适配数据、缓存和下游分类头，没有重写编码器。模型配置通过官方使用的 OmegaConf 解析（避免普通 YAML 将 `1e-05` 读成字符串），193 个参数项全部严格匹配。默认逐条完整音频做 16 kHz 单声道与 layer_norm，关闭 mask，复用官方 `extract_features`，移除模型额外 token 后对最后一层帧表示取均值。编码器始终冻结、eval、inference_mode。下游为 Linear `768→6`，可配置 MLP `768→256→6`；MLP 作用于已汇聚的整句特征，不等同于作者 IEMOCAP 逐帧非线性投影再汇聚的头，未宣称完全复现该头。

数据源仍为 CREMA-D。现有演员隔离划分及固定六类映射保持；特征标准化只拟合训练集。论文的预训练数据描述见其表 1/附录 B.1；本项目没有重新预训练模型，没有把 emotion2vec+ 的九分类结果映射为六类。

# Vesper（保留的历史实现）

- Repository: https://github.com/HappyColor/Vesper
- Commit: `2b62af3e15fbd526fac6d4f1d6fea279c306244d`
- Location: `third_party/Vesper` (完整 Git clone；原始文件未修改)
- License: [上游 Apache-2.0 原文](third_party/Vesper/LICENSE)。保留源文件内原有版权及其他许可证声明。
- Paper: https://arxiv.org/abs/2307.10757
- Dataset: https://github.com/CheyneyComputerScience/CREMA-D （数据集许可独立于代码许可）

项目直接加载 `models/vesper.py` 和 `configs/model_config.py`，复用官方 CNN、Transformer、相对位置注意力、`VesperFinetuneWrapper._weighted_sum`。直接按文件导入是因为上游 `models/__init__.py` 引用不存在的 `models.utils`。不运行上游预训练入口。

架构来自官方配置：7 层卷积，1024 维表示，Vesper-4 为 4 层 Transformer（FFN 4096，16 heads），位置卷积 128、16 groups，320 relative buckets；Vesper-12 使用 12 层。下游采用 CNN 输出与各 Transformer 层输出的 softmax 可学习加权和，线性投影 `1024→256`、有效帧均值、线性分类 `256→6`。默认冻结编码器/CNN，使用 SGD、动量、余弦学习率与交叉熵。该默认下游方式训练层权重和分类头，不更新 Vesper 编码器；`freeze_upstream: false` 可训练 Transformer。

适配差异全部在 `ser/`，不修改作者文件：

1. 逐条音频转单声道/16 kHz，保留官方 CREMA-D 首 3 秒裁剪；官方波形 layer_norm 提到 padding 之前，避免补零改变均值/方差。
2. 用真实 7 层卷积公式计算有效帧数，替换上游通过 reshape 近似换算 padding mask 的方式。
3. 在带 bias 的 projector 之后屏蔽补齐帧，修正上游先置零再投影导致的 bias 累加。
4. 冻结编码器保持 eval 模式；仅下游权重/分类头更新。无全局拟合预处理统计量。可选白噪声增强仅用于已划分的训练集，默认关闭。
5. 按演员划分 70/15/15，使用独立验证集 Macro-F1 选择模型和 early stopping；上游 CREMA-D 数据加载器采用样本随机划分，因此不使用该加载器。
6. 自有单 GPU/CPU 训练入口代替上游 DDP 启动流程，保留严格加载、完整断点、推理配置和分类标签。

权重加载复用作者的 `model['vesper.*']` 约定；允许去除 DDP `module.` 前缀。只忽略作者下游加载同样忽略的 `mask_emb` 和 `encoder.layer_norm.{weight,bias}`。其余 encoder 参数全部 `strict=True`；层数或形状不匹配立即失败。分类头重新训练。格式/shape 检查无法证明第三方文件是否真正经过作者预训练，使用者仍需确认来源；项目不会把 WavLM 转换成 Vesper，也不自动从头预训练。
