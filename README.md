# 冻结 emotion2vec 的 CREMA-D 情绪分类课程项目

使用用户指定的 [emotion2vec/emotion2vec_base](https://huggingface.co/emotion2vec/emotion2vec_base)，冻结官方预训练编码器，提取 768 维情绪特征，再训练 Linear 六分类头。支持 MLP 配置选项；本次仅训练 Linear，不进行模型对比。项目没有重新预训练或微调编码器，不宣称复现 [ACL 2024 论文](https://aclanthology.org/2024.findings-acl.931/)分数。

## 已完成的真实运行

2026-10-16 在当前容器 RTX 3090 Ti 24 GB / PyTorch 2.0.1 / CUDA 11.7 上完成：官方权重下载及 SHA256 校验、全部参数严格加载、真实音频 smoke test、7,442 条音频特征提取、正式分类头训练、验证集选模、测试集评估、WAV/MP3 单文件预测。

| 划分 | 演员数 | 样本数 | Accuracy | Macro-F1 | UAR |
|---|---:|---:|---:|---:|---:|
| train | 64 | 5229 | — | — | — |
| val | 14 | 1148 | 74.22% | 74.34% | 74.50% |
| test | 13 | 1065 | 70.05% | 69.92% | 70.50% |

固定 seed=42，三组演员无重叠且均包含六类。只按 **val Macro-F1** 保存最佳模型：第 39 轮最佳，第 54 轮触发 early stopping。测试集仅在训练结束后评估，没有用其选择模型。现有音频、压缩包和 Docker 文件保留。

## 环境与权重

当前独立环境已配置好；进入后执行后续命令：

```bash
cd /workspace
source .venv-emotion2vec/bin/activate
```

若需重建该环境，先使用现有 Docker/PyTorch 环境（见 [README-Docker.md](README-Docker.md)）：

```bash
python -m venv --system-site-packages .venv-emotion2vec
.venv-emotion2vec/bin/python -m pip install -r requirements-emotion2vec.txt
source .venv-emotion2vec/bin/activate
python scripts/download_emotion2vec.py
python -m ser check-checkpoint
```

权重已放在 `checkpoints/emotion2vec_base/`。固定 Hugging Face revision：`0c9a3152734f9d7a7a05b4ee6bfb3c109d288664`；`emotion2vec_base.pt` 为 1,125,606,009 字节，SHA256 为 `4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd`。下载器校验完整性并保存 `provenance.json`，不需要账户凭据。

直接复用固定版 **FunASR 1.2.6** 的官方 Emotion2vec 类，按仓库 `config.yaml` 构造并 `strict=True` 加载全部 193 个参数项。编码器约 93.79M 参数，保持 `eval()` 和 `requires_grad=False`，只训练下游头。使用的是 base 特征模型，不是 emotion2vec+ 或其九分类输出。

## 数据与特征

当前已有 `data/AudioWAV/`，无需重新下载。若没有数据，可从[官方 CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)取得 AudioWAV（注意真实音频需要 Git LFS），或解压 [Kaggle 数据](https://www.kaggle.com/datasets/ejlok1/cremad)到该目录；修改 `configs/emotion2vec.yaml` 的 `data.root` 即可使用其他位置。

```bash
# 仅在没有数据时下载官方 WAV，避免视频及 MP3 重复下载
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CheyneyComputerScience/CREMA-D.git data/CREMA-D
git -C data/CREMA-D lfs pull --include='AudioWAV/*.wav' --exclude=''
# 然后将 data.root 改为 data/CREMA-D/AudioWAV
```

```bash
python -m ser prepare
python -m ser extract
```

- 复用 `data/manifest.csv` 中的演员分组；默认保留已有划分。明确更换数据/种子时才用 `prepare --force`。
- manifest 包含绝对路径、演员 ID、情绪代码/名称/ID、时长、split。训练与评估核对 manifest 和音频状态，防止旧缓存混入新划分。
- 使用文件名 intended emotion：`ANG=0 angry, DIS=1 disgust, FEA=2 fear, HAP=3 happy, NEU=4 neutral, SAD=5 sad`；没有 surprise。强度后缀为 X 的 `1040_ITH_SAD_X.wav` 是有效 SAD 音频，保留原名。
- 优先 WAV，也支持 MP3。完整解码检查发现损坏、空文件、过短音频或 Git LFS 指针时拒绝生成 manifest；明细在 `data/audio_audit.json`。仅显式 `--skip-invalid` 才跳过。
- 统一转单声道、16 kHz；**默认保留完整音频**，逐条做官方 layer_norm，再提取最后一层帧特征并取时间均值。逐条提取不引入 padding，训练与预测共用预处理。
- 特征缓存为 `data/features/emotion2vec_base.pt`，另有逐条 `.npy` 缓存供中断后继续。完成缓存后再次运行 `extract` 只做一致性核对；数据或预处理变化后需要 `extract --force`。
- 特征标准化的均值/标准差**只从训练集**拟合，随分类头保存；val/test 只应用既有统计量。冻结特征方案不使用数据增强。

## 训练、恢复与后台运行

全部配置在 `configs/emotion2vec.yaml`，相对路径以项目根目录解析。默认 Linear `768→6`，AdamW，batch size 128，学习率 1e-3，最多 100 epochs，patience 15。

```bash
python -m ser train
# 或一次完成缓存提取、训练和最终评估
python -m ser run
# 从最近完整 epoch 恢复，保持原配置和输出目录
python -m ser train --resume outputs/cremad_emotion2vec_linear/last.pt
```

**现有正式结果已存在**，新训练会拒绝覆盖。重新训练时先将配置中的 `train.output_dir` 改为新目录。若改用 MLP，设置 `model.head: mlp`，同时换输出目录；无需重新提取特征。

`best.pt` 保存验证集 Macro-F1 最高的分类头；`last.pt` 包含模型、标准化统计量、标签、编码器来源/配置、优化器、调度器、随机数及 early stopping 状态。仅保存每个完成的 epoch，未完成的 epoch 会在恢复时重跑。

```bash
# 在自己的容器终端启动；容器须保持运行
bash scripts/run_emotion2vec_background.sh configs/emotion2vec.yaml
# 脚本打印 PID 和日志位置
```

## 评估与单文件预测

```bash
python -m ser evaluate --checkpoint outputs/cremad_emotion2vec_linear/best.pt --split test
python -m ser predict --checkpoint outputs/cremad_emotion2vec_linear/best.pt --audio data/AudioWAV/1001_DFA_ANG_XX.wav
python -m ser predict --checkpoint outputs/cremad_emotion2vec_linear/best.pt --audio /path/to/sample.mp3 --output outputs/prediction.json
```

预测输出六类概率和预测情绪。支持 `--device cpu`；迁移权重目录后可用 `--encoder-dir /path/to/emotion2vec_base`。**推理需要分类头 best.pt 和官方编码器目录两部分**；分类头不重复打包 1.1 GB 的公开权重，但保存并验证其 SHA256、模型配置与特征提取版本。

正式输出在 `outputs/cremad_emotion2vec_linear/`：

- `best.pt`、`last.pt`、`config.json`、`labels.json`、`train.log`、`history.jsonl`。
- `val_metrics.json`、`test_metrics.json`：Accuracy、Macro-F1、UAR、每类 precision/recall/F1/support、混淆矩阵；数值为 0–1。
- `{val,test}_predictions.csv`：逐条预测及六类概率。
- `{val,test}_confusion_matrix.csv/png`：混淆矩阵数据与图片。

## 本地音频预测 App（English UI + FastAPI）

界面已接入 **early stopping 训练过程中按验证集 Macro-F1 保存的 `best.pt`（第 39 轮）**，不加载最后一轮模型。直接启动，无需重新训练：

```bash
cd /workspace
./run_app.sh
```

打开 **http://localhost:8000**。VS Code 连接容器时，在 Ports 面板转发 **8000** 端口。此地址对应当前容器；若从宿主机独立浏览器访问 Docker 内服务，需配置端口映射，再用 `./run_app.sh --host 0.0.0.0` 启动。其他端口可通过 `./run_app.sh --port 8001` 指定。

界面、错误提示和预测标签全部为英文。页面由 FastAPI 提供 HTML/CSS/JavaScript，支持上传 WAV/MP3、拖放文件、音频播放、六类概率条形图、下载预测 JSON，以及三段本地音频示例。20 MB 以内，最长 30 秒，完整处理所选片段。模型在首次识别时加载并缓存至进程结束；替换模型文件后需重启 App。输入变化后会清除旧结果，并忽略已取消请求的返回值。上传文件使用临时目录，处理完毕自动删除。

音频来源栏新增 **Record audio**：点击 **Start recording** 并允许麦克风，讲话后点 **Stop recording**，可先回放，再点 **Analyze emotion**。提供录音计时和实际输入音量条，最多 30 秒自动停止。录音在浏览器端生成单声道 PCM16 WAV，沿用同一 FastAPI 接口、16 kHz 重采样、标准化和 `best.pt`。点击识别之前，录音不会发送到服务器。

停止录音、切换来源或离开页面会释放麦克风；取消权限请求后，即使权限稍后才获准，也不会继续录音。麦克风需要 **localhost 或 HTTPS** 及浏览器授权（[浏览器接口要求](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)）；通过远程 IP 的普通 HTTP 页面访问时，请改用端口转发后的 `http://localhost:8000` 或 HTTPS。

模型、标签和标准化统计量全部来自现有最佳 checkpoint，预处理与 CLI 共用。界面会核对训练日志中的最佳轮次，并显示模型选择信息。

当前环境已安装依赖；重建时执行：

```bash
.venv-emotion2vec/bin/python -m pip install -r requirements-ui.txt
# 或在已激活的环境中直接运行
uvicorn app:app --host 127.0.0.1 --port 8000
# 也支持 python app.py
```

后端入口为 `app.py`，前端位于 `web/`，推理服务为 `ser/app_service.py`。API 文档：`http://localhost:8000/docs`；主要接口为 `GET /api/model`、`GET /api/examples`、`POST /api/predict`、`GET /health`。FastAPI 上传示例：

```bash
curl -F 'file=@data/AudioWAV/1001_DFA_ANG_XX.wav' http://localhost:8000/api/predict
```

已验证 API 上传、格式/大小/损坏音频拒绝、模型单实例缓存、以及真实 WAV/MP3 预测与 CLI 的一致性。录音另验证了 WAV 编码、权限拒绝/延迟授权取消、录音资源清理、音频线程的精确时长上限，并用实际浏览器 WAV 编码函数编码真实语音后完成最佳模型 API 推理。录音检查记录位于 `outputs/microphone_js_validation.json` 和 `outputs/microphone_api_validation.json`。容器中无法实际操作用户的物理麦克风；当前环境没有浏览器工具，未进行浏览器视觉截图检查。旧 Streamlit 页面快照保存在 `legacy/streamlit/`，不参与当前 App 启动。

## TensorBoard 实验记录

训练默认启用 `train.tensorboard: true`，日志写入该次实验输出目录的 `tensorboard/`。当前已将正式实验的 **54 轮原始历史**及最终评估导入；没有重新训练或修改模型。

```bash
source .venv-emotion2vec/bin/activate
python -m tensorboard.main --logdir outputs/cremad_emotion2vec_linear/tensorboard --port 6006 --host 127.0.0.1
```

浏览器打开 `http://localhost:6006`。在 VS Code 容器中运行时，转发容器的 6006 端口。查看多次实验时，可将 `--logdir` 改为 `outputs`。

- Scalars：`Loss/train`、`Loss/val`、`Accuracy/val`、`MacroF1/val`、`UAR/val`、`LearningRate`、最佳验证分数和 early stopping 计数。后续训练还会记录 `Accuracy/train`（每个训练 batch 更新前预测的累计准确率）；旧历史没有该字段，未补造数据。
- 最终结果：`Final/val/*`、`Final/test/*` 记录最佳模型的整体和每类指标；测试集只在训练结束后记录。
- Images：`ConfusionMatrix/val`、`ConfusionMatrix/test`。Text：配置、标签及日志来源。
- 横轴 step 是从 1 开始的 epoch。历史导入的事件时间为导入时刻；后续训练实时记录，每轮刷新，恢复时清除恢复点以后的旧事件显示。

其他已有实验可以从原始结果导入；重复执行会保留已有事件，避免重复追加：

```bash
python -m ser export-tensorboard --run-dir outputs/cremad_emotion2vec_linear
```

接入使用 PyTorch 官方 [SummaryWriter](https://docs.pytorch.org/docs/2.0/tensorboard.html)。TensorBoard 已安装，无需安装 TensorFlow。

## 验证及来源

```bash
python -m ser smoke
python -m unittest discover -s tests -v
```

真实模型 smoke test 验证严格加载、冻结编码器、小批次分类头前后向/更新、保存重载、单文件预测一致性，记录在 `outputs/emotion2vec_smoke/report.json`。它不是正式实验分数。离线测试另覆盖音频解码、演员隔离、训练集统计量和两种分类头重载。

当前状态见 `outputs/validation_status.json`，实际环境见 `outputs/environment.json`。论文与源码来源及许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。原 Vesper 实现和历史审计保留，旧入口为 `python -m ser.vesper`，历史说明在 [README-Vesper.md](README-Vesper.md)；当前项目不再依赖未发布的 Vesper 权重。
