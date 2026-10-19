# CREMA-D Speech Emotion Recognition with a Frozen emotion2vec Encoder

This course project uses [emotion2vec/emotion2vec_base](https://huggingface.co/emotion2vec/emotion2vec_base) as a frozen pretrained encoder to extract 768-dimensional emotion features, then trains a linear classifier for six emotion classes. An MLP head is also configurable, but the reported experiment uses only the linear head, without model comparisons. The encoder is neither pretrained again nor fine-tuned. This project does not claim to reproduce the scores reported in the [ACL 2024 paper](https://aclanthology.org/2024.findings-acl.931/).

## Completed Experiment

On October 16, 2026, the following steps were completed in the project container using an RTX 3090 Ti with 24 GB VRAM, PyTorch 2.0.1, and CUDA 11.7: downloading and verifying the official weights with SHA256, strict loading of all parameters, a smoke test with real audio, feature extraction for all 7,442 recordings, classifier training, validation-based model selection, test-set evaluation, and single-file WAV/MP3 prediction.

| Split | Actors | Samples | Accuracy | Macro-F1 | UAR |
|---|---:|---:|---:|---:|---:|
| train | 64 | 5229 | — | — | — |
| val | 14 | 1148 | 74.22% | 74.34% | 74.50% |
| test | 13 | 1065 | 70.05% | 69.92% | 70.50% |

The random seed is fixed at 42. Actors do not overlap across splits, and every split contains all six classes. The best checkpoint is selected exclusively by **validation Macro-F1**: epoch 39 was the best, and early stopping triggered at epoch 54. The test set was evaluated only after training and was not used for model selection. Existing audio files, archives, and Docker files have been preserved.

## Environment and Weights

The dedicated environment is already configured in the existing workspace. Activate it before running the commands below:

```bash
cd /workspace
source .venv-emotion2vec/bin/activate
```

To recreate the environment, start from the Docker/PyTorch environment described in [README-Docker.md](README-Docker.md):

```bash
python -m venv --system-site-packages .venv-emotion2vec
.venv-emotion2vec/bin/python -m pip install -r requirements-emotion2vec.txt
source .venv-emotion2vec/bin/activate
python scripts/download_emotion2vec.py
python -m ser check-checkpoint
```

The weights are stored locally in `checkpoints/emotion2vec_base/`. The Hugging Face revision is pinned to `0c9a3152734f9d7a7a05b4ee6bfb3c109d288664`. The file `emotion2vec_base.pt` contains 1,125,606,009 bytes, with SHA256 `4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd`. The downloader verifies integrity and saves `provenance.json`; no account credentials are required.

The project reuses the official Emotion2vec class from **FunASR 1.2.6**, constructs it from the model repository's `config.yaml`, and loads all 193 parameter entries with `strict=True`. The encoder has approximately 93.79 million parameters and remains in `eval()` mode with `requires_grad=False`. Only the downstream head is trained. This is the base feature model, not emotion2vec+ or its nine-class prediction model.

## Data and Features

The existing workspace already contains `data/AudioWAV/`; no additional download is needed. If the data is missing, obtain AudioWAV from the [official CREMA-D repository](https://github.com/CheyneyComputerScience/CREMA-D), using Git LFS to download the actual audio, or extract the [Kaggle dataset](https://www.kaggle.com/datasets/ejlok1/cremad) into that directory. To use another location, update `data.root` in `configs/emotion2vec.yaml`.

```bash
# Download official WAV files only if the data is missing; skip videos and duplicate MP3 files.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CheyneyComputerScience/CREMA-D.git data/CREMA-D
git -C data/CREMA-D lfs pull --include='AudioWAV/*.wav' --exclude=''
# Then set data.root to data/CREMA-D/AudioWAV.
```

```bash
python -m ser prepare
python -m ser extract
```

- Actor assignments are reused from `data/manifest.csv`. Existing splits are preserved by default. Use `prepare --force` only when intentionally changing the data or seed.
- The manifest records absolute paths, actor IDs, emotion codes/names/IDs, durations, and splits. Training and evaluation check the manifest and audio state to prevent stale caches from being mixed with a new split.
- Labels use the intended emotion encoded in each filename: `ANG=0 angry, DIS=1 disgust, FEA=2 fear, HAP=3 happy, NEU=4 neutral, SAD=5 sad`. There is no surprise class. The file `1040_ITH_SAD_X.wav`, whose intensity suffix is X, is valid SAD audio and retains its original name.
- WAV is preferred, and MP3 is also supported. Full decoding checks reject manifest creation if corrupted, empty, or excessively short audio files or Git LFS pointers are found. Details are written to `data/audio_audit.json`. Invalid files are skipped only with the explicit `--skip-invalid` option.
- Audio is converted to mono at 16 kHz. **Full recordings are retained by default.** Each recording undergoes the official layer normalization, followed by last-layer frame feature extraction and temporal mean pooling. Recordings are processed individually without padding. Training and prediction share the same preprocessing.
- Features are cached in `data/features/emotion2vec_base.pt`, with per-recording `.npy` files for resuming interrupted extraction. Once extraction is complete, running `extract` again only checks consistency. Use `extract --force` after changing data or preprocessing.
- Feature normalization means and standard deviations are fitted **on the training split only** and saved with the classifier. Validation and test features use those saved statistics. The frozen-feature pipeline does not use data augmentation.

## Training, Resuming, and Background Execution

All settings are in `configs/emotion2vec.yaml`. Relative paths are resolved from the project root. Defaults are a linear `768→6` head, AdamW, batch size 128, learning rate 1e-3, a maximum of 100 epochs, and early-stopping patience of 15.

```bash
python -m ser train
# Or run feature extraction, training, and final evaluation together.
python -m ser run
# Resume from the most recent completed epoch using the original configuration and output directory.
python -m ser train --resume outputs/cremad_emotion2vec_linear/last.pt
```

**The existing experiment results are already present locally.** A new training run refuses to overwrite them. Before retraining, set `train.output_dir` to a new directory. To use the MLP option, set `model.head: mlp` and choose a new output directory; features do not need to be extracted again.

`best.pt` stores the classifier with the highest validation Macro-F1. `last.pt` includes the model, normalization statistics, labels, encoder provenance/configuration, optimizer, scheduler, random states, and early-stopping state. Checkpoints are saved only after completed epochs; an interrupted epoch is rerun when training resumes.

```bash
# Start from your own container terminal; the container must remain running.
bash scripts/run_emotion2vec_background.sh configs/emotion2vec.yaml
# The script prints the PID and log location.
```

## Evaluation and Single-File Prediction

```bash
python -m ser evaluate --checkpoint outputs/cremad_emotion2vec_linear/best.pt --split test
python -m ser predict --checkpoint outputs/cremad_emotion2vec_linear/best.pt --audio data/AudioWAV/1001_DFA_ANG_XX.wav
python -m ser predict --checkpoint outputs/cremad_emotion2vec_linear/best.pt --audio /path/to/sample.mp3 --output outputs/prediction.json
```

Prediction returns the predicted emotion and probabilities for all six classes. CPU inference is supported with `--device cpu`. If the encoder directory has moved, specify `--encoder-dir /path/to/emotion2vec_base`. **Inference requires both the classifier checkpoint `best.pt` and the official encoder directory.** The classifier checkpoint does not duplicate the 1.1 GB of public weights, but records and verifies their SHA256, model configuration, and feature extraction version.

Experiment artifacts are stored in `outputs/cremad_emotion2vec_linear/`:

- `best.pt`, `last.pt`, `config.json`, `labels.json`, `train.log`, and `history.jsonl`.
- `val_metrics.json` and `test_metrics.json`: Accuracy, Macro-F1, UAR, per-class precision/recall/F1/support, and confusion matrices. Metric values range from 0 to 1.
- `{val,test}_predictions.csv`: per-recording predictions and probabilities for all six classes.
- `{val,test}_confusion_matrix.csv/png`: confusion matrix data and images.

## Local Audio Prediction App (English UI + FastAPI)

The app uses **`best.pt`, selected by validation Macro-F1 during training with early stopping (epoch 39)**, rather than the final epoch's model. Start it without retraining:

```bash
cd /workspace
./run_app.sh
```

Open **http://localhost:8000**. When VS Code is attached to the container, forward port **8000** in its Ports panel. The address must point to this container. To access the Docker service from a separate host browser, configure port publishing and start the app with `./run_app.sh --host 0.0.0.0`. Use `./run_app.sh --port 8001` to select another port.

The interface, error messages, and prediction labels are in English. FastAPI serves the HTML/CSS/JavaScript frontend, which supports WAV/MP3 uploads, drag and drop, audio playback, a six-class probability chart, prediction JSON downloads, and three local audio examples. Inputs are limited to 20 MB and 30 seconds, and the full selected clip is processed. The model loads on the first prediction and is cached until the process exits. Restart the app after replacing model files. Changing the input clears previous results, and responses from canceled requests are ignored. Uploaded files are stored temporarily and deleted after processing.

To record audio, select **Record audio**, click **Start recording**, and grant microphone permission. After speaking, click **Stop recording**, optionally play back the recording, then select **Analyze emotion**. A timer and live input level meter are provided; recording stops automatically after 30 seconds. The browser generates a mono PCM16 WAV file, which uses the same FastAPI endpoint, 16 kHz resampling, normalization, and `best.pt`. Audio is not sent to the server until you request analysis.

Stopping a recording, switching input sources, or leaving the page releases the microphone. Canceling a permission request prevents recording even if permission is granted later. Microphone access requires **localhost or HTTPS** and browser permission ([browser API requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)). If you are accessing a remote IP over plain HTTP, use the forwarded `http://localhost:8000` address or HTTPS instead.

The model, labels, and normalization statistics come from the existing best checkpoint. Preprocessing is shared with the CLI. The UI checks the best epoch against the training log and displays model selection information.

Dependencies are installed in the existing environment. When recreating it, run:

```bash
.venv-emotion2vec/bin/python -m pip install -r requirements-ui.txt
# Or start directly from the activated environment.
uvicorn app:app --host 127.0.0.1 --port 8000
# python app.py is also supported.
```

The backend entry point is `app.py`, the frontend is in `web/`, and the inference service is `ser/app_service.py`. API documentation is available at `http://localhost:8000/docs`. Main endpoints are `GET /api/model`, `GET /api/examples`, `POST /api/predict`, and `GET /health`. Example upload:

```bash
curl -F 'file=@data/AudioWAV/1001_DFA_ANG_XX.wav' http://localhost:8000/api/predict
```

Previous checks covered API uploads, rejection of unsupported formats/oversized files/corrupted audio, single-instance model caching, and consistency between real WAV/MP3 API predictions and CLI predictions. Recording checks additionally covered WAV encoding, permission rejection and cancellation of delayed permission grants, recording resource cleanup, and the audio thread's exact duration limit. Real speech was encoded using the browser's actual WAV encoding function and passed through the best-model prediction API. Records are in `outputs/microphone_js_validation.json` and `outputs/microphone_api_validation.json`. The user's physical microphone could not be operated from the container. No browser tools were available in that environment, so no browser screenshot-based visual checks were performed. The previous Streamlit UI snapshot is retained in `legacy/streamlit/` and is not used to start the current app.

## TensorBoard Experiment Logs

Training enables `train.tensorboard: true` by default and writes logs to `tensorboard/` under the experiment output directory. The existing experiment's **54 epochs of original history** and final evaluation have been imported without retraining or modifying the model.

```bash
source .venv-emotion2vec/bin/activate
python -m tensorboard.main --logdir outputs/cremad_emotion2vec_linear/tensorboard --port 6006 --host 127.0.0.1
```

Open `http://localhost:6006`. When running inside a VS Code container session, forward container port 6006. To view multiple experiments, set `--logdir` to `outputs`.

- Scalars include `Loss/train`, `Loss/val`, `Accuracy/val`, `MacroF1/val`, `UAR/val`, `LearningRate`, the best validation score, and the early-stopping counter. Future training runs also record `Accuracy/train`, accumulated from predictions made before each training batch update. This field is absent from the original history and has not been fabricated.
- Final results under `Final/val/*` and `Final/test/*` contain overall and per-class metrics for the best model. Test metrics are logged only after training ends.
- Images include `ConfusionMatrix/val` and `ConfusionMatrix/test`. Text entries contain the configuration, labels, and log provenance.
- The horizontal step axis represents epochs starting at 1. Imported events use the import time as their timestamp. Future runs log events in real time and flush each epoch. Resuming training clears displayed stale events beyond the resume point.

Other existing experiments can be imported from their original results. Repeated execution preserves existing events rather than appending duplicates:

```bash
python -m ser export-tensorboard --run-dir outputs/cremad_emotion2vec_linear
```

The integration uses PyTorch's [SummaryWriter](https://docs.pytorch.org/docs/2.0/tensorboard.html). TensorBoard is installed; TensorFlow is not required.

## Validation and Sources

```bash
python -m ser smoke
python -m unittest discover -s tests -v
```

The real-model smoke test checks strict parameter loading, a frozen encoder, a small classifier forward/backward pass and update, checkpoint saving/reloading, and consistency of single-file predictions. Its record is stored in `outputs/emotion2vec_smoke/report.json`; it is not a formal experiment score. Offline tests also cover audio decoding, actor separation, training-only statistics, and reloading both classifier head types.

See `outputs/validation_status.json` for the recorded validation status and `outputs/environment.json` for the recorded environment. Paper/code sources and license notes are in [THIRD_PARTY.md](THIRD_PARTY.md). The original Vesper implementation and historical audit records are preserved. Its legacy entry point is `python -m ser.vesper`, with documentation in [README-Vesper.md](README-Vesper.md). The current project no longer depends on unreleased Vesper weights.
