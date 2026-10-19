'use strict';

const byId = (id) => document.getElementById(id);
const labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'];
const titleCase = (word) => word.charAt(0).toUpperCase() + word.slice(1);
let selectedFile = null;
let result = null;
let audioUrl = null;
let modelReady = false;
let busy = false;
let revision = 0;
let currentRequest = null;
let samples = [];
const microphone = new MicrophoneRecorder({
  onState(state) {
    const messages = { idle: 'Ready to record', requesting: 'Waiting for microphone permission…', recording: 'Recording — speak into your microphone', stopping: 'Finishing recording…' };
    byId('record-status').textContent = messages[state];
    byId('record-button-label').textContent = state === 'recording' ? 'Stop recording' : state === 'requesting' ? 'Cancel' : state === 'stopping' ? 'Finishing…' : 'Start recording';
    byId('record-button').disabled = state === 'stopping';
    byId('record-button').classList.toggle('recording', state === 'recording');
    if (state === 'idle') setMicLevel(0);
    updateButton();
  },
  onProgress(seconds, level) {
    const elapsed = Math.min(30, Math.floor(seconds));
    byId('record-timer').textContent = `00:${String(elapsed).padStart(2, '0')} / 00:30`;
    setMicLevel(level);
  },
  onComplete(file, seconds) {
    selectFile(file);
    byId('record-timer').textContent = `00:${String(Math.floor(seconds)).padStart(2, '0')} / 00:30`;
    byId('record-status').textContent = 'Recording ready — play it back or analyze';
    byId('processing-note').textContent = 'Recording ready. Click Analyze emotion when you are ready.';
  },
  onError(message) { showError(message); },
});

function setMicLevel(level) {
  byId('mic-level').style.width = `${Math.round(level * 100)}%`;
  byId('mic-meter').setAttribute('aria-valuenow', String(Math.round(level * 100)));
}

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed. Please try again.');
  }
  return data;
}

function updateButton() {
  byId('analyze-button').disabled = !selectedFile || !modelReady || busy || microphone.state !== 'idle';
  byId('button-text').textContent = busy ? 'Analyzing audio…' : 'Analyze emotion';
  byId('spinner').hidden = !busy;
  document.querySelector('.button-arrow').hidden = busy;
  byId('result-panel').setAttribute('aria-busy', String(busy));
}

function showError(message) {
  byId('input-error').textContent = message;
  byId('input-error').hidden = !message;
}

function clearResult() {
  result = null;
  byId('prediction-result').hidden = true;
  byId('empty-state').hidden = false;
  byId('probability-chart').replaceChildren();
}

function clearInput() {
  microphone.cancel();
  byId('record-timer').textContent = '00:00 / 00:30';
  revision += 1;
  if (currentRequest) currentRequest.abort();
  currentRequest = null;
  selectedFile = null;
  busy = false;
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = null;
  const player = byId('audio-player');
  player.pause();
  player.removeAttribute('src');
  player.load();
  byId('audio-preview').hidden = true;
  byId('audio-file').value = '';
  byId('processing-note').textContent = 'Choose a file, sample, or microphone recording to begin.';
  clearResult();
  showError('');
  updateButton();
}

function selectFile(file) {
  clearInput();
  if (!file) return;
  if (!/\.(wav|mp3)$/i.test(file.name)) {
    showError('Please choose a WAV or MP3 audio file.');
    return;
  }
  if (file.size === 0) {
    showError('This file is empty. Please choose another audio file.');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError('This file exceeds the 20 MB limit. Please choose a smaller file.');
    return;
  }
  selectedFile = file;
  audioUrl = URL.createObjectURL(file);
  byId('audio-player').src = audioUrl;
  byId('file-name').textContent = file.name;
  byId('file-meta').textContent = `${(file.size / 1024).toFixed(0)} KB · ${file.name.split('.').pop().toUpperCase()}`;
  byId('audio-preview').hidden = false;
  byId('processing-note').textContent = 'Ready when you are.';
  updateButton();
}

function setSource(mode) {
  clearInput();
  for (const option of ['upload', 'example', 'record']) {
    const active = option === mode;
    byId(`${option}-panel`).hidden = !active;
    byId(`${option}-tab`).classList.toggle('active', active);
    byId(`${option}-tab`).setAttribute('aria-pressed', String(active));
  }
  byId('sample-select').value = '';
}

function toggleRecording() {
  if (microphone.state === 'recording') { microphone.stop(); return; }
  if (microphone.state === 'requesting') { microphone.cancel(); return; }
  if (microphone.state !== 'idle') return;
  clearInput();
  byId('processing-note').textContent = 'Recording stays in your browser until you click Analyze emotion.';
  microphone.start();
}

async function chooseSample() {
  clearInput();
  const sample = samples.find((item) => item.id === byId('sample-select').value);
  if (!sample) return;
  const version = revision;
  byId('processing-note').textContent = 'Loading sample audio…';
  try {
    const response = await fetch(sample.audio_url);
    if (!response.ok) throw new Error('This sample could not be loaded. Please try uploading a file.');
    const blob = await response.blob();
    if (version !== revision) return;
    selectFile(new File([blob], sample.filename, { type: 'audio/wav' }));
  } catch (error) {
    if (version === revision) {
      showError(error.message);
      byId('processing-note').textContent = 'Select another sample or upload a file.';
    }
  }
}

function renderResult(data) {
  result = data;
  byId('predicted-emotion').textContent = titleCase(data.prediction);
  byId('predicted-probability').textContent = `${(data.probabilities[data.prediction] * 100).toFixed(2)}%`;
  byId('result-meta').textContent = `${data.duration_seconds.toFixed(2)} seconds of audio · Best checkpoint, epoch ${data.best_epoch}`;
  const chart = byId('probability-chart');
  chart.replaceChildren();
  const sorted = [...labels].sort((a, b) => data.probabilities[b] - data.probabilities[a]);
  for (const label of sorted) {
    const probability = data.probabilities[label];
    const row = document.createElement('div');
    row.className = `prob-row${label === data.prediction ? ' top' : ''}`;
    const name = document.createElement('span');
    name.className = 'prob-label';
    name.textContent = titleCase(label);
    const track = document.createElement('div');
    track.className = 'prob-track';
    const fill = document.createElement('div');
    fill.className = 'prob-fill';
    fill.style.width = `${Math.max(0, Math.min(1, probability)) * 100}%`;
    track.appendChild(fill);
    const value = document.createElement('span');
    value.className = 'prob-value';
    value.textContent = `${(probability * 100).toFixed(1)}%`;
    row.append(name, track, value);
    chart.appendChild(row);
  }
  chart.setAttribute('aria-label', labels.map((label) => `${label}: ${(data.probabilities[label] * 100).toFixed(1)} percent`).join(', '));
  byId('empty-state').hidden = true;
  byId('prediction-result').hidden = false;
}

async function analyze() {
  if (!selectedFile || busy || !modelReady) return;
  const version = revision;
  const form = new FormData();
  form.append('file', selectedFile);
  busy = true;
  showError('');
  clearResult();
  updateButton();
  byId('processing-note').textContent = 'Loading the saved model and analyzing your audio…';
  currentRequest = new AbortController();
  try {
    const data = await fetch('/api/predict', { method: 'POST', body: form, signal: currentRequest.signal }).then(readJson);
    if (version !== revision) return;
    renderResult(data);
    byId('processing-note').textContent = `Analysis complete · ${data.inference_seconds.toFixed(2)} s processing time`;
  } catch (error) {
    if (version === revision && error.name !== 'AbortError') {
      showError(error.message);
      byId('processing-note').textContent = 'Analysis failed. Please try another clip.';
    }
  } finally {
    if (version === revision) {
      busy = false;
      currentRequest = null;
      updateButton();
    }
  }
}

function downloadResult() {
  if (!result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'emotion_prediction.json';
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function initialize() {
  await Promise.allSettled([
    fetch('/api/model').then(readJson).then((info) => {
      byId('best-epoch').textContent = info.best_epoch;
      byId('val-f1').textContent = `${(info.val_macro_f1 * 100).toFixed(2)}%`;
      byId('model-status').textContent = 'Best checkpoint ready';
      byId('status-dot').classList.add('ready');
      modelReady = true;
      updateButton();
    }).catch((error) => {
      byId('model-status').textContent = 'Model unavailable';
      byId('model-error').textContent = error.message;
      byId('model-error').hidden = false;
    }),
    fetch('/api/examples').then(readJson).then((data) => {
      samples = data;
      for (const sample of samples) {
        const option = document.createElement('option');
        option.value = sample.id;
        option.textContent = sample.title;
        byId('sample-select').appendChild(option);
      }
      if (!samples.length) {
        byId('sample-select').disabled = true;
        byId('sample-select').options[0].textContent = 'No local samples available';
      }
    }).catch(() => {
      byId('sample-select').disabled = true;
      byId('sample-select').options[0].textContent = 'Samples unavailable — upload a file instead';
    })
  ]);
}

byId('browse-button').addEventListener('click', () => byId('audio-file').click());
byId('audio-file').addEventListener('change', (event) => selectFile(event.target.files[0]));
byId('clear-button').addEventListener('click', clearInput);
byId('upload-tab').addEventListener('click', () => setSource('upload'));
byId('example-tab').addEventListener('click', () => setSource('example'));
byId('record-tab').addEventListener('click', () => setSource('record'));
byId('record-button').addEventListener('click', toggleRecording);
byId('sample-select').addEventListener('change', chooseSample);
byId('analyze-button').addEventListener('click', analyze);
byId('download-button').addEventListener('click', downloadResult);
byId('dropzone').addEventListener('dragover', (event) => { event.preventDefault(); byId('dropzone').classList.add('dragover'); });
byId('dropzone').addEventListener('dragleave', () => byId('dropzone').classList.remove('dragover'));
byId('dropzone').addEventListener('drop', (event) => {
  event.preventDefault();
  byId('dropzone').classList.remove('dragover');
  selectFile(event.dataTransfer.files[0]);
});
window.addEventListener('pagehide', () => microphone.cancel());
document.addEventListener('visibilitychange', () => {
  if (document.hidden && microphone.state === 'recording') microphone.stop();
  if (document.hidden && microphone.state === 'requesting') microphone.cancel();
});
initialize();
