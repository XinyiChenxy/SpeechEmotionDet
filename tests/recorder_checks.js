/* Pure JavaScript checks; no microphone permission or physical device is used. */
async function runRecorderChecks(recorderSource, processorSource) {
  const assert = (value, message) => { if (!value) throw new Error(message); };
  const checks = [];
  const timers = new Map();
  let timerId = 0;
  const setTimer = (callback) => { const id = ++timerId; timers.set(id, callback); return id; };
  const clearTimer = (id) => timers.delete(id);
  let stops = 0;
  let latestNode = null;
  class FakeFile { constructor(parts, name, options) { this.parts = parts; this.name = name; this.type = options.type; } }
  class Context {
    constructor() { this.sampleRate = 48000; this.state = 'suspended'; this.audioWorklet = { addModule: async () => {} }; }
    resume() { this.state = 'running'; return Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
  }
  class WorkletNode {
    constructor() {
      latestNode = this;
      this.port = { onmessage: null, postMessage: () => this.port.onmessage?.({ data: { done: true } }) };
    }
    connect() {}
    disconnect() {}
  }
  const stream = () => ({ getTracks: () => [{ stop() { stops += 1; } }] });
  const nav = { mediaDevices: { getUserMedia: async () => stream() } };
  const win = { isSecureContext: true, AudioContext: Context, AudioWorkletNode: WorkletNode };
  const { MicrophoneRecorder, encodeMonoWav } = new Function('window', 'navigator', 'File', 'setTimeout', 'clearTimeout',
    recorderSource + '\nreturn { MicrophoneRecorder, encodeMonoWav };')(win, nav, FakeFile, setTimer, clearTimer);
  const wav = new DataView(encodeMonoWav([new Float32Array([-2, -1, -.5, 0, .5, 1, 2])], 48000));
  assert(wav.getUint32(24, true) === 48000 && wav.getUint16(22, true) === 1, 'WAV sample rate/channels');
  assert(wav.getUint32(40, true) === 14 && wav.getInt16(44, true) === -32768 && wav.getInt16(56, true) === 32767, 'PCM clipping/header');
  checks.push('PCM16 mono WAV encoding and clipping');
  const completed = [], errors = [], states = [];
  const recorder = new MicrophoneRecorder({ onState: (s) => states.push(s), onProgress() {},
    onComplete: (file, seconds) => completed.push({ file, seconds }), onError: (error) => errors.push(error) });
  await recorder.start();
  assert(recorder.state === 'recording', 'Starts after microphone permission');
  latestNode.port.onmessage({ data: { samples: new Float32Array(24000).fill(.2) } });
  recorder.stop();
  assert(completed.length === 1 && completed[0].seconds === .5 && stops === 1, 'Stop exports audio and releases microphone');
  assert(timers.size === 0 && recorder.state === 'idle', 'Stop clears timers');
  assert(new DataView(completed[0].file.parts[0]).getUint32(40, true) === 48000, 'Recorded sample count preserved');
  checks.push('Start/stop, WAV export, microphone and timer cleanup');
  nav.mediaDevices.getUserMedia = async () => { const e = new Error(); e.name = 'NotAllowedError'; throw e; };
  await recorder.start();
  assert(errors.at(-1).includes('denied') && recorder.state === 'idle', 'Permission rejection returns usable UI');
  checks.push('Denied microphone permission');
  let grant;
  nav.mediaDevices.getUserMedia = () => new Promise((resolve) => { grant = resolve; });
  const pending = recorder.start();
  recorder.cancel();
  grant(stream());
  await pending;
  assert(stops === 2 && completed.length === 1 && recorder.state === 'idle', 'Late permission must not restart a cancelled recording');
  checks.push('Cancel pending permission and stop late-arriving microphone tracks');
  nav.mediaDevices.getUserMedia = async () => stream();
  await recorder.start();
  latestNode.port.onmessage({ data: { samples: new Float32Array(10) } });
  recorder.stop();
  assert(completed.length === 1 && errors.at(-1).includes('too short'), 'Short clip rejected');
  checks.push('Very short recording rejection');
  win.isSecureContext = false;
  await recorder.start();
  assert(errors.at(-1).includes('HTTPS or localhost'), 'Insecure context guidance');
  checks.push('Insecure HTTP microphone guidance');
  let Processor;
  class ProcessorBase { constructor() { this.messages = []; this.port = { postMessage: (message) => this.messages.push(message) }; } }
  new Function('AudioWorkletProcessor', 'registerProcessor', processorSource)(ProcessorBase, (_name, value) => { Processor = value; });
  const processor = new Processor({ processorOptions: { maxFrames: 6 } });
  processor.process([[new Float32Array([1, 1, 1, 1]), new Float32Array([-1, -1, -1, -1])]]);
  processor.process([[new Float32Array([.5, .5, .5, .5])]]);
  assert(processor.messages[0].samples.every((value) => value === 0), 'Stereo mixdown');
  assert(processor.frames === 6 && processor.messages[1].samples.length === 2 && processor.messages[1].done, 'Hard frame cap avoids >30 second recordings');
  assert(processor.process([]) === false, 'Processor stops after cap');
  checks.push('Audio-thread stereo mixdown, exact frame limit and automatic stop');
  return { status: 'passed', checks, physical_microphone_tested: false };
}

// Optional standalone runner when Node.js is available.
if (typeof require === 'function' && typeof module !== 'undefined' && require.main === module) {
  const fs = require('fs');
  const path = require('path');
  const root = path.resolve(__dirname, '..');
  runRecorderChecks(fs.readFileSync(path.join(root, 'web/recorder.js'), 'utf8'),
    fs.readFileSync(path.join(root, 'web/pcm-recorder.js'), 'utf8'))
    .then((report) => console.log(JSON.stringify(report, null, 2)))
    .catch((error) => { console.error(error); process.exitCode = 1; });
}
