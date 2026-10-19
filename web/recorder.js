'use strict';

// Export at the AudioContext's actual sample rate. The server applies the same
// 16 kHz resampling and utterance normalization as every other WAV input.
function encodeMonoWav(chunks, sampleRate) {
  const count = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + count * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((letter, i) => view.setUint8(offset + i, letter.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + count * 2, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, count * 2, true);
  let offset = 44;
  for (const chunk of chunks) {
    for (const input of chunk) {
      const value = Math.max(-1, Math.min(1, input));
      view.setInt16(offset, Math.round(value * (value < 0 ? 32768 : 32767)), true);
      offset += 2;
    }
  }
  return buffer;
}

class MicrophoneRecorder {
  constructor({ onState, onProgress, onComplete, onError, maxSeconds = 30 }) {
    this.callbacks = { onState, onProgress, onComplete, onError };
    this.maxSeconds = maxSeconds;
    this.session = null;
    this.state = 'idle';
  }

  setState(state) {
    this.state = state;
    this.callbacks.onState(state);
  }

  async start() {
    if (this.session) return;
    if (!window.isSecureContext) {
      this.callbacks.onError('Microphone access requires HTTPS or localhost. Open this app at http://localhost:8000, or use HTTPS.');
      return;
    }
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!navigator.mediaDevices?.getUserMedia || !Context || !window.AudioWorkletNode) {
      this.callbacks.onError('This browser does not support microphone recording. Please upload a WAV or MP3 file instead.');
      return;
    }
    const session = { chunks: [], frames: 0, stream: null, context: null, node: null, source: null, timer: null, fallback: null };
    this.session = session;
    this.setState('requesting');
    try {
      // Create/resume in the button gesture, before waiting for permission.
      session.context = new Context();
      const resumed = session.context.resume();
      resumed.catch(() => {}); // Permission cancellation may close the context first.
      const granted = navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      // A cancelled permission prompt may resolve later; stop those tracks too.
      session.stream = await granted;
      if (this.session !== session) { this.cleanup(session); return; }
      await resumed;
      if (this.session !== session) return;
      if (!session.context.audioWorklet) throw new Error('worklet-unavailable');
      await session.context.audioWorklet.addModule('/static/pcm-recorder.js');
      if (this.session !== session) return;
      session.node = new window.AudioWorkletNode(session.context, 'pcm-recorder', {
        processorOptions: { maxFrames: Math.floor(session.context.sampleRate * this.maxSeconds) },
      });
      session.node.port.onmessage = (event) => {
        if (this.session !== session) return;
        if (event.data.samples?.length) {
          const samples = event.data.samples;
          session.chunks.push(samples);
          session.frames += samples.length;
          const rms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
          this.callbacks.onProgress(session.frames / session.context.sampleRate, Math.min(1, rms * 4));
        }
        if (event.data.done) this.finish(session);
      };
      session.node.onprocessorerror = () => this.fail(session, 'Recording stopped unexpectedly. Please try again.');
      session.source = session.context.createMediaStreamSource(session.stream);
      session.source.connect(session.node);
      session.node.connect(session.context.destination); // Processor outputs silence.
      this.setState('recording');
      session.timer = setTimeout(() => this.stop(), this.maxSeconds * 1000);
    } catch (error) {
      // Handle late permission rejection after the user switches away silently.
      if (this.session !== session) return;
      const messages = {
        NotAllowedError: 'Microphone access was denied. Allow microphone access in your browser and try again.',
        NotFoundError: 'No microphone was found. Connect a microphone and try again.',
        NotReadableError: 'The microphone is unavailable. Close other apps using it and try again.',
      };
      this.fail(session, messages[error.name] || 'Could not start the microphone. Please try again or upload an audio file.');
    }
  }

  stop() {
    const session = this.session;
    if (!session || this.state !== 'recording') return;
    this.setState('stopping');
    clearTimeout(session.timer);
    // All earlier audio messages precede this acknowledgement on the same port.
    session.fallback = setTimeout(() => this.finish(session), 1000);
    session.node.port.postMessage('stop');
  }

  finish(session) {
    if (this.session !== session) return;
    const sampleRate = session.context.sampleRate;
    const chunks = session.chunks;
    this.session = null;
    this.cleanup(session);
    this.setState('idle');
    if (session.frames < sampleRate * 0.25) {
      this.callbacks.onError('That recording was too short. Record at least a quarter second of speech and try again.');
      return;
    }
    const audio = encodeMonoWav(chunks, sampleRate);
    this.callbacks.onComplete(new File([audio], 'microphone-recording.wav', { type: 'audio/wav' }), session.frames / sampleRate);
  }

  fail(session, message) {
    if (this.session !== session) return;
    this.cancel();
    this.callbacks.onError(message);
  }

  cancel() {
    const session = this.session;
    this.session = null;
    if (session) this.cleanup(session);
    this.setState('idle');
  }

  cleanup(session) {
    clearTimeout(session.timer);
    clearTimeout(session.fallback);
    session.stream?.getTracks().forEach((track) => track.stop());
    if (session.source) session.source.disconnect();
    if (session.node) { session.node.port.onmessage = null; session.node.disconnect(); }
    if (session.context && session.context.state !== 'closed') session.context.close().catch(() => {});
  }
}
