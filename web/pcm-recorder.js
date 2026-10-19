'use strict';

// Runs on the browser audio thread. Capture mono PCM without playing the mic
// back through the speakers. The frame cap keeps exported WAVs <= 30 seconds.
class PCMRecorderProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.maxFrames = options.processorOptions.maxFrames;
    this.frames = 0;
    this.running = true;
    this.port.onmessage = (event) => {
      if (event.data === 'stop' && this.running) {
        this.running = false;
        this.port.postMessage({ done: true });
      }
    };
  }

  process(inputs) {
    if (!this.running) return false;
    const channels = inputs[0];
    if (!channels || !channels.length || !channels[0].length) return true;
    const count = Math.min(channels[0].length, this.maxFrames - this.frames);
    const mono = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      for (const channel of channels) mono[i] += channel[i] / channels.length;
    }
    this.frames += count;
    const done = this.frames >= this.maxFrames;
    this.port.postMessage({ samples: mono, done }, [mono.buffer]);
    if (done) this.running = false;
    return this.running;
  }
}

registerProcessor('pcm-recorder', PCMRecorderProcessor);
