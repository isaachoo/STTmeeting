// Converts the microphone's Float32 samples into little-endian 16-bit PCM and
// posts them to the main thread in fixed-size chunks.
//
// No resampling here: we tell Deepgram whatever rate the AudioContext gave us,
// which is exact and cheaper than interpolating in JavaScript.

const CHUNK_SAMPLES = 2048;

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Int16Array(CHUNK_SAMPLES);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      let sample = channel[i];
      if (sample > 1) sample = 1;
      else if (sample < -1) sample = -1;
      this.buffer[this.filled++] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;

      if (this.filled === CHUNK_SAMPLES) {
        const chunk = this.buffer.slice();
        this.port.postMessage(chunk.buffer, [chunk.buffer]);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
