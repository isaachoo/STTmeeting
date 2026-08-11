/* Cantonese meeting copilot -- browser side.
 *
 * Owns three things: microphone capture, one WebSocket to the server, and
 * rendering the four panels. All state that matters lives on the server, so a
 * page refresh (or a dropped socket) replays it from the `snapshot` event.
 */

/* A very small event bus over one WebSocket. JSON text frames carry events in
 * both directions; binary frames carry microphone audio upstream. It reconnects
 * on its own, because a five-hour meeting will lose the socket at least once. */
class Bus {
  constructor(path) {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    this.url = `${scheme}://${location.host}${path}`;
    this.handlers = new Map();
    this.backoff = 500;
    this.ws = null;
    this.pending = [];
    this.connect();
  }

  on(event, fn) {
    if (!this.handlers.has(event)) this.handlers.set(event, []);
    this.handlers.get(event).push(fn);
  }

  fire(event, data) {
    (this.handlers.get(event) || []).forEach((fn) => fn(data));
  }

  connect() {
    this.ws = new WebSocket(this.url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.backoff = 500;
      const queued = this.pending.splice(0);
      queued.forEach((frame) => this.ws.send(frame));
      this.fire('__open');
    };
    this.ws.onclose = () => {
      this.fire('__close');
      setTimeout(() => this.connect(), this.backoff);
      this.backoff = Math.min(this.backoff * 2, 10000);
    };
    this.ws.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      this.fire(message.event, message.data);
    };
  }

  get connected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  /* Control events are held until the socket is up rather than dropped: a Start
   * click during a reconnect must still start the meeting. Audio is different --
   * see sendAudio, where stale samples are worth less than a live stream. */
  emit(event, data = {}) {
    const frame = JSON.stringify({ event, data });
    if (this.connected) {
      this.ws.send(frame);
    } else if (this.pending.length < 50) {
      this.pending.push(frame);
    }
  }

  sendAudio(buffer) {
    if (!this.connected) return;
    // If the socket is backing up, drop the chunk rather than grow the buffer
    // without limit -- stale audio is worth less than a responsive page.
    if (this.ws.bufferedAmount > 1_000_000) return;
    this.ws.send(buffer);
  }
}

const socket = new Bus('/ws');

const el = (id) => document.getElementById(id);
const ui = {
  connDot: el('conn-dot'),
  statusText: el('status-text'),
  elapsed: el('elapsed'),
  btnStart: el('btn-start'),
  btnStop: el('btn-stop'),
  setup: el('setup'),
  inTitle: el('in-title'),
  inBrief: el('in-brief'),
  inMicMode: el('in-mic-mode'),
  transcript: el('transcript'),
  interim: el('interim'),
  autoscroll: el('chk-autoscroll'),
  advice: el('advice'),
  notes: el('notes'),
  notesUpdated: el('notes-updated'),
  userNotes: el('user-notes'),
  notesSaveHint: el('notes-save-hint'),
  summary: el('summary'),
  log: el('log'),
  sttModel: el('stt-model'),
  askForm: el('ask-form'),
  inAsk: el('in-ask'),
  costTotal: el('cost-total'),
  costStt: el('cost-stt'),
  costLlm: el('cost-llm'),
  statAudio: el('stat-audio'),
  statCalls: el('stat-calls'),
  statTokens: el('stat-tokens'),
  btnDlTranscript: el('btn-dl-transcript'),
  btnDlNotes: el('btn-dl-notes'),
};

let audioContext = null;
let mediaStream = null;
let workletNode = null;
let streaming = false;
let running = false;
let startedAtMs = null;
let latestNotes = null;

// ------------------------------------------------------------------ utilities

function log(message, isError = false) {
  const line = document.createElement('div');
  if (isError) line.className = 'err';
  const t = new Date().toLocaleTimeString();
  line.textContent = `${t}  ${message}`;
  ui.log.prepend(line);
  while (ui.log.childElementCount > 100) ui.log.lastElementChild.remove();
}

function clockFromSeconds(total) {
  const s = Math.max(0, Math.floor(total));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
}

function usd(value) {
  return `$${Number(value || 0).toFixed(4)}`;
}

function download(filename, text) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

setInterval(() => {
  if (running && startedAtMs) {
    ui.elapsed.textContent = clockFromSeconds((Date.now() - startedAtMs) / 1000);
  }
}, 500);

// --------------------------------------------------------------- microphone

async function startCapture() {
  const roomMode = ui.inMicMode.value === 'room';
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      // Room mode hands Deepgram the untouched signal, which is usually better
      // when the mic is picking up several people across a table.
      echoCancellation: false,
      noiseSuppression: !roomMode,
      autoGainControl: true,
    },
  });

  // Ask for 16 kHz: plenty for speech and a quarter of the bytes. Browsers that
  // refuse simply give us their own rate, which we report to the server as-is.
  audioContext = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: 16000,
  });
  await audioContext.audioWorklet.addModule('/static/js/pcm-worklet.js');

  const source = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, 'pcm-processor');
  workletNode.port.onmessage = (event) => {
    if (streaming) socket.sendAudio(event.data);
  };

  // A muted path to the speakers keeps the graph pulling samples in every
  // browser without playing the meeting back into the room.
  const silence = audioContext.createGain();
  silence.gain.value = 0;
  source.connect(workletNode);
  workletNode.connect(silence);
  silence.connect(audioContext.destination);

  return audioContext.sampleRate;
}

function stopCapture() {
  streaming = false;
  if (workletNode) {
    workletNode.port.onmessage = null;
    workletNode.disconnect();
    workletNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
}

// -------------------------------------------------------------------- actions

ui.btnStart.addEventListener('click', async () => {
  ui.btnStart.disabled = true;
  ui.statusText.textContent = 'asking for the microphone…';
  try {
    const sampleRate = await startCapture();
    log(`microphone open at ${sampleRate} Hz`);
    socket.emit('start_meeting', {
      title: ui.inTitle.value,
      brief: ui.inBrief.value,
      sample_rate: sampleRate,
    });
  } catch (err) {
    stopCapture();
    ui.btnStart.disabled = false;
    ui.statusText.textContent = 'microphone blocked';
    log(`could not open the microphone: ${err.message}`, true);
  }
});

ui.btnStop.addEventListener('click', () => {
  ui.btnStop.disabled = true;
  stopCapture();
  socket.emit('stop_meeting', {});
  ui.statusText.textContent = 'wrapping up…';
});

ui.askForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = ui.inAsk.value.trim();
  if (!question) return;
  socket.emit('ask', { question });
  ui.inAsk.value = '';
  log(`asked: ${question}`);
});

let notesSaveTimer = null;
ui.userNotes.addEventListener('input', () => {
  ui.notesSaveHint.textContent = 'unsaved…';
  clearTimeout(notesSaveTimer);
  notesSaveTimer = setTimeout(() => {
    socket.emit('user_notes', { text: ui.userNotes.value });
    ui.notesSaveHint.textContent = `saved ${new Date().toLocaleTimeString()}`;
  }, 800);
});

ui.btnDlTranscript.addEventListener('click', () => {
  const lines = [...ui.transcript.querySelectorAll('.line')].map((line) => {
    const who = line.querySelector('.who')?.textContent ?? '?';
    const at = line.querySelector('.at')?.textContent ?? '';
    return `[${at}] ${who}: ${line.querySelector('.said').textContent}`;
  });
  if (!lines.length) return log('nothing to download yet');
  download(`transcript-${Date.now()}.txt`, lines.join('\n'));
});

ui.btnDlNotes.addEventListener('click', () => {
  if (!latestNotes) return log('no notes yet');
  download(`notes-${Date.now()}.md`, notesToMarkdown(latestNotes, ui.userNotes.value));
});

// ------------------------------------------------------------------ rendering

function clearIfEmpty(container) {
  const placeholder = container.querySelector('.empty');
  if (placeholder) placeholder.remove();
}

function addSegment(seg) {
  clearIfEmpty(ui.transcript);
  const nearBottom =
    ui.transcript.scrollHeight - ui.transcript.scrollTop - ui.transcript.clientHeight < 120;

  const line = document.createElement('p');
  line.className = 'line';

  const who = document.createElement('span');
  who.className = `who${seg.speaker !== null && seg.speaker !== undefined ? ` s${seg.speaker % 4}` : ''}`;
  who.textContent = seg.speaker_label || '?';

  const said = document.createElement('span');
  said.className = 'said';
  said.textContent = seg.text;

  const at = document.createElement('span');
  at.className = 'at';
  at.textContent = clockFromSeconds(seg.at || 0);

  line.append(who, said, at);
  ui.transcript.append(line);

  if (ui.autoscroll.checked && nearBottom) {
    ui.transcript.scrollTop = ui.transcript.scrollHeight;
  }
}

function copyOnClick(node, text) {
  node.title = 'click to copy';
  node.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(text);
      const original = node.textContent;
      node.textContent = `${original}  ✓ copied`;
      setTimeout(() => { node.textContent = original; }, 1200);
    } catch {
      log('clipboard is not available in this browser', true);
    }
  });
}

function addAdviceCard(payload) {
  clearIfEmpty(ui.advice);
  const card = document.createElement('div');
  card.className = 'card';

  const time = document.createElement('span');
  time.className = 'card-time';
  time.textContent = new Date((payload.at || Date.now() / 1000) * 1000).toLocaleTimeString();
  card.append(time);

  if (payload.key_point) {
    const point = document.createElement('p');
    point.className = 'point';
    point.textContent = payload.key_point;
    card.append(point);
  }
  if (payload.watch_out) {
    const watch = document.createElement('p');
    watch.className = 'watch';
    watch.textContent = payload.watch_out;
    card.append(watch);
  }
  if (payload.questions && payload.questions.length) {
    const heading = document.createElement('h4');
    heading.textContent = 'You could ask';
    const list = document.createElement('ul');
    payload.questions.forEach((q) => {
      const item = document.createElement('li');
      item.textContent = q;
      copyOnClick(item, q);
      list.append(item);
    });
    card.append(heading, list);
  }
  ui.advice.prepend(card);
  trim(ui.advice);
}

function addAnswerCard(payload) {
  clearIfEmpty(ui.advice);
  const card = document.createElement('div');
  card.className = `card answer${payload.from_user ? ' mine' : ''}`;

  const time = document.createElement('span');
  time.className = 'card-time';
  time.textContent = new Date((payload.at || Date.now() / 1000) * 1000).toLocaleTimeString();

  const heading = document.createElement('h4');
  heading.textContent = payload.from_user ? 'You asked' : 'Answer to a question raised';

  const question = document.createElement('p');
  question.className = 'q';
  question.textContent = payload.question;

  const answer = document.createElement('p');
  answer.className = 'a';
  answer.textContent = payload.answer;

  card.append(time, heading, question, answer);

  if (payload.sources && payload.sources.length) {
    const list = document.createElement('ul');
    list.className = 'sources';
    payload.sources.forEach((source, i) => {
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = source.url;
      link.target = '_blank';
      link.rel = 'noreferrer noopener';
      link.textContent = `[${i + 1}] ${source.title || source.url}`;
      item.append(link);
      list.append(item);
    });
    card.append(list);
  } else {
    const flag = document.createElement('p');
    flag.className = 'flag';
    flag.textContent = payload.web_enabled
      ? 'no web results — answered from model knowledge'
      : 'web search is off — answered from model knowledge';
    card.append(flag);
  }

  ui.advice.prepend(card);
  trim(ui.advice);
}

function trim(container, max = 60) {
  while (container.childElementCount > max) container.lastElementChild.remove();
}

function renderNotes(notes) {
  latestNotes = notes;
  ui.notes.innerHTML = '';

  const hasAnything =
    notes.summary ||
    notes.decisions?.length ||
    notes.action_items?.length ||
    notes.open_questions?.length ||
    notes.topics?.length;

  if (!hasAnything) {
    ui.notes.innerHTML = '<p class="empty">Nothing worth noting yet.</p>';
    return;
  }

  if (notes.summary) {
    const p = document.createElement('p');
    p.className = 'notes-summary';
    p.textContent = notes.summary;
    ui.notes.append(p);
  }

  const listSection = (title, items) => {
    if (!items || !items.length) return;
    const section = document.createElement('div');
    section.className = 'notes-section';
    const h = document.createElement('h4');
    h.textContent = title;
    const ul = document.createElement('ul');
    items.forEach((item) => {
      const li = document.createElement('li');
      li.textContent = item;
      ul.append(li);
    });
    section.append(h, ul);
    ui.notes.append(section);
  };

  listSection('Decisions', notes.decisions);

  if (notes.action_items?.length) {
    const section = document.createElement('div');
    section.className = 'notes-section';
    const h = document.createElement('h4');
    h.textContent = 'Action items';
    const ul = document.createElement('ul');
    notes.action_items.forEach((item) => {
      const li = document.createElement('li');
      const who = document.createElement('span');
      who.className = 'who-chip';
      who.textContent = `${item.who || 'unassigned'}: `;
      li.append(who, document.createTextNode(item.what || ''));
      if (item.due) {
        const due = document.createElement('span');
        due.className = 'due-chip';
        due.textContent = ` (${item.due})`;
        li.append(due);
      }
      ul.append(li);
    });
    section.append(h, ul);
    ui.notes.append(section);
  }

  listSection('Open questions', notes.open_questions);

  if (notes.topics?.length) {
    const section = document.createElement('div');
    section.className = 'notes-section';
    const h = document.createElement('h4');
    h.textContent = 'Topics';
    section.append(h);
    notes.topics.forEach((topic) => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = topic;
      section.append(chip);
    });
    ui.notes.append(section);
  }

  ui.notesUpdated.textContent = `updated ${new Date().toLocaleTimeString()}`;
}

function notesToMarkdown(notes, userNotes) {
  const out = ['# Meeting notes', ''];
  if (notes.summary) out.push('## Summary', notes.summary, '');
  if (notes.decisions?.length) {
    out.push('## Decisions', ...notes.decisions.map((d) => `- ${d}`), '');
  }
  if (notes.action_items?.length) {
    out.push('## Action items');
    notes.action_items.forEach((item) => {
      const due = item.due ? ` — due ${item.due}` : '';
      out.push(`- **${item.who || 'unassigned'}**: ${item.what}${due}`);
    });
    out.push('');
  }
  if (notes.open_questions?.length) {
    out.push('## Open questions', ...notes.open_questions.map((q) => `- ${q}`), '');
  }
  if (notes.topics?.length) out.push('## Topics', notes.topics.join(', '), '');
  if (userNotes?.trim()) out.push('## My notes', userNotes.trim(), '');
  return out.join('\n');
}

function renderCost(cost) {
  ui.costTotal.textContent = usd(cost.total_usd);
  ui.costStt.textContent = usd(cost.stt_usd);
  ui.costLlm.textContent = usd(cost.llm_usd);
  ui.statAudio.textContent = `${Number(cost.audio_minutes || 0).toFixed(2)} min`;
  ui.statCalls.textContent = cost.llm_calls ?? 0;
  ui.statTokens.textContent = `${cost.prompt_tokens ?? 0} / ${cost.completion_tokens ?? 0}`;
  if (cost.stt_model) ui.sttModel.textContent = cost.stt_model;
}

function setRunning(isRunning) {
  running = isRunning;
  ui.btnStart.hidden = isRunning;
  ui.btnStart.disabled = isRunning;
  ui.btnStop.hidden = !isRunning;
  ui.btnStop.disabled = !isRunning;
  ui.setup.hidden = isRunning;
  streaming = isRunning && !!workletNode;
}

// ---------------------------------------------------------------- socket wiring

socket.on('__open', () => {
  ui.connDot.classList.add('on');
  log('connected to the server');
});

socket.on('__close', () => {
  ui.connDot.classList.remove('on', 'listening');
  ui.statusText.textContent = 'server disconnected';
  log('lost the server connection', true);
});

socket.on('snapshot', (snap) => {
  if (!snap || !snap.meeting_id) {
    setRunning(false);
    return;
  }
  // Arrives on a fresh page load and on every socket reconnect. Repaint from
  // the server's state either way; whether we are still capturing depends on
  // whether this page still owns the microphone, decided below.
  ui.transcript.innerHTML = '';
  (snap.segments || []).forEach(addSegment);
  if (!snap.segments?.length) {
    ui.transcript.innerHTML = '<p class="empty">Nothing transcribed yet.</p>';
  }
  ui.advice.innerHTML = '';
  (snap.cards || []).forEach((card) => {
    if (card.kind === 'answer') addAnswerCard(card);
    else addAdviceCard(card);
  });
  if (!snap.cards?.length) {
    ui.advice.innerHTML = '<p class="empty">No suggestions yet.</p>';
  }
  if (snap.notes) renderNotes(snap.notes);
  if (snap.user_notes) ui.userNotes.value = snap.user_notes;
  if (snap.summary) ui.summary.textContent = snap.summary;
  if (snap.cost) {
    renderCost(snap.cost);
    startedAtMs = Date.now() - (snap.cost.elapsed_seconds || 0) * 1000;
  }
  ui.inTitle.value = snap.title || '';
  ui.inBrief.value = snap.brief || '';

  if (snap.running) {
    // setRunning resumes streaming only if the worklet is still alive, which
    // tells the two cases apart: a dropped-and-restored socket on this page
    // (keep sending audio) versus a reloaded tab (the mic went with the old
    // page, so the meeting continues on the server but deaf to this one).
    setRunning(true);
    if (workletNode) {
      ui.statusText.textContent = 'meeting running';
      log('socket reconnected; still sending audio');
    } else {
      ui.statusText.textContent = 'meeting running — this tab is not sending audio';
      log('reattached to a running meeting, but the microphone was lost with the '
          + 'previous page; stop and start again to resume capture', true);
    }
  } else {
    setRunning(false);
    ui.statusText.textContent = 'last meeting finished';
  }
});

socket.on('meeting_started', (snap) => {
  setRunning(true);
  startedAtMs = Date.now();
  ui.transcript.innerHTML = '<p class="empty">Listening…</p>';
  ui.advice.innerHTML = '<p class="empty">No suggestions yet.</p>';
  ui.notes.innerHTML = '<p class="empty">Nothing worth noting yet.</p>';
  log(`meeting ${snap.meeting_id} started`);
});

socket.on('meeting_stopped', (payload) => {
  setRunning(false);
  stopCapture();
  ui.connDot.classList.remove('listening');
  ui.statusText.textContent = 'finished';
  ui.interim.textContent = '';
  if (payload.notes) renderNotes(payload.notes);
  log(`meeting ${payload.meeting_id} finished and saved`);
});

socket.on('status', (status) => {
  const detail = status.detail ? ` — ${status.detail}` : '';
  ui.statusText.textContent = `${status.state}${detail}`;
  if (status.state === 'listening') {
    ui.connDot.classList.add('listening');
    streaming = !!workletNode;
  } else {
    ui.connDot.classList.remove('listening');
  }
  if (status.state) log(`status: ${status.state}${detail}`);
});

socket.on('interim', (payload) => {
  ui.interim.textContent = payload.text || '';
});

socket.on('segment', addSegment);
socket.on('advice', addAdviceCard);
socket.on('answer', addAnswerCard);
socket.on('notes', (payload) => renderNotes(payload.notes || {}));
socket.on('summary', (payload) => { ui.summary.textContent = payload.summary || ''; });
socket.on('cost', renderCost);
socket.on('usage', () => {});

socket.on('copilot_error', (payload) => {
  log(`copilot (${payload.where}): ${payload.message}`, true);
});

socket.on('error', (payload) => {
  const message = payload?.message || String(payload);
  ui.statusText.textContent = 'error';
  log(message, true);
  if (!running) ui.btnStart.disabled = false;
});

window.addEventListener('beforeunload', () => {
  if (running) stopCapture();
});
