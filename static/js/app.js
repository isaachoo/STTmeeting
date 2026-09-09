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
  costTotal: el('cost-total'),
  btnSession: el('btn-session'),
  btnStart: el('btn-start'),
  btnPause: el('btn-pause'),
  btnStop: el('btn-stop'),
  btnReview: el('btn-review'),
  btnAsk: el('btn-ask'),
  drawer: el('drawer'),

  askDialog: el('ask-dialog'),
  askThread: el('ask-thread'),
  askQuick: el('ask-quick'),
  askDialogForm: el('ask-dialog-form'),
  askDialogInput: el('ask-dialog-input'),
  askWeb: el('ask-web'),
  btnAskClose: el('btn-ask-close'),
  btnAskExpand: el('btn-ask-expand'),

  resumeBanner: el('resume-banner'),
  resumeText: el('resume-text'),
  btnResume: el('btn-resume'),
  btnFinishInterrupted: el('btn-finish-interrupted'),

  savedBrief: el('in-saved-brief'),
  btnBriefLoad: el('btn-brief-load'),
  btnBriefSave: el('btn-brief-save'),
  btnBriefDelete: el('btn-brief-delete'),
  briefHint: el('brief-hint'),

  setup: el('setup'),
  inTitle: el('in-title'),
  inAgenda: el('in-agenda'),
  inGoal: el('in-goal'),
  inContext: el('in-context'),
  inGlossary: el('in-glossary'),
  inMicMode: el('in-mic-mode'),
  inMyRole: el('in-my-role'),
  inProvider: el('in-provider'),
  inLanguage: el('in-language'),
  inModel: el('in-model'),
  attendees: el('attendees'),
  btnAddAttendee: el('btn-add-attendee'),

  transcript: el('transcript'),
  interim: el('interim'),
  autoscroll: el('chk-autoscroll'),
  speakerChips: el('speaker-chips'),
  filterBar: el('filter-bar'),
  filterText: el('filter-text'),
  btnClearFilter: el('btn-clear-filter'),
  suggestion: el('speaker-suggestion'),

  advice: el('advice'),
  askForm: el('ask-form'),
  inAsk: el('in-ask'),

  attendee: el('attendee'),
  chkSpeak: el('chk-speak'),

  notes: el('notes'),
  notesUpdated: el('notes-updated'),
  userNotes: el('user-notes'),
  notesSaveHint: el('notes-save-hint'),

  summary: el('summary'),
  log: el('log'),
  history: el('history'),
  btnRefreshHistory: el('btn-refresh-history'),
  keysBanner: el('keys-banner'),
  keysBannerText: el('keys-banner-text'),
  btnOpenKeys: el('btn-open-keys'),
  keysForm: el('keys-form'),
  keysPath: el('keys-path'),
  keysHint: el('keys-hint'),
  btnSaveKeys: el('btn-save-keys'),
  btnSaveMd: el('btn-save-md'),
  btnSaveJson: el('btn-save-json'),
  audioNote: el('audio-note'),

  dCostTotal: el('d-cost-total'),
  dCostStt: el('d-cost-stt'),
  dCostLlm: el('d-cost-llm'),
  dStatAudio: el('d-stat-audio'),
  dStatCalls: el('d-stat-calls'),
  dStatTokens: el('d-stat-tokens'),

  popover: el('name-popover'),
  popLabel: el('pop-label'),
  popScope: el('pop-scope'),
  popRoster: el('pop-roster'),
  popForm: el('pop-form'),
  popName: el('pop-name'),
  popClear: el('pop-clear'),
};

let audioContext = null;
let mediaStream = null;
let workletNode = null;
let streaming = false;
let running = false;
let startedAtMs = null;
let meetingId = null;
let speakerNames = {}; // diarisation index (as string) -> name
let knownSpeakers = new Set();
let rosterNames = []; // from the pre-meeting attendee list
let popoverSpeaker = null;
let popoverSegment = null;
let paused = false;
let speakerFilter = new Set();   // empty = show everyone
let provider = 'deepgram';

// ------------------------------------------------------------------ utilities

function log(message, isError = false) {
  const line = document.createElement('div');
  if (isError) line.className = 'err';
  line.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
  ui.log.prepend(line);
  while (ui.log.childElementCount > 100) ui.log.lastElementChild.remove();
}

function clockFromSeconds(total) {
  const s = Math.max(0, Math.floor(total));
  const pad = (n) => String(n).padStart(2, '0');
  const h = Math.floor(s / 3600);
  return h > 0
    ? `${h}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`
    : `${pad(Math.floor(s / 60))}:${pad(s % 60)}`;
}

function usd(value) {
  return `$${Number(value || 0).toFixed(4)}`;
}

function timeOf(payload) {
  return new Date((payload.at || Date.now() / 1000) * 1000).toLocaleTimeString();
}

function clearIfEmpty(container) {
  const placeholder = container.querySelector('.empty');
  if (placeholder) placeholder.remove();
}

function trim(container, max = 60) {
  while (container.childElementCount > max) container.lastElementChild.remove();
}

function labelFor(speaker) {
  if (speaker === null || speaker === undefined) return '?';
  return speakerNames[String(speaker)] || `S${speaker + 1}`;
}

function copyButton(text, label = 'Copy') {
  const button = document.createElement('button');
  button.className = 'ghost small';
  button.textContent = label;
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied ✓';
      setTimeout(() => { button.textContent = label; }, 1200);
    } catch {
      log('clipboard is not available in this browser', true);
    }
  });
  return button;
}

setInterval(() => {
  if (running && startedAtMs) {
    ui.elapsed.textContent = clockFromSeconds((Date.now() - startedAtMs) / 1000);
  }
}, 500);

// ------------------------------------------------------- pre-meeting attendees

function addAttendeeRow(person = {}) {
  const index = ui.attendees.childElementCount;
  const row = document.createElement('div');
  row.className = 'attendee-row';

  const name = document.createElement('input');
  name.type = 'text';
  name.placeholder = 'Name';
  name.id = `attendee-name-${index}`;
  name.value = person.name || '';
  name.className = 'a-name';

  const role = document.createElement('input');
  role.type = 'text';
  role.placeholder = 'Role (optional)';
  role.value = person.role || '';
  role.className = 'a-role';

  const meWrap = document.createElement('label');
  meWrap.className = 'me-toggle';
  const me = document.createElement('input');
  me.type = 'checkbox';
  me.className = 'a-me';
  me.checked = !!person.is_me;
  // Exactly one person is the user.
  me.addEventListener('change', () => {
    if (!me.checked) return;
    ui.attendees.querySelectorAll('.a-me').forEach((other) => {
      if (other !== me) other.checked = false;
    });
  });
  meWrap.append(me, document.createTextNode('me'));

  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'ghost drop';
  remove.textContent = '×';
  remove.title = 'Remove';
  remove.addEventListener('click', () => row.remove());

  row.append(name, role, meWrap, remove);
  ui.attendees.append(row);
  return row;
}

ui.btnAddAttendee.addEventListener('click', () => addAttendeeRow().querySelector('.a-name').focus());

function collectBrief() {
  const attendees = [...ui.attendees.querySelectorAll('.attendee-row')]
    .map((row) => ({
      name: row.querySelector('.a-name').value.trim(),
      role: row.querySelector('.a-role').value.trim(),
      is_me: row.querySelector('.a-me').checked,
    }))
    .filter((person) => person.name);

  rosterNames = attendees.map((person) => person.name);

  return {
    title: ui.inTitle.value.trim(),
    agenda: ui.inAgenda.value.trim(),
    my_goal: ui.inGoal.value.trim(),
    my_role: ui.inMyRole.value.trim(),
    context: ui.inContext.value.trim(),
    attendees,
    glossary: ui.inGlossary.value
      .split(/[,\n]/)
      .map((term) => term.trim())
      .filter(Boolean),
  };
}

function fillBrief(brief) {
  if (!brief) return;
  ui.inTitle.value = brief.title || '';
  ui.inAgenda.value = brief.agenda || '';
  ui.inGoal.value = brief.my_goal || '';
  ui.inMyRole.value = brief.my_role || '';
  ui.inContext.value = brief.context || '';
  ui.inGlossary.value = (brief.glossary || []).join(', ');
  ui.attendees.innerHTML = '';
  (brief.attendees || []).forEach(addAttendeeRow);
  rosterNames = (brief.attendees || []).map((person) => person.name);
  if (!ui.attendees.childElementCount) addAttendeeRow();
}

// --------------------------------------------------------------- microphone

async function startCapture() {
  const roomMode = ui.inMicMode.value === 'room';
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      // Room mode hands Deepgram the untouched signal, which is usually better
      // when the mic is picking up several people across a table -- noise
      // suppression tends to eat the quieter, further-away voices.
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

ui.btnStart.addEventListener('click', () => startMeeting(null));

/* Start a new meeting, or -- with `resumeId` -- carry on an interrupted one.
 * Same microphone flow either way; the server tells them apart. */
async function startMeeting(resumeId) {
  ui.btnStart.disabled = true;
  ui.btnResume.disabled = true;
  ui.statusText.textContent = 'asking for the microphone…';
  try {
    const brief = collectBrief();
    const sampleRate = await startCapture();
    log(`microphone open at ${sampleRate} Hz`);
    const payload = {
      brief,
      sample_rate: sampleRate,
      language: ui.inLanguage.value,
      model: ui.inModel.value,
      provider: ui.inProvider.value,
    };
    if (resumeId) payload.resume_meeting_id = resumeId;
    socket.emit('start_meeting', payload);
    // Remember the choice for next time.
    localStorage.setItem('stt_language', ui.inLanguage.value);
    localStorage.setItem('stt_model', ui.inModel.value);
    localStorage.setItem('stt_provider', ui.inProvider.value);
  } catch (err) {
    stopCapture();
    ui.btnStart.disabled = false;
    ui.btnResume.disabled = false;
    ui.statusText.textContent = 'microphone blocked';
    log(`could not open the microphone: ${err.message}`, true);
  }
}

ui.btnPause.addEventListener('click', () => {
  // Optimistic locally so the button responds instantly; the server's `paused`
  // event is the authority and will correct this if it disagrees.
  setPaused(!paused);
  socket.emit('pause', { paused });
});

function setPaused(next) {
  paused = next;
  ui.btnPause.textContent = paused ? 'Resume' : 'Pause';
  ui.btnPause.classList.toggle('primary', paused);
  // Stop feeding the socket. The worklet keeps running, so resuming is instant
  // and does not ask for the microphone again.
  streaming = running && !paused && !!workletNode;
  if (paused) {
    ui.interim.textContent = '';
    ui.connDot.classList.remove('listening');
  }
}

function addPauseMarker(payload) {
  clearIfEmpty(ui.transcript);
  const marker = document.createElement('p');
  marker.className = 'pause-marker';
  const at = clockFromSeconds(payload.at || 0);
  if (payload.resumed) {
    // Continued after an interruption: the gap above this line is the time
    // the app was not running, not a silence in the room.
    marker.textContent = `▶ continued after an interruption at ${at}`;
  } else {
    marker.textContent = payload.paused ? `⏸ paused at ${at}` : `▶ resumed at ${at}`;
  }
  ui.transcript.append(marker);
  if (ui.autoscroll.checked) ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

ui.btnStop.addEventListener('click', () => {
  ui.btnStop.disabled = true;
  stopCapture();
  socket.emit('stop_meeting');
  ui.statusText.textContent = 'wrapping up…';
});

ui.btnSession.addEventListener('click', () => {
  ui.drawer.hidden = !ui.drawer.hidden;
  if (!ui.drawer.hidden) {
    loadHistory();
    loadKeys();
  }
});

ui.askForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = ui.inAsk.value.trim();
  if (!question) return;
  askCopilot(question, false);
  ui.inAsk.value = '';
});

// ------------------------------------------------------------ ask the copilot
//
// One question, one answer, from wherever it was typed. Answers arrive as
// `answer` events and are shown twice: as a card in the Copilot panel, and as
// a turn in the dialog's conversation thread, which is the roomier place to
// read a summary or ask a follow-up.

let askPending = null; // the thread element waiting for the next answer

function askCopilot(question, web) {
  if (!running) {
    log('start a meeting first — there is nothing to ask about yet', true);
    return;
  }
  socket.emit('ask', { question, web: !!web });
  log(`asked: ${question}${web ? ' (with web search)' : ''}`);

  if (ui.askThread.querySelector('.empty')) ui.askThread.innerHTML = '';
  const turn = document.createElement('div');
  turn.className = 'chat-turn pending';
  const q = document.createElement('div');
  q.className = 'chat-q';
  q.textContent = question;
  const a = document.createElement('div');
  a.className = 'chat-a muted';
  a.textContent = web ? 'searching and reading the transcript…' : 'reading the transcript…';
  turn.append(q, a);
  ui.askThread.append(turn);
  ui.askThread.scrollTop = ui.askThread.scrollHeight;
  askPending = turn;
}

/* Escape the model's text, then turn [#42] into a button that jumps to line 42.
 * Escaping first matters: the answer goes into innerHTML. */
function answerHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML.replace(
    /\[#(\d+)\]/g,
    (whole, index) => `<button class="cite" data-line="${index}" title="jump to line ${index}">#${index}</button>`
  );
}

function wireCitations(container) {
  container.querySelectorAll('button.cite').forEach((button) => {
    button.addEventListener('click', () => jumpToLine(Number(button.dataset.line)));
  });
}

function jumpToLine(index) {
  const line = ui.transcript.querySelector(`.line[data-index="${index}"]`);
  if (!line) return;
  if (line.classList.contains('hidden-by-filter')) clearSpeakerFilter();
  ui.autoscroll.checked = false; // otherwise the next segment yanks it away
  line.scrollIntoView({ behavior: 'smooth', block: 'center' });
  line.classList.remove('flash');
  void line.offsetWidth; // restart the animation if the same line is hit twice
  line.classList.add('flash');
}

function askThreadTurn(payload) {
  const turn = askPending && askPending.isConnected ? askPending : document.createElement('div');
  askPending = null;
  turn.className = 'chat-turn';
  turn.innerHTML = '';

  const q = document.createElement('div');
  q.className = 'chat-q';
  q.textContent = payload.question;
  const a = document.createElement('div');
  a.className = 'chat-a';
  a.innerHTML = answerHtml(payload.answer);
  wireCitations(a);
  turn.append(q, a);

  const sources = payload.sources || [];
  if (sources.length) turn.append(sourceList(payload));
  const meta = document.createElement('div');
  meta.className = 'chat-meta';
  const bits = [timeOf(payload)];
  if (payload.web_requested && !sources.length) bits.push('web search found nothing usable');
  if ((payload.cited || []).length) bits.push(`${payload.cited.length} line${payload.cited.length > 1 ? 's' : ''} cited`);
  meta.textContent = bits.join(' · ');
  turn.append(meta);

  if (!turn.isConnected) {
    if (ui.askThread.querySelector('.empty')) ui.askThread.innerHTML = '';
    ui.askThread.append(turn);
  }
  ui.askThread.scrollTop = ui.askThread.scrollHeight;
}

function openAskDialog(prefill = '') {
  if (!running) {
    log('the copilot can only be asked during a meeting', true);
    return;
  }
  if (!ui.askDialog.open) ui.askDialog.showModal();
  if (prefill) ui.askDialogInput.value = prefill;
  ui.askDialogInput.focus();
  ui.askThread.scrollTop = ui.askThread.scrollHeight;
}

ui.btnAsk.addEventListener('click', () => openAskDialog());
ui.btnAskExpand.addEventListener('click', () => openAskDialog(ui.inAsk.value.trim()));
ui.btnAskClose.addEventListener('click', () => ui.askDialog.close());
ui.askDialog.addEventListener('click', (event) => {
  // A click on the backdrop (outside the dialog's box) closes it.
  const box = ui.askDialog.getBoundingClientRect();
  const inside = event.clientX >= box.left && event.clientX <= box.right
    && event.clientY >= box.top && event.clientY <= box.bottom;
  if (!inside) ui.askDialog.close();
});
ui.askDialogForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = ui.askDialogInput.value.trim();
  if (!question) return;
  askCopilot(question, ui.askWeb.checked);
  ui.askDialogInput.value = '';
});
ui.askQuick.querySelectorAll('button[data-q]').forEach((button) => {
  button.addEventListener('click', () => askCopilot(button.dataset.q, false));
});
document.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    if (ui.askDialog.open) ui.askDialog.close(); else openAskDialog();
  }
});

// -------------------------------------------------------- interrupted meetings
//
// A meeting that was never stopped -- the app or the laptop died -- is offered
// for resuming. Everything said is already in the database; resuming continues
// the same meeting with the copilot's memory restored.

let interrupted = null; // the most recent unfinished meeting, if any

async function checkInterrupted() {
  try {
    const response = await fetch('/api/meetings');
    const meetings = await response.json();
    interrupted = meetings.find((m) => m.interrupted) || null;
  } catch (err) {
    interrupted = null;
  }
  renderResumeBanner();
}

function renderResumeBanner() {
  if (!interrupted || running) {
    ui.resumeBanner.hidden = true;
    return;
  }
  const when = interrupted.started_at
    ? new Date(interrupted.started_at * 1000).toLocaleString() : '';
  const title = interrupted.title || `Meeting ${interrupted.id}`;
  ui.resumeText.textContent =
    `“${title}” (${when}, ${interrupted.segments || 0} lines) was never stopped. `
    + 'Everything said is saved. Continue it, or close it and review later.';
  ui.btnResume.disabled = false;
  ui.resumeBanner.hidden = false;
}

async function resumeMeeting(meeting) {
  // The stored brief goes into the form so it can be adjusted before continuing;
  // the transcriber settings are whatever the form says now.
  fillBrief(meeting.brief_json || {});
  if (meeting.language) ui.inLanguage.value = meeting.language;
  if (meeting.provider) ui.inProvider.value = meeting.provider;
  await startMeeting(meeting.id);
}

async function finishInterrupted(meeting) {
  try {
    const response = await fetch(`/api/meetings/${meeting.id}/finish`, { method: 'POST' });
    if (!response.ok) throw new Error((await response.json()).error || response.statusText);
    log(`meeting ${meeting.id} closed — it is in Past meetings for review`);
  } catch (err) {
    log(`could not close meeting ${meeting.id}: ${err.message}`, true);
  }
  await checkInterrupted();
  loadHistory();
}

ui.btnResume.addEventListener('click', () => { if (interrupted) resumeMeeting(interrupted); });
ui.btnFinishInterrupted.addEventListener('click', () => { if (interrupted) finishInterrupted(interrupted); });

// ------------------------------------------------------------- saved briefs
//
// A meeting background typed in advance and kept by name. The list also offers
// the brief of every past meeting, so a recurring meeting is one click.

let briefOptions = { saved: [], meetings: [] };

async function loadBriefOptions() {
  try {
    const [briefsRes, meetingsRes] = await Promise.all([fetch('/api/briefs'), fetch('/api/meetings')]);
    briefOptions.saved = (await briefsRes.json()).briefs || [];
    briefOptions.meetings = ((await meetingsRes.json()) || [])
      .filter((m) => m.brief_json && Object.values(m.brief_json).some((v) => v && String(v).length && !(Array.isArray(v) && !v.length)));
  } catch (err) {
    log(`could not load saved backgrounds: ${err.message}`, true);
    return;
  }
  const current = ui.savedBrief.value;
  ui.savedBrief.innerHTML = '<option value="">Saved backgrounds…</option>';
  if (briefOptions.saved.length) {
    const group = document.createElement('optgroup');
    group.label = 'Saved';
    briefOptions.saved.forEach((item) => {
      const option = document.createElement('option');
      option.value = `saved:${item.id}`;
      option.textContent = item.name;
      group.append(option);
    });
    ui.savedBrief.append(group);
  }
  if (briefOptions.meetings.length) {
    const group = document.createElement('optgroup');
    group.label = 'From a past meeting';
    briefOptions.meetings.slice(0, 30).forEach((meeting) => {
      const option = document.createElement('option');
      option.value = `meeting:${meeting.id}`;
      const when = meeting.started_at ? new Date(meeting.started_at * 1000).toLocaleDateString() : '';
      option.textContent = `${meeting.title || `Meeting ${meeting.id}`} (${when})`;
      group.append(option);
    });
    ui.savedBrief.append(group);
  }
  if ([...ui.savedBrief.options].some((o) => o.value === current)) ui.savedBrief.value = current;
  ui.btnBriefDelete.hidden = !ui.savedBrief.value.startsWith('saved:');
}

function selectedBrief() {
  const value = ui.savedBrief.value;
  if (value.startsWith('saved:')) {
    const item = briefOptions.saved.find((b) => String(b.id) === value.slice(6));
    return item ? { brief: item.brief, name: item.name, saved: item } : null;
  }
  if (value.startsWith('meeting:')) {
    const meeting = briefOptions.meetings.find((m) => String(m.id) === value.slice(8));
    return meeting ? { brief: meeting.brief_json, name: meeting.title } : null;
  }
  return null;
}

ui.savedBrief.addEventListener('change', () => {
  ui.btnBriefDelete.hidden = !ui.savedBrief.value.startsWith('saved:');
});

ui.btnBriefLoad.addEventListener('click', () => {
  const chosen = selectedBrief();
  if (!chosen) {
    ui.briefHint.textContent = 'pick one first';
    return;
  }
  fillBrief(chosen.brief);
  saveBriefDraft();
  ui.briefHint.textContent = `loaded “${chosen.name || 'brief'}”`;
  setTimeout(() => { ui.briefHint.textContent = ''; }, 2500);
});

ui.btnBriefSave.addEventListener('click', async () => {
  const brief = collectBrief();
  const suggested = (selectedBrief() && selectedBrief().saved ? selectedBrief().name : brief.title) || '';
  const name = prompt('Save this background as:', suggested);
  if (name === null) return;
  if (!name.trim()) {
    ui.briefHint.textContent = 'a name is needed';
    return;
  }
  try {
    const response = await fetch('/api/briefs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim(), brief }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    await loadBriefOptions();
    ui.savedBrief.value = `saved:${data.saved.id}`;
    ui.btnBriefDelete.hidden = false;
    ui.briefHint.textContent = `saved “${data.saved.name}”`;
    setTimeout(() => { ui.briefHint.textContent = ''; }, 2500);
  } catch (err) {
    ui.briefHint.textContent = `could not save: ${err.message}`;
  }
});

ui.btnBriefDelete.addEventListener('click', async () => {
  const chosen = selectedBrief();
  if (!chosen || !chosen.saved) return;
  if (!confirm(`Delete the saved background “${chosen.name}”?`)) return;
  try {
    const response = await fetch(`/api/briefs/${chosen.saved.id}`, { method: 'DELETE' });
    if (!response.ok) throw new Error((await response.json()).error || response.statusText);
    await loadBriefOptions();
    ui.briefHint.textContent = 'deleted';
    setTimeout(() => { ui.briefHint.textContent = ''; }, 2000);
  } catch (err) {
    ui.briefHint.textContent = `could not delete: ${err.message}`;
  }
});

/* Whatever is typed into the form survives a reload, so preparation done an
 * hour before the meeting is not lost to an accidental refresh. */
const DRAFT_KEY = 'brief_draft';
let draftTimer = null;
function saveBriefDraft() {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(() => {
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify(collectBrief())); } catch (err) { /* storage full or blocked */ }
  }, 400);
}
function restoreBriefDraft() {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return;
    const draft = JSON.parse(raw);
    if (draft && Object.values(draft).some((v) => (Array.isArray(v) ? v.length : v))) fillBrief(draft);
  } catch (err) { /* a corrupt draft is not worth an error */ }
}
ui.setup.addEventListener('input', saveBriefDraft);
ui.setup.addEventListener('change', saveBriefDraft);

let notesSaveTimer = null;
ui.userNotes.addEventListener('input', () => {
  ui.notesSaveHint.textContent = 'unsaved…';
  clearTimeout(notesSaveTimer);
  notesSaveTimer = setTimeout(() => {
    socket.emit('user_notes', { text: ui.userNotes.value });
    ui.notesSaveHint.textContent = `saved ${new Date().toLocaleTimeString()}`;
  }, 800);
});

function saveSession(suffix) {
  if (!meetingId) {
    log('no meeting to save yet', true);
    return;
  }
  // A plain navigation: the server sets Content-Disposition, the browser saves.
  window.location = `/api/meetings/${meetingId}/export.${suffix}`;
}

// ------------------------------------------------------------------ API keys

/* The form only ever shows a masked value, so an empty field means "leave this
 * alone". Saving must never wipe a key the user did not touch. */
async function loadKeys() {
  try {
    const response = await fetch('/api/settings');
    const data = await response.json();
    ui.keysPath.textContent = data.path;
    ui.keysForm.innerHTML = '';

    data.fields.forEach((field) => {
      const name = field.name;
      const wrap = document.createElement('div');
      wrap.className = 'keys-field';

      const label = document.createElement('label');
      label.textContent = field.label;
      label.htmlFor = `key-${name}`;

      const input = document.createElement('input');
      input.type = 'text';
      input.id = `key-${name}`;
      input.dataset.key = name;
      input.autocomplete = 'off';
      if (field.secret) {
        input.placeholder = field.set ? field.value : 'not set';
      } else {
        input.value = field.value || '';
        // Remember what it arrived as, so an untouched field is not "saved".
        input.dataset.original = field.value || '';
      }

      const source = document.createElement('span');
      source.className = `source${field.source === 'saved in the app' ? ' saved' : ''}`;
      source.textContent = field.set ? field.source : 'not set';

      wrap.append(label, input, source);
      ui.keysForm.append(wrap);
    });
  } catch (err) {
    log(`could not load settings: ${err.message}`, true);
  }
}

async function saveKeys() {
  const payload = {};
  ui.keysForm.querySelectorAll('input[data-key]').forEach((input) => {
    const value = input.value.trim();
    if (!value) return;                                   // blank = leave alone
    if (value === (input.dataset.original || '')) return;  // shown, not changed
    payload[input.dataset.key] = value;
  });
  if (!Object.keys(payload).length) {
    ui.keysHint.textContent = 'nothing to save';
    return;
  }

  ui.btnSaveKeys.disabled = true;
  ui.keysHint.textContent = 'saving…';
  try {
    const response = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);

    ui.keysHint.textContent = data.changed.length
      ? `saved ${data.changed.length} value${data.changed.length === 1 ? '' : 's'}`
      : 'no change';
    applyMissingKeys(data.missing_keys || []);
    await loadKeys();
    log(`settings saved: ${data.changed.join(', ') || 'no change'}`);
  } catch (err) {
    ui.keysHint.textContent = `failed: ${err.message}`;
    log(`could not save settings: ${err.message}`, true);
  } finally {
    ui.btnSaveKeys.disabled = false;
  }
}

function applyMissingKeys(missing) {
  if (!missing.length) {
    ui.keysBanner.hidden = true;
    return;
  }
  ui.keysBannerText.innerHTML = `Missing: <strong>${missing.join(', ')}</strong>.`;
  ui.keysBanner.hidden = false;
}

ui.btnSaveKeys.addEventListener('click', saveKeys);
ui.btnOpenKeys.addEventListener('click', () => {
  ui.drawer.hidden = false;
  loadKeys();
  loadHistory();
  el('key-DEEPGRAM_API_KEY')?.focus();
});

ui.btnSaveMd.addEventListener('click', () => saveSession('md'));
ui.btnSaveJson.addEventListener('click', () => saveSession('json'));
ui.btnRefreshHistory.addEventListener('click', loadHistory);

let historyRequest = 0;
async function loadHistory() {
  // Two loads can overlap (open the drawer, then close a meeting a moment
  // later); only the newest response may paint, or a stale one wins.
  const mine = ++historyRequest;
  try {
    const response = await fetch('/api/meetings');
    const meetings = await response.json();
    if (mine !== historyRequest) return;
    ui.history.innerHTML = '';
    if (!meetings.length) {
      ui.history.innerHTML = '<p class="muted small">Nothing saved yet.</p>';
      return;
    }
    meetings.forEach((meeting) => {
      const row = document.createElement('div');
      row.className = 'history-row';

      const left = document.createElement('div');
      const title = document.createElement('div');
      title.textContent = meeting.title || `Meeting ${meeting.id}`;
      const meta = document.createElement('div');
      meta.className = 'meta';
      const when = meeting.started_at
        ? new Date(meeting.started_at * 1000).toLocaleString()
        : '';
      const mins = ((meeting.audio_seconds || 0) / 60).toFixed(0);
      meta.textContent = `${when} · ${mins} min · ${meeting.segments || 0} lines`;
      left.append(title, meta);

      if (meeting.interrupted) {
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.textContent = 'interrupted';
        title.append(badge);
      } else if (meeting.running) {
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.textContent = 'running';
        title.append(badge);
      }

      const links = document.createElement('div');
      if (meeting.interrupted && !running) {
        const resume = document.createElement('a');
        resume.href = '#';
        resume.className = 'act';
        resume.textContent = 'Resume';
        resume.title = 'Continue this meeting where it stopped';
        resume.addEventListener('click', (event) => {
          event.preventDefault();
          ui.drawer.hidden = true;
          resumeMeeting(meeting);
        });
        const finish = document.createElement('a');
        finish.href = '#';
        finish.className = 'act';
        finish.textContent = 'Close';
        finish.title = 'Mark it finished without resuming';
        finish.addEventListener('click', (event) => {
          event.preventDefault();
          finishInterrupted(meeting);
        });
        links.append(resume, finish);
      }
      const review = document.createElement('a');
      review.href = `/review/${meeting.id}`;
      review.textContent = 'Review';
      review.title = 'Ask questions, draft actions, write the minutes';
      links.append(review);
      ['md', 'json'].forEach((suffix) => {
        const link = document.createElement('a');
        link.href = `/api/meetings/${meeting.id}/export.${suffix}`;
        link.textContent = suffix.toUpperCase();
        link.style.marginLeft = '8px';
        links.append(link);
      });

      row.append(left, links);
      ui.history.append(row);
    });
  } catch (err) {
    log(`could not load history: ${err.message}`, true);
  }
}

// ------------------------------------------------------- speaker names

function renderSpeakerChips() {
  ui.speakerChips.innerHTML = '';
  if (!knownSpeakers.size && running) {
    // The local engine does not diarise, so no voice-level chips ever appear.
    // Without this the panel head just looks broken.
    const hint = document.createElement('span');
    hint.className = 'muted small';
    hint.textContent = provider === 'local'
      ? 'no speaker separation — click a line to name it'
      : 'listening for voices…';
    ui.speakerChips.append(hint);
    return;
  }
  [...knownSpeakers].sort((a, b) => a - b).forEach((speaker) => {
    const named = speakerNames[String(speaker)];
    const chip = document.createElement('span');
    const active = speakerFilter.has(speaker);
    chip.className = `chip-speaker s${speaker % 4}${named ? '' : ' unnamed'}`
      + (active ? ' active' : '')
      + (speakerFilter.size && !active ? ' dimmed' : '');

    // The label filters; the pencil renames. Two jobs on one chip needs two
    // targets, or every rename starts with an accidental filter.
    const label = document.createElement('button');
    label.className = 'chip-label';
    label.textContent = named || `S${speaker + 1}`;
    label.title = 'Click to show only this speaker';
    label.addEventListener('click', () => toggleSpeakerFilter(speaker));

    const rename = document.createElement('button');
    rename.className = 'chip-rename';
    rename.textContent = named ? '✎' : '✎ name';
    rename.title = 'Name this voice';
    rename.addEventListener('click', (event) => {
      event.stopPropagation();
      openPopover(speaker, chip);
    });

    chip.append(label, rename);
    ui.speakerChips.append(chip);
  });
}

/* Filtering is a VIEW ONLY. The server keeps sending every line and the copilot
 * keeps seeing the whole conversation -- otherwise the advice would quietly
 * degrade whenever the screen was filtered. */
function toggleSpeakerFilter(speaker) {
  if (speakerFilter.has(speaker)) speakerFilter.delete(speaker);
  else speakerFilter.add(speaker);
  applySpeakerFilter();
  renderSpeakerChips();
}

function clearSpeakerFilter() {
  speakerFilter.clear();
  applySpeakerFilter();
  renderSpeakerChips();
}

function lineIsVisible(speaker) {
  if (!speakerFilter.size) return true;
  return speaker !== '' && speakerFilter.has(Number(speaker));
}

function applySpeakerFilter() {
  ui.transcript.querySelectorAll('.line').forEach((line) => {
    line.classList.toggle('hidden-by-filter', !lineIsVisible(line.dataset.speaker));
  });
  ui.transcript.querySelectorAll('.pause-marker').forEach((marker) => {
    marker.classList.toggle('hidden-by-filter', speakerFilter.size > 0);
  });

  if (!speakerFilter.size) {
    ui.filterBar.hidden = true;
    return;
  }
  const names = [...speakerFilter].sort((a, b) => a - b).map(labelFor).join(', ');
  const shown = ui.transcript.querySelectorAll('.line:not(.hidden-by-filter)').length;
  ui.filterText.textContent = `Showing ${shown} line${shown === 1 ? '' : 's'} from ${names}`
    + ' — the copilot still sees everything';
  ui.filterBar.hidden = false;
}

ui.btnClearFilter.addEventListener('click', clearSpeakerFilter);

function relabelTranscript() {
  ui.transcript.querySelectorAll('.line').forEach((line) => {
    // A per-line correction outranks the voice-level name: the user told us
    // this line specifically, so renaming the voice must not undo it.
    if (line.dataset.override) return;
    const speaker = line.dataset.speaker;
    if (speaker === '' || speaker === undefined) return;
    const who = line.querySelector('.who');
    if (who) who.textContent = labelFor(Number(speaker));
  });
}

function openPopover(speaker, anchor, segmentIndex = null) {
  popoverSpeaker = speaker;
  popoverSegment = segmentIndex;
  const isLine = segmentIndex !== null;
  ui.popLabel.textContent = isLine
    ? `this line (${labelFor(speaker)})`
    : (speaker === null || speaker === undefined ? '?' : `S${speaker + 1}`);
  ui.popScope.textContent = isLine
    ? 'Changes this line only — use the chip above to rename the whole voice.'
    : 'Renames every line from this voice.';
  ui.popName.value = isLine
    ? (lineOverride(segmentIndex) || '')
    : (speakerNames[String(speaker)] || '');

  // Offer the roster first: one click is the common case.
  ui.popRoster.innerHTML = '';
  const used = new Set(Object.values(speakerNames));
  rosterNames.forEach((name) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ghost small';
    button.textContent = name;
    if (used.has(name) && speakerNames[String(speaker)] !== name) {
      button.title = 'already assigned to another voice';
      button.style.opacity = '0.5';
    }
    button.addEventListener('click', () => applyName(name));
    ui.popRoster.append(button);
  });

  const box = anchor.getBoundingClientRect();
  ui.popover.hidden = false;
  const width = ui.popover.offsetWidth || 250;
  ui.popover.style.left = `${Math.max(8, Math.min(box.left, window.innerWidth - width - 8))}px`;
  ui.popover.style.top = `${box.bottom + window.scrollY + 6}px`;
  ui.popName.focus();
}

function closePopover() {
  ui.popover.hidden = true;
  popoverSpeaker = null;
  popoverSegment = null;
}

function lineOverride(index) {
  const line = ui.transcript.querySelector(`.line[data-index="${index}"]`);
  return line ? line.dataset.override || '' : '';
}

/* One popover, two scopes: a whole voice (from a chip) or a single line (from
 * the tag on that line). Per-line exists because diarisation splits one person
 * across two voices and merges two into one, which renaming a voice cannot fix. */
function applyName(name) {
  if (popoverSegment !== null) {
    socket.emit('name_segment', { index: popoverSegment, name });
  } else if (popoverSpeaker !== null && popoverSpeaker !== undefined) {
    socket.emit('name_speaker', { speaker: popoverSpeaker, name });
  }
  closePopover();
}

ui.popForm.addEventListener('submit', (event) => {
  event.preventDefault();
  applyName(ui.popName.value.trim());
});

ui.popClear.addEventListener('click', () => applyName(''));

document.addEventListener('click', (event) => {
  if (ui.popover.hidden) return;
  if (ui.popover.contains(event.target)) return;
  if (event.target.classList && event.target.classList.contains('chip-speaker')) return;
  closePopover();
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closePopover();
});

function renderSuggestion(suggestion) {
  if (!suggestion || !(suggestion.proposals || []).length) {
    ui.suggestion.hidden = true;
    return;
  }
  ui.suggestion.innerHTML = '';

  const heading = document.createElement('div');
  heading.innerHTML = '<strong>The copilot thinks it knows who is who.</strong>';
  const list = document.createElement('ul');
  suggestion.proposals.forEach((proposal) => {
    const item = document.createElement('li');
    item.textContent = `${proposal.label} is ${proposal.name} (${proposal.confidence} confidence)`;
    if (proposal.evidence) {
      const evidence = document.createElement('span');
      evidence.className = 'ev';
      evidence.textContent = ` — "${proposal.evidence}"`;
      item.append(evidence);
    }
    list.append(item);
  });

  const actions = document.createElement('div');
  actions.className = 'actions';
  const accept = document.createElement('button');
  accept.className = 'primary small';
  accept.textContent = 'Use these names';
  accept.addEventListener('click', () => socket.emit('speaker_suggestion', { accept: true }));
  const dismiss = document.createElement('button');
  dismiss.className = 'ghost small';
  dismiss.textContent = 'No thanks';
  dismiss.addEventListener('click', () => socket.emit('speaker_suggestion', { accept: false }));
  actions.append(accept, dismiss);

  ui.suggestion.append(heading, list, actions);
  ui.suggestion.hidden = false;
}

// ------------------------------------------------------------------ rendering

function addSegment(seg) {
  clearIfEmpty(ui.transcript);
  const nearBottom =
    ui.transcript.scrollHeight - ui.transcript.scrollTop - ui.transcript.clientHeight < 120;

  const line = document.createElement('p');
  line.className = 'line';
  line.dataset.speaker = seg.speaker === null || seg.speaker === undefined ? '' : seg.speaker;
  line.dataset.index = seg.index;
  if (seg.speaker_name) line.dataset.override = seg.speaker_name;

  const who = document.createElement('button');
  who.className = `who${seg.speaker !== null && seg.speaker !== undefined ? ` s${seg.speaker % 4}` : ''}`
    + (seg.speaker_name ? ' overridden' : '');
  who.textContent = seg.speaker_label || labelFor(seg.speaker);
  who.title = 'Wrong speaker? Click to fix just this line';
  who.addEventListener('click', (event) => {
    event.stopPropagation();
    openPopover(seg.speaker, who, seg.index);
  });

  const said = document.createElement('span');
  said.className = 'said';
  said.textContent = seg.text;

  const at = document.createElement('span');
  at.className = 'at';
  at.textContent = clockFromSeconds(seg.at || 0);

  line.append(who, said, at);
  if (!lineIsVisible(line.dataset.speaker)) line.classList.add('hidden-by-filter');
  ui.transcript.append(line);

  if (seg.speaker !== null && seg.speaker !== undefined && !knownSpeakers.has(seg.speaker)) {
    knownSpeakers.add(seg.speaker);
    renderSpeakerChips();
  }

  if (ui.autoscroll.checked && nearBottom) {
    ui.transcript.scrollTop = ui.transcript.scrollHeight;
  }
}

function addAdviceCard(payload) {
  clearIfEmpty(ui.advice);
  const card = document.createElement('div');
  card.className = 'card';

  const time = document.createElement('span');
  time.className = 'card-time';
  time.textContent = timeOf(payload);
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
  if ((payload.questions || []).length) {
    const heading = document.createElement('h4');
    heading.textContent = 'You could ask';
    const list = document.createElement('ul');
    payload.questions.forEach((question) => {
      const item = document.createElement('li');
      item.textContent = question;
      item.title = 'click to copy';
      item.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(question);
          const original = item.textContent;
          item.textContent = `${original}  ✓ copied`;
          setTimeout(() => { item.textContent = original; }, 1200);
        } catch {
          log('clipboard is not available in this browser', true);
        }
      });
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
  card.className = 'card answer mine';

  const time = document.createElement('span');
  time.className = 'card-time';
  time.textContent = timeOf(payload);

  const heading = document.createElement('h4');
  heading.textContent = 'You asked';

  const question = document.createElement('p');
  question.className = 'q';
  question.textContent = payload.question;

  const answer = document.createElement('p');
  answer.className = 'a';
  answer.innerHTML = answerHtml(payload.answer);
  wireCitations(answer);

  card.append(time, heading, question, answer);
  // A meeting question answered from the transcript has no web sources, and
  // that is not a shortcoming worth a flag; only show the flag when the user
  // asked for the web and got nothing.
  if ((payload.sources || []).length || payload.web_requested) card.append(sourceList(payload));
  ui.advice.prepend(card);
  trim(ui.advice);
}

function sourceList(payload) {
  if ((payload.sources || []).length) {
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
    return list;
  }
  const flag = document.createElement('p');
  flag.className = 'flag';
  flag.textContent = payload.web_enabled
    ? 'no web results — from model knowledge'
    : 'web search off — from model knowledge';
  return flag;
}

const KIND_LABELS = {
  answer: 'answering the room',
  question: 'wants to ask',
  clarification: 'wants to clarify',
  challenge: 'pushing back',
  info: 'adding information',
};

function addAttendeeTurn(payload) {
  clearIfEmpty(ui.attendee);
  const turn = document.createElement('div');
  turn.className = `turn${payload.urgency === 'high' ? ' high' : ''}`;

  const head = document.createElement('div');
  head.className = 'turn-head';
  const kind = document.createElement('span');
  kind.className = 'turn-kind';
  kind.textContent = KIND_LABELS[payload.kind] || payload.kind || 'says';
  const time = document.createElement('span');
  time.className = 'turn-time';
  time.textContent = timeOf(payload);
  head.append(kind, time);

  const say = document.createElement('p');
  say.className = 'say';
  say.textContent = payload.say;

  turn.append(head, say);

  if (payload.why) {
    const why = document.createElement('p');
    why.className = 'why';
    why.textContent = payload.why;
    turn.append(why);
  }

  const actions = document.createElement('div');
  actions.className = 'turn-actions';
  actions.append(copyButton(payload.say, 'Copy'));
  if (window.speechSynthesis) {
    const speakButton = document.createElement('button');
    speakButton.className = 'ghost small';
    speakButton.textContent = 'Speak';
    speakButton.addEventListener('click', () => speak(payload.say, true));
    actions.append(speakButton);
  }
  turn.append(actions);

  if (payload.kind === 'answer' || (payload.sources || []).length) {
    turn.append(sourceList(payload));
  }

  ui.attendee.prepend(turn);
  trim(ui.attendee, 40);
  speak(payload.say);
}

function speak(text, force = false) {
  if (!window.speechSynthesis || !text) return;
  if (!force && !ui.chkSpeak.checked) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'zh-HK';
  const voice = speechSynthesis
    .getVoices()
    .find((v) => (v.lang || '').toLowerCase().startsWith('zh-hk'));
  if (voice) utterance.voice = voice;
  speechSynthesis.speak(utterance);
}

function renderNotes(notes) {
  ui.notes.innerHTML = '';

  const hasAnything =
    notes.summary ||
    (notes.decisions || []).length ||
    (notes.action_items || []).length ||
    (notes.open_questions || []).length ||
    (notes.topics || []).length;

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

  if ((notes.action_items || []).length) {
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

  if ((notes.topics || []).length) {
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

function renderCost(cost) {
  ui.costTotal.textContent = usd(cost.total_usd);
  if (cost.provider) {
    const free = !cost.stt_usd_per_minute;
    ui.costTotal.title = free
      ? `${cost.provider}: transcription is free`
      : `${cost.provider}: $${cost.stt_usd_per_minute}/min of audio`;
  }
  ui.dCostTotal.textContent = usd(cost.total_usd);
  ui.dCostStt.textContent = usd(cost.stt_usd);
  ui.dCostLlm.textContent = usd(cost.llm_usd);
  ui.dStatAudio.textContent = `${Number(cost.audio_minutes || 0).toFixed(2)} min`;
  ui.dStatCalls.textContent = cost.llm_calls ?? 0;
  ui.dStatTokens.textContent = `${cost.prompt_tokens ?? 0} / ${cost.completion_tokens ?? 0}`;
}

function setRunning(isRunning) {
  running = isRunning;
  ui.btnStart.hidden = isRunning;
  ui.btnStart.disabled = isRunning;
  ui.btnPause.hidden = !isRunning;
  ui.btnStop.hidden = !isRunning;
  ui.btnStop.disabled = !isRunning;
  ui.btnAsk.hidden = !isRunning;
  ui.setup.hidden = isRunning;
  if (isRunning) {
    ui.resumeBanner.hidden = true;
    ui.btnReview.hidden = true;
  } else if (ui.askDialog.open) {
    ui.askDialog.close();
  }
  if (!isRunning) setPaused(false);
  streaming = isRunning && !paused && !!workletNode;
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

socket.on('snapshot', (snap) => paintSnapshot(snap, false));

/* Paint the whole page from a server snapshot. Used on connect and reconnect,
 * and -- with `fresh` -- when a meeting starts, because a *resumed* meeting
 * starts with a transcript, cards and notes already in it. */
function paintSnapshot(snap, fresh) {
  if (!snap || !snap.meeting_id) {
    setRunning(false);
    if (!ui.attendees.childElementCount) addAttendeeRow();
    return;
  }
  // Arrives on a fresh page load and on every socket reconnect. Repaint from
  // the server's state either way; whether we are still capturing depends on
  // whether this page still owns the microphone, decided below.
  meetingId = snap.meeting_id;
  speakerNames = snap.speaker_names || {};
  knownSpeakers = new Set();
  speakerFilter.clear();
  ui.filterBar.hidden = true;
  // A running meeting's brief always wins. After a meeting has finished, keep
  // whatever the user has started typing for the next one instead.
  let hasDraft = false;
  try { hasDraft = !!localStorage.getItem(DRAFT_KEY); } catch (err) { hasDraft = false; }
  if (snap.running || !hasDraft) fillBrief(snap.brief);
  if (snap.language) ui.inLanguage.value = snap.language;
  if (snap.provider) {
    provider = snap.provider;
    ui.inProvider.value = snap.provider;
  }

  ui.transcript.innerHTML = '';
  // Lines and pause/continue seams, in the order they happened.
  const items = [
    ...(snap.segments || []).map((seg) => ({ at: seg.at || 0, order: 1, seg })),
    ...(snap.markers || []).map((marker) => ({ at: marker.at || 0, order: 0, marker })),
  ].sort((a, b) => (a.at - b.at) || (a.order - b.order));
  items.forEach((item) => (item.seg ? addSegment(item.seg) : addPauseMarker(item.marker)));
  if (!(snap.segments || []).length) {
    ui.transcript.innerHTML = '<p class="empty">Nothing transcribed yet.</p>';
  }
  renderSpeakerChips();
  renderSuggestion(snap.speaker_suggestion);

  ui.advice.innerHTML = '';
  ui.askThread.innerHTML = '';
  askPending = null;
  (snap.cards || []).forEach((card) => {
    if (card.kind === 'answer') {
      addAnswerCard(card);
      if (card.from_user) askThreadTurn(card);
    } else {
      addAdviceCard(card);
    }
  });
  if (!(snap.cards || []).length) {
    ui.advice.innerHTML = '<p class="empty">No suggestions yet.</p>';
  }
  if (!ui.askThread.childElementCount) {
    ui.askThread.innerHTML = '<p class="empty">Nothing asked yet. Answers cite transcript lines as <code>#12</code> — click one to jump to it.</p>';
  }

  ui.attendee.innerHTML = '';
  const speakWas = ui.chkSpeak.checked;
  ui.chkSpeak.checked = false; // replaying history must not read it all aloud
  (snap.attendee_turns || []).forEach(addAttendeeTurn);
  ui.chkSpeak.checked = speakWas;
  if (!(snap.attendee_turns || []).length) {
    ui.attendee.innerHTML = '<p class="empty">Nothing to say yet.</p>';
  }

  if (snap.notes) renderNotes(snap.notes);
  if (snap.user_notes) ui.userNotes.value = snap.user_notes;
  if (snap.summary) ui.summary.textContent = snap.summary;
  if (snap.cost) {
    renderCost(snap.cost);
    startedAtMs = Date.now() - (snap.cost.elapsed_seconds || 0) * 1000;
  }

  if (snap.running) {
    // setRunning resumes streaming only if the worklet is still alive, which
    // tells the two cases apart: a dropped-and-restored socket on this page
    // (keep sending audio) versus a reloaded tab (the mic went with the old
    // page, so the meeting continues on the server but deaf to this one).
    setRunning(true);
    setPaused(!!snap.paused);
    if (fresh) {
      ui.statusText.textContent = snap.resumed ? 'continuing the meeting…' : 'listening…';
    } else if (workletNode) {
      ui.statusText.textContent = snap.resumed ? 'meeting running (continued)' : 'meeting running';
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
}

socket.on('meeting_started', (snap) => {
  // The same painter as a reconnect: a brand-new meeting paints empty panels,
  // a resumed one paints everything said before the interruption.
  paintSnapshot(snap, true);
  if (!(snap.segments || []).length) {
    ui.transcript.innerHTML = '<p class="empty">Listening…</p>';
  }
  log(snap.resumed
    ? `meeting ${snap.meeting_id} continued — ${(snap.segments || []).length} lines restored`
    : `meeting ${snap.meeting_id} started`);
});

socket.on('meeting_stopped', (payload) => {
  setRunning(false);
  stopCapture();
  ui.connDot.classList.remove('listening');
  ui.statusText.textContent = 'finished';
  ui.interim.textContent = '';
  if (payload.notes) renderNotes(payload.notes);
  log(`meeting ${payload.meeting_id} finished and saved`);
  showReviewLink(payload.meeting_id);
  loadHistory();
  loadBriefOptions();
  checkInterrupted();
});

/* The meeting is over but the work on it is not. This is the one moment the user
 * definitely wants the review workspace, so it is offered rather than left to be
 * found in the drawer. */
function showReviewLink(id) {
  if (!id) return;
  ui.btnReview.href = `/review/${id}`;
  ui.btnReview.hidden = false;
}

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
socket.on('answer', (payload) => {
  addAnswerCard(payload);
  if (payload.from_user) askThreadTurn(payload);
});
socket.on('attendee', addAttendeeTurn);
socket.on('notes', (payload) => renderNotes(payload.notes || {}));
socket.on('summary', (payload) => { ui.summary.textContent = payload.summary || ''; });
socket.on('cost', renderCost);
socket.on('usage', () => {});

socket.on('speakers', (payload) => {
  speakerNames = payload.speaker_names || {};
  renderSpeakerChips();
  relabelTranscript();
  applySpeakerFilter();
});

socket.on('paused', (payload) => {
  setPaused(!!payload.paused);
  addPauseMarker(payload);
  if (payload.resumed) {
    log('continuing the interrupted meeting — everything before this point was already saved');
    return;
  }
  ui.statusText.textContent = payload.paused ? 'paused — microphone muted' : 'listening';
  log(payload.paused ? 'paused; no audio is being sent or billed' : 'resumed');
});

socket.on('segment_speaker', (seg) => {
  const line = ui.transcript.querySelector(`.line[data-index="${seg.index}"]`);
  if (!line) return;
  line.dataset.override = seg.speaker_name || '';
  const who = line.querySelector('.who');
  if (who) {
    who.textContent = seg.speaker_label;
    who.classList.toggle('overridden', !!seg.speaker_name);
  }
});

socket.on('speaker_suggestion', renderSuggestion);
socket.on('speaker_suggestion_cleared', () => { ui.suggestion.hidden = true; });

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

// Start with one empty attendee row so the field is obviously fillable, then
// bring back anything typed before a reload, and offer saved backgrounds and
// any meeting that was cut off.
addAttendeeRow();
restoreBriefDraft();
loadBriefOptions();
checkInterrupted();

// Restore the language/model chosen last time.
const savedLanguage = localStorage.getItem('stt_language');
const savedModel = localStorage.getItem('stt_model');
const savedProvider = localStorage.getItem('stt_provider');
if (savedLanguage) ui.inLanguage.value = savedLanguage;
if (savedModel) ui.inModel.value = savedModel;
if (savedProvider) ui.inProvider.value = savedProvider;
