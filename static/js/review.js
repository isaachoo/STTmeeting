/* The review workspace: one saved meeting, four panels.
 *
 * No WebSocket here. A finished meeting does not change under you, so this is
 * plain fetch(), and the page can be reloaded or bookmarked without losing
 * anything. The only thing that takes time is the copilot reading the whole
 * transcript, and that runs as a job on the server which this polls.
 */

const MEETING_ID = window.MEETING_ID;
const el = (id) => document.getElementById(id);

const ui = {
  title: el('r-title'),
  meta: el('r-meta'),
  digestState: el('r-digest-state'),
  btnBuildDigest: el('btn-build-digest'),
  job: el('r-job'),
  jobText: el('r-job-text'),
  jobFill: el('r-job-fill'),
  error: el('r-error'),

  transcript: el('r-transcript'),
  search: el('r-search'),
  searchCount: el('r-search-count'),
  onlyMatches: el('r-only-matches'),
  speakerChips: el('r-speaker-chips'),

  chat: el('r-chat'),
  chatCost: el('r-chat-cost'),
  askForm: el('r-ask-form'),
  ask: el('r-ask'),
  btnClearChat: el('btn-clear-chat'),

  actions: el('r-actions'),
  actionsCount: el('r-actions-count'),
  addAction: el('r-add-action'),
  newWho: el('r-new-who'),
  newWhat: el('r-new-what'),
  newDue: el('r-new-due'),
  btnDraft: el('btn-draft-actions'),
  proposals: el('r-proposals'),
  proposalList: el('r-proposal-list'),
  btnAcceptAll: el('btn-accept-all'),
  btnDismissProposals: el('btn-dismiss-proposals'),

  reportList: el('r-report-list'),
  reportView: el('r-report-view'),
  reportTitle: el('r-report-title'),
  reportBody: el('r-report-body'),
  reportHint: el('r-report-hint'),
  btnReportSave: el('btn-report-save'),
  btnReportCopy: el('btn-report-copy'),
  btnReportDownload: el('btn-report-download'),
  btnReportDelete: el('btn-report-delete'),

  popover: el('r-popover'),
  popLabel: el('r-pop-label'),
  popScope: el('r-pop-scope'),
  popRoster: el('r-pop-roster'),
  popForm: el('r-pop-form'),
  popName: el('r-pop-name'),
  popClear: el('r-pop-clear'),
};

let data = null;              // the whole review payload
let byIndex = new Map();      // transcript line number -> segment
let speakerNames = {};        // "0" -> "Alan"
let speakerFilter = new Set();
let proposals = [];
let currentReport = null;
let popoverSpeaker = null;    // voice being named
let popoverSegment = null;    // single line being named
let reportDirty = false;

// -------------------------------------------------------------- utilities

function showError(message) {
  if (!message) {
    ui.error.hidden = true;
    return;
  }
  ui.error.textContent = message;
  ui.error.hidden = false;
  setTimeout(() => { ui.error.hidden = true; }, 12000);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (err) {
    payload = null;
  }
  if (!response.ok) {
    throw new Error((payload && payload.error) || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function clock(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
  const s = String(total % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

function usd(value) {
  return `$${(value || 0).toFixed(4)}`;
}

function labelFor(segment) {
  if (segment.speaker_name) return segment.speaker_name;
  if (segment.speaker === null || segment.speaker === undefined) return '?';
  return speakerNames[String(segment.speaker)] || `S${segment.speaker + 1}`;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}

function rosterNames() {
  const brief = (data && data.meeting.brief_json) || {};
  const fromBrief = (brief.attendees || []).map((a) => a.name).filter(Boolean);
  const named = Object.values(speakerNames).filter(Boolean);
  return [...new Set([...fromBrief, ...named])];
}

// --------------------------------------------------------------- loading

async function load() {
  try {
    data = await api(`/api/meetings/${MEETING_ID}/review`);
  } catch (err) {
    showError(`could not load the meeting: ${err.message}`);
    return;
  }
  speakerNames = data.speaker_names || {};
  // Lines are addressed by their transcript number, which is not necessarily
  // their position in the array -- citations and links use the number.
  byIndex = new Map(data.segments.map((segment) => [segment.idx, segment]));
  renderMeta();
  renderTranscript();
  renderSpeakerChips();
  renderChat();
  renderActions();
  renderReportList();
  renderDigestState();
  if (data.job) followJob(data.job);
}

function renderMeta() {
  const meeting = data.meeting;
  ui.title.textContent = meeting.title || `Meeting ${MEETING_ID}`;
  const when = meeting.started_at ? new Date(meeting.started_at * 1000).toLocaleString() : '';
  const minutes = ((meeting.audio_seconds || 0) / 60).toFixed(0);
  const parts = [when, `${minutes} min`, `${data.segments.length} lines`];
  if (meeting.stt_model) parts.push(meeting.stt_model);
  if (data.running) parts.push('still running');
  ui.meta.textContent = parts.filter(Boolean).join(' · ');
}

function renderDigestState() {
  const digest = data.digest || {};
  if (digest.built && !digest.stale) {
    const failed = (digest.failed_sections || []).length;
    ui.digestState.textContent = failed
      ? `read (${failed} section${failed > 1 ? 's' : ''} failed)`
      : `whole meeting read (${digest.sections} sections)`;
    ui.digestState.className = failed ? 'small warn-text' : 'small good-text';
    ui.btnBuildDigest.textContent = failed ? 'Read again' : 'Re-read';
  } else if (digest.built && digest.stale) {
    ui.digestState.textContent = 'the transcript changed since it was read';
    ui.digestState.className = 'small warn-text';
    ui.btnBuildDigest.textContent = 'Re-read';
  } else {
    const sections = digest.estimated_sections || 0;
    ui.digestState.textContent = sections
      ? `not read yet — ${sections} section${sections > 1 ? 's' : ''} to read`
      : 'nothing to read';
    ui.digestState.className = 'small muted';
    ui.btnBuildDigest.textContent = 'Read the whole meeting';
  }
}

// ------------------------------------------------------------ transcript

function renderTranscript() {
  ui.transcript.innerHTML = '';
  if (!data.segments.length) {
    ui.transcript.innerHTML = '<p class="empty">This meeting has no transcript.</p>';
    return;
  }
  const fragment = document.createDocumentFragment();
  data.segments.forEach((segment) => {
    const line = document.createElement('div');
    line.className = 'line';
    line.dataset.index = segment.idx;
    line.dataset.speaker = segment.speaker === null ? '' : String(segment.speaker);
    line.id = `line-${segment.idx}`;

    const time = document.createElement('button');
    time.className = 'line-time mono small';
    time.textContent = clock(segment.at);
    time.title = 'Copy a link to this line';
    time.addEventListener('click', () => copyLineLink(segment.idx));

    const who = document.createElement('button');
    who.className = `line-who s${(segment.speaker || 0) % 4}`;
    who.textContent = labelFor(segment);
    who.title = 'Change who said this';
    who.addEventListener('click', (event) => {
      event.stopPropagation();
      openPopover(segment.speaker, who, segment.idx);
    });

    const text = document.createElement('span');
    text.className = 'line-text';
    text.textContent = segment.text;

    line.append(time, who, text);
    fragment.append(line);
  });
  ui.transcript.append(fragment);
  applyFilters();
}

/* Search and speaker filter are both views. Neither changes what the copilot
 * sees -- an answer built from a filtered transcript would be quietly wrong. */
function applyFilters() {
  const needle = ui.search.value.trim().toLowerCase();
  const onlyMatches = ui.onlyMatches.checked && needle.length > 0;
  let matches = 0;

  ui.transcript.querySelectorAll('.line').forEach((line) => {
    const speaker = line.dataset.speaker;
    const visibleSpeaker = !speakerFilter.size
      || (speaker !== '' && speakerFilter.has(Number(speaker)));
    const textNode = line.querySelector('.line-text');
    const segment = byIndex.get(Number(line.dataset.index));
    const raw = segment ? segment.text : textNode.textContent;
    const hit = needle ? raw.toLowerCase().includes(needle) : false;
    if (hit) matches += 1;

    line.classList.toggle('hit', hit);
    line.hidden = !visibleSpeaker || (onlyMatches && !hit);
    textNode.innerHTML = needle && hit ? highlight(raw, needle) : escapeHtml(raw);
  });

  ui.searchCount.textContent = needle
    ? `${matches} line${matches === 1 ? '' : 's'}`
    : '';
}

function highlight(text, needle) {
  const lower = text.toLowerCase();
  let out = '';
  let from = 0;
  for (;;) {
    const at = lower.indexOf(needle, from);
    if (at === -1) break;
    out += escapeHtml(text.slice(from, at));
    out += `<mark>${escapeHtml(text.slice(at, at + needle.length))}</mark>`;
    from = at + needle.length;
  }
  return out + escapeHtml(text.slice(from));
}

function jumpToLine(index) {
  const line = el(`line-${index}`);
  if (!line) return;
  if (line.hidden) {
    // A citation into a line the filter is hiding is a dead end otherwise.
    clearSpeakerFilter();
    ui.onlyMatches.checked = false;
    applyFilters();
  }
  line.scrollIntoView({ behavior: 'smooth', block: 'center' });
  line.classList.remove('flash');
  // Force a reflow so the animation restarts when the same line is clicked twice.
  void line.offsetWidth;
  line.classList.add('flash');
}

async function copyLineLink(index) {
  const url = `${location.origin}${location.pathname}#line-${index}`;
  try {
    await navigator.clipboard.writeText(url);
    ui.searchCount.textContent = 'link copied';
    setTimeout(() => applyFilters(), 1500);
  } catch (err) {
    location.hash = `line-${index}`;
  }
}

ui.search.addEventListener('input', applyFilters);
ui.onlyMatches.addEventListener('change', applyFilters);

// -------------------------------------------------------------- speakers

function renderSpeakerChips() {
  ui.speakerChips.innerHTML = '';
  const seen = [...new Set(
    data.segments.map((s) => s.speaker).filter((s) => s !== null && s !== undefined)
  )].sort((a, b) => a - b);

  if (!seen.length) {
    const hint = document.createElement('span');
    hint.className = 'muted small';
    hint.textContent = 'no speaker separation — click a name to fix one line';
    ui.speakerChips.append(hint);
    return;
  }

  seen.forEach((speaker) => {
    const named = speakerNames[String(speaker)];
    const active = speakerFilter.has(speaker);
    const chip = document.createElement('span');
    chip.className = `chip-speaker s${speaker % 4}${named ? '' : ' unnamed'}`
      + (active ? ' active' : '')
      + (speakerFilter.size && !active ? ' dimmed' : '');

    const label = document.createElement('button');
    label.className = 'chip-label';
    label.textContent = named || `S${speaker + 1}`;
    label.title = 'Click to show only this speaker';
    label.addEventListener('click', () => {
      if (speakerFilter.has(speaker)) speakerFilter.delete(speaker);
      else speakerFilter.add(speaker);
      applyFilters();
      renderSpeakerChips();
    });

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

  if (speakerFilter.size) {
    const clear = document.createElement('button');
    clear.className = 'ghost small';
    clear.textContent = 'show everyone';
    clear.addEventListener('click', clearSpeakerFilter);
    ui.speakerChips.append(clear);
  }
}

function clearSpeakerFilter() {
  speakerFilter.clear();
  applyFilters();
  renderSpeakerChips();
}

function openPopover(speaker, anchor, segmentIndex = null) {
  if (data.running) {
    showError('This meeting is still running — rename speakers in the live console.');
    return;
  }
  popoverSpeaker = speaker;
  popoverSegment = segmentIndex;

  const current = segmentIndex === null
    ? (speakerNames[String(speaker)] || '')
    : (byIndex.get(segmentIndex) || {}).speaker_name || '';
  ui.popLabel.textContent = segmentIndex === null
    ? (speakerNames[String(speaker)] || `S${speaker + 1}`)
    : `this line`;
  ui.popScope.textContent = segmentIndex === null
    ? 'Renames this voice everywhere in the meeting.'
    : 'Changes this one line only, overriding the voice it was assigned to.';
  ui.popName.value = current;

  ui.popRoster.innerHTML = '';
  rosterNames().forEach((name) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ghost small';
    button.textContent = name;
    button.addEventListener('click', () => applyName(name));
    ui.popRoster.append(button);
  });

  const box = anchor.getBoundingClientRect();
  ui.popover.hidden = false;
  ui.popover.style.top = `${Math.min(box.bottom + 6, window.innerHeight - 200)}px`;
  ui.popover.style.left = `${Math.min(box.left, window.innerWidth - 280)}px`;
  ui.popName.focus();
}

function closePopover() {
  ui.popover.hidden = true;
  popoverSpeaker = null;
  popoverSegment = null;
}

async function applyName(name) {
  const cleaned = (name || '').trim();
  try {
    if (popoverSegment !== null) {
      const index = popoverSegment;
      await api(`/api/meetings/${MEETING_ID}/segments/${index}/speaker`, {
        method: 'POST',
        body: JSON.stringify({ name: cleaned }),
      });
      const segment = byIndex.get(index);
      if (segment) segment.speaker_name = cleaned;
    } else {
      const result = await api(`/api/meetings/${MEETING_ID}/speakers`, {
        method: 'POST',
        body: JSON.stringify({ speaker: popoverSpeaker, name: cleaned }),
      });
      speakerNames = result.speaker_names || {};
    }
  } catch (err) {
    showError(err.message);
    return;
  }
  closePopover();
  relabel();
  renderSpeakerChips();
}

function relabel() {
  ui.transcript.querySelectorAll('.line').forEach((line) => {
    const segment = byIndex.get(Number(line.dataset.index));
    if (segment) line.querySelector('.line-who').textContent = labelFor(segment);
  });
}

ui.popForm.addEventListener('submit', (event) => {
  event.preventDefault();
  applyName(ui.popName.value);
});
ui.popClear.addEventListener('click', () => applyName(''));
document.addEventListener('click', (event) => {
  if (!ui.popover.hidden && !ui.popover.contains(event.target)) closePopover();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closePopover();
});

// ------------------------------------------------------- ask the meeting

function renderChat() {
  const turns = data.chat || [];
  ui.chat.innerHTML = '';
  if (!turns.length) {
    ui.chat.innerHTML = `<p class="empty">
      Ask anything about what was said. Answers cite the transcript lines they
      came from; click a citation to jump to it.</p>`;
    ui.chatCost.textContent = '';
    return;
  }
  turns.forEach((turn) => ui.chat.append(chatTurn(turn)));
  const total = turns.reduce((sum, turn) => sum + (turn.cost_usd || 0), 0);
  ui.chatCost.textContent = `${turns.length} asked · ${usd(total)}`;
  ui.chat.scrollTop = ui.chat.scrollHeight;
}

function chatTurn(turn) {
  const block = document.createElement('div');
  block.className = 'chat-turn';

  const question = document.createElement('div');
  question.className = 'chat-q';
  question.textContent = turn.question;

  const answer = document.createElement('div');
  answer.className = 'chat-a';
  answer.innerHTML = withCitations(turn.answer || '');
  answer.querySelectorAll('button.cite').forEach((button) => {
    button.addEventListener('click', () => jumpToLine(Number(button.dataset.line)));
  });

  block.append(question, answer);
  if (turn.cost_usd) {
    const cost = document.createElement('div');
    cost.className = 'muted small';
    cost.textContent = usd(turn.cost_usd);
    block.append(cost);
  }
  return block;
}

/* [#42] becomes a button that scrolls to line 42. Escaped first: an answer is
 * model output, and it goes into innerHTML. */
function withCitations(text) {
  return escapeHtml(text).replace(
    /\[#(\d+)\]/g,
    (whole, index) => `<button class="cite" data-line="${index}" title="jump to line ${index}">#${index}</button>`
  );
}

ui.askForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = ui.ask.value.trim();
  if (!question) return;
  ui.ask.value = '';

  const pending = document.createElement('div');
  pending.className = 'chat-turn pending';
  pending.innerHTML = `<div class="chat-q">${escapeHtml(question)}</div>
    <div class="chat-a muted">reading the transcript…</div>`;
  if (ui.chat.querySelector('.empty')) ui.chat.innerHTML = '';
  ui.chat.append(pending);
  ui.chat.scrollTop = ui.chat.scrollHeight;

  try {
    const result = await api(`/api/meetings/${MEETING_ID}/ask`, {
      method: 'POST',
      body: JSON.stringify({ question }),
    });
    data.chat.push(result);
    renderChat();
    if (!result.used_digest) {
      ui.reportHint.textContent = '';
      showHint();
    }
  } catch (err) {
    pending.querySelector('.chat-a').textContent = `could not answer: ${err.message}`;
    pending.querySelector('.chat-a').classList.add('warn-text');
    pending.classList.remove('pending');
  }
});

function showHint() {
  if (data.digest && data.digest.built) return;
  ui.digestState.textContent = 'answered from the live notes only — read the meeting for better answers';
  ui.digestState.className = 'small warn-text';
}

ui.btnClearChat.addEventListener('click', async () => {
  if (!(data.chat || []).length) return;
  if (!confirm('Delete every question and answer for this meeting?')) return;
  try {
    await api(`/api/meetings/${MEETING_ID}/chat`, { method: 'DELETE' });
    data.chat = [];
    renderChat();
  } catch (err) {
    showError(err.message);
  }
});

// ---------------------------------------------------------- action items

function renderActions() {
  const actions = data.actions || [];
  ui.actions.innerHTML = '';
  const open = actions.filter((a) => a.status !== 'done').length;
  ui.actionsCount.textContent = actions.length
    ? `${open} open of ${actions.length}`
    : '';

  if (!actions.length) {
    const row = document.createElement('tr');
    row.innerHTML = `<td colspan="5" class="empty">
      Nothing yet. Add items yourself, or let the copilot draft them from the
      meeting and accept the ones that are right.</td>`;
    ui.actions.append(row);
    return;
  }

  actions.forEach((action) => ui.actions.append(actionRow(action)));
}

function actionRow(action) {
  const row = document.createElement('tr');
  row.className = action.status === 'done' ? 'done' : '';

  const tick = document.createElement('td');
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = action.status === 'done';
  box.title = 'Mark done';
  box.addEventListener('change', () =>
    patchAction(action, { status: box.checked ? 'done' : 'open' }));
  tick.append(box);

  const cell = (field, placeholder, className) => {
    const td = document.createElement('td');
    const input = document.createElement('input');
    input.type = 'text';
    input.value = action[field] || '';
    input.placeholder = placeholder;
    if (className) input.className = className;
    input.addEventListener('change', () => patchAction(action, { [field]: input.value }));
    td.append(input);
    return td;
  };

  const remove = document.createElement('td');
  const button = document.createElement('button');
  button.className = 'ghost small';
  button.textContent = '×';
  button.title = 'Delete';
  button.addEventListener('click', async () => {
    try {
      await api(`/api/meetings/${MEETING_ID}/actions/${action.id}`, { method: 'DELETE' });
      data.actions = data.actions.filter((a) => a.id !== action.id);
      renderActions();
    } catch (err) {
      showError(err.message);
    }
  });
  remove.append(button);

  row.append(tick, cell('who', 'unassigned'), cell('what', ''), cell('due', '—'), remove);
  if (action.source === 'copilot') row.title = 'drafted by the copilot';
  return row;
}

async function patchAction(action, fields) {
  try {
    const result = await api(`/api/meetings/${MEETING_ID}/actions/${action.id}`, {
      method: 'PATCH',
      body: JSON.stringify(fields),
    });
    Object.assign(action, result.action);
    renderActions();
  } catch (err) {
    showError(err.message);
  }
}

ui.addAction.addEventListener('submit', async (event) => {
  event.preventDefault();
  const what = ui.newWhat.value.trim();
  if (!what) return;
  try {
    const result = await api(`/api/meetings/${MEETING_ID}/actions`, {
      method: 'POST',
      body: JSON.stringify({
        who: ui.newWho.value.trim(),
        what,
        due: ui.newDue.value.trim(),
      }),
    });
    data.actions = result.actions;
    ui.newWho.value = '';
    ui.newWhat.value = '';
    ui.newDue.value = '';
    renderActions();
  } catch (err) {
    showError(err.message);
  }
});

ui.btnDraft.addEventListener('click', () => {
  startJob(`/api/meetings/${MEETING_ID}/actions/draft`, {}, (result) => {
    proposals = result.proposals || [];
    renderProposals();
    if (result.digest_built) refreshDigestState();
    if (!proposals.length) showError('The copilot found nothing that was actually agreed.');
  });
});

function renderProposals() {
  ui.proposalList.innerHTML = '';
  ui.proposals.hidden = !proposals.length;
  if (!proposals.length) return;

  proposals.forEach((proposal, i) => {
    const row = document.createElement('div');
    row.className = 'proposal';

    const text = document.createElement('div');
    const who = document.createElement('strong');
    who.textContent = proposal.who || 'unassigned';
    text.append(who, document.createTextNode(`: ${proposal.what}`));
    if (proposal.due) {
      const due = document.createElement('span');
      due.className = 'muted small';
      due.textContent = ` — due ${proposal.due}`;
      text.append(due);
    }
    if (proposal.confidence === 'low') {
      const flag = document.createElement('span');
      flag.className = 'warn-text small';
      flag.textContent = ' — sounded like an intention, check it';
      text.append(flag);
    }
    if ((proposal.lines || []).length) {
      const refs = document.createElement('span');
      refs.className = 'refs';
      proposal.lines.slice(0, 4).forEach((line) => {
        const button = document.createElement('button');
        button.className = 'cite';
        button.textContent = `#${line}`;
        button.addEventListener('click', () => jumpToLine(line));
        refs.append(button);
      });
      text.append(refs);
    }

    const accept = document.createElement('button');
    accept.className = 'ghost small';
    accept.textContent = 'Accept';
    accept.addEventListener('click', () => acceptProposals([proposal], [i]));

    row.append(text, accept);
    ui.proposalList.append(row);
  });
}

async function acceptProposals(items, indices) {
  try {
    const result = await api(`/api/meetings/${MEETING_ID}/actions`, {
      method: 'POST',
      body: JSON.stringify({
        items: items.map((p) => ({ ...p, source: 'copilot' })),
      }),
    });
    data.actions = result.actions;
    proposals = proposals.filter((_, i) => !indices.includes(i));
    renderActions();
    renderProposals();
  } catch (err) {
    showError(err.message);
  }
}

ui.btnAcceptAll.addEventListener('click', () =>
  acceptProposals(proposals, proposals.map((_, i) => i)));
ui.btnDismissProposals.addEventListener('click', () => {
  proposals = [];
  renderProposals();
});

// -------------------------------------------------------------- reports

document.querySelectorAll('.report-kinds .kind').forEach((button) => {
  button.addEventListener('click', () => {
    startJob(
      `/api/meetings/${MEETING_ID}/reports`,
      { kind: button.dataset.kind },
      (report) => {
        data.reports.unshift(report);
        renderReportList();
        openReport(report);
        if (report.digest_built) refreshDigestState();
      }
    );
  });
});

function renderReportList() {
  const reports = data.reports || [];
  ui.reportList.innerHTML = '';
  if (!reports.length) {
    ui.reportList.innerHTML = `<p class="empty small">
      Pick one above. Each is written from the whole meeting, and you can edit it
      before you send it.</p>`;
    return;
  }
  reports.forEach((report) => {
    const row = document.createElement('button');
    row.className = 'report-row'
      + (currentReport && currentReport.id === report.id ? ' active' : '');
    const when = report.created_at
      ? new Date(report.created_at * 1000).toLocaleString()
      : '';
    row.innerHTML = `<span>${escapeHtml(report.title || report.kind)}</span>
      <span class="muted small">${escapeHtml(when)}</span>`;
    row.addEventListener('click', () => openReport(report));
    ui.reportList.append(row);
  });
}

async function openReport(report) {
  if (reportDirty && !confirm('Discard the edits you have not saved?')) return;
  let full = report;
  if (report.body === undefined) {
    try {
      full = (await api(`/api/reports/${report.id}`)).report;
    } catch (err) {
      showError(err.message);
      return;
    }
  }
  currentReport = full;
  reportDirty = false;
  ui.reportView.hidden = false;
  ui.reportTitle.textContent = full.title || full.kind;
  ui.reportBody.value = full.body || '';
  ui.btnReportDownload.href = `/api/reports/${full.id}/download.md`;
  ui.reportHint.textContent = full.model ? `written by ${full.model}` : '';
  renderReportList();
}

ui.reportBody.addEventListener('input', () => {
  reportDirty = true;
  ui.reportHint.textContent = 'unsaved edits';
});

ui.btnReportSave.addEventListener('click', async () => {
  if (!currentReport) return;
  try {
    const result = await api(`/api/reports/${currentReport.id}`, {
      method: 'PUT',
      body: JSON.stringify({ body: ui.reportBody.value }),
    });
    currentReport = result.report;
    reportDirty = false;
    ui.reportHint.textContent = 'saved';
  } catch (err) {
    showError(err.message);
  }
});

ui.btnReportCopy.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(ui.reportBody.value);
    ui.reportHint.textContent = 'copied';
  } catch (err) {
    ui.reportBody.select();
    ui.reportHint.textContent = 'select and copy';
  }
});

ui.btnReportDelete.addEventListener('click', async () => {
  if (!currentReport) return;
  if (!confirm('Delete this report?')) return;
  try {
    await api(`/api/reports/${currentReport.id}`, { method: 'DELETE' });
    data.reports = data.reports.filter((r) => r.id !== currentReport.id);
    currentReport = null;
    reportDirty = false;
    ui.reportView.hidden = true;
    renderReportList();
  } catch (err) {
    showError(err.message);
  }
});

// ------------------------------------------------------------------ jobs

ui.btnBuildDigest.addEventListener('click', () => {
  const sections = (data.digest || {}).estimated_sections || 0;
  if (sections > 4 && !confirm(
    `Reading this meeting takes ${sections} passes over the transcript, one LLM `
    + `call each. It happens once and everything afterwards reuses it. Go ahead?`
  )) return;
  startJob(`/api/meetings/${MEETING_ID}/digest`, {}, () => refreshDigestState());
});

async function startJob(path, body, onDone) {
  showError('');
  try {
    const started = await api(path, { method: 'POST', body: JSON.stringify(body) });
    followJob(started.job, onDone);
  } catch (err) {
    showError(err.message);
  }
}

function followJob(job, onDone) {
  setJobBanner(job);
  const poll = async () => {
    let current;
    try {
      current = (await api(`/api/jobs/${job.id}`)).job;
    } catch (err) {
      ui.job.hidden = true;
      showError(`lost track of the job: ${err.message}`);
      return;
    }
    setJobBanner(current);
    if (current.status === 'running') {
      setTimeout(poll, 1200);
      return;
    }
    ui.job.hidden = true;
    if (current.status === 'error') {
      showError(`${current.label} failed: ${current.error}`);
      return;
    }
    if (onDone) onDone(current.result);
  };
  setTimeout(poll, 800);
}

function setJobBanner(job) {
  ui.job.hidden = false;
  const stage = job.stage ? ` — ${job.stage}` : '';
  const count = job.total ? ` ${job.done}/${job.total}` : '';
  ui.jobText.textContent = `${job.label}${stage}${count} · ${job.elapsed}s`;
  const percent = job.total ? Math.round((job.done / job.total) * 100) : 0;
  ui.jobFill.style.width = `${percent}%`;
}

async function refreshDigestState() {
  try {
    const fresh = await api(`/api/meetings/${MEETING_ID}/review`);
    data.digest = fresh.digest;
    renderDigestState();
  } catch (err) {
    /* the banner is cosmetic; a failed refresh is not worth an error */
  }
}

// ------------------------------------------------------------------ start

load().then(() => {
  // #line-42 in the URL: land on that line, from a copied link.
  if (location.hash.startsWith('#line-')) {
    jumpToLine(Number(location.hash.slice(6)));
  }
});

window.addEventListener('beforeunload', (event) => {
  if (!reportDirty) return;
  event.preventDefault();
  event.returnValue = '';
});
