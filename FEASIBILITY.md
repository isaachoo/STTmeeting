# Feasibility: eight requested features

Assessed against the current codebase. Verdicts are honest about which parts are
straightforward, which carry real risk, and which need a decision from you first.

**Summary**

| # | Feature | Verdict | Effort | Risk |
|---|---|---|---|---|
| 1 | Pause while recording | Easy | 2–3 h | none |
| 2 | Choose STT model / language | Easy | 3–4 h | one unknown |
| 3 | Speechmatics as an STT option | Medium | 1 day | low — Cantonese confirmed |
| 4 | My role → tailored suggestions | Easy | 2 h | none |
| 5 | Post-meeting review workspace | Large | 3–4 days | scope |
| 6 | Change the speaker on one line | Medium | 4–6 h | none |
| 7 | Filter by speaker | Easy | 2 h | none |
| 8 | Teams / Zoom meetings | Medium–hard | 1–2 days | platform limits |

Everything here is possible. Item 5 is the one that changes the app's shape; item 8
is the one with constraints outside our control.

---

## 1. Pause while recording — Easy

**Verdict: straightforward, no complications.**

A Pause button that stops sending audio, keeps the meeting open, and resumes cleanly.

- The browser simply stops passing chunks to the socket. The AudioWorklet keeps
  running so resume is instant, with no second microphone permission prompt.
- The Deepgram connection already sends `KeepAlive` frames whenever no audio is
  queued, so a pause of any length holds the socket open without reconnecting.
- **Paused time is free.** Deepgram bills for audio streamed, not for connection
  time, so a 20-minute break costs nothing.
- The copilot must also pause — otherwise it keeps thinking about a conversation
  that stopped. One flag check in `on_utterance`.
- The transcript should show a visible `⏸ paused 14:32 – 14:51` marker, and the
  cost readout should stop advancing.

**Effort**: 2–3 hours. **Risk**: none.

---

## 2. Choose the STT model and language — Easy, with one unknown

**Verdict: easy to build. One fact needs checking before I can offer "multilingual"
honestly.**

The plumbing already exists — `DEEPGRAM_MODELS` and `DEEPGRAM_LANGUAGE` are config
values. The work is surfacing them per meeting instead of only in `.env`:

- Language dropdown: Cantonese `zh-HK`, Mandarin `zh-CN` / `zh-TW`, English `en`,
  and Multilingual if it applies.
- Model dropdown: `nova-3`, `nova-2`, with the automatic fallback still in place.
- Remembered as your default, overridable per meeting.

**The unknown**: Deepgram's Nova-3 Multilingual advertises code-switching across a
set of languages, and I have **not confirmed Cantonese is in that set**. If it is,
multilingual mode is potentially a real improvement for your Cantonese-plus-English
speech. If it is not, selecting it would silently give you worse Cantonese.

I will verify before wiring it up, and if unconfirmed the dropdown ships with
single-language options only rather than a trap.

**Effort**: 3–4 hours. **Risk**: low.

---

## 3. Speechmatics as a second STT provider — Medium

**Verdict: viable and worth doing. Cantonese and real-time diarization are both
confirmed on their side.**

The `stt/` package was built for exactly this: `STTEngine` is an abstract interface
and `DeepgramLiveSTT` is one implementation. Adding a provider means writing a
sibling class, not restructuring anything.

What it involves:

- A new `stt/speechmatics_live.py` speaking their WebSocket protocol
  (`StartRecognition` / `AddAudio` / `AddPartialTranscript` / `AddTranscript` /
  `EndOfStream`), which differs from Deepgram's but carries the same information.
- Their auth model uses short-lived JWTs rather than a static key header.
- Mapping their speaker labels (`S1`, `S2`…) onto our diarisation indices so the
  naming, chips and filters keep working unchanged.
- A provider dropdown next to the model dropdown, plus `SPEECHMATICS_API_KEY`.
- The same fallback discipline: if a provider fails to connect, say so clearly
  rather than sitting silent.

**The real value**: you get to A/B the same meeting style against two engines and
keep whichever transcribes your Cantonese better. That answers the accuracy question
with evidence instead of opinion.

**Cost note**: Speechmatics real-time is generally priced above Deepgram. I will put
the actual per-hour figure in the UI cost readout so you can compare like for like.

**Effort**: 1 day. **Risk**: low.

---

## 4. My role in the meeting — Easy, high value

**Verdict: the cheapest item on this list, and probably the biggest quality jump.**

Partly there already — the attendee roster has a role field and an "is me" flag. What
is missing is using it deliberately.

- A dedicated field: *"My role in this meeting"* — free text, e.g. "IT manager
  proposing the budget", "vendor-side project lead", "chairing, need to stay neutral".
- It goes into the think prompt so both the coaching and the AI attendee reason from
  your position: a finance lead gets cost exposure and commitments flagged; a project
  lead gets scope and dependency risks; a chair gets "nobody has owned this decision".
- Empty means general-purpose advice, exactly as now.

Same cost, same latency — it only changes what the model is told.

**Effort**: 2 hours. **Risk**: none.

---

## 5. Post-meeting review workspace — Large, and the most valuable

**Verdict: very possible, and the feature that turns this from a recorder into
something you work with. It is also the largest item, so scope matters.**

A `/meeting/<id>` page reusing the live layout, but built for working rather than
listening:

- **Transcript** — full, searchable, speaker-filtered, click a line to jump.
- **Ask the meeting** — a chat panel answering from the transcript, with citations
  back to the exact lines so you can verify rather than trust.
- **Action items** — editable: add, edit, assign, set due dates, tick off. Persisted
  properly rather than regenerated each time.
- **Generate** — minutes, executive summary, decision log, follow-up email, a
  status update for your boss. Pick a format, edit the result, export.

**The one technical problem worth naming**: a five-hour transcript will not fit in a
prompt. Sending it whole would be slow and expensive, and quality would drop. The
answer is retrieval — index the transcript in chunks, pull only the relevant parts
for each question, and always show which lines an answer came from. That is the bulk
of the engineering and it is well-understood work.

**Decision I need from you**: which reports actually matter. Building four good ones
beats building ten mediocre ones. My default would be minutes, action items, an
executive summary, and a follow-up email.

**Effort**: 3–4 days. **Risk**: scope only.

---

## 6. Change the speaker on a single line — Medium

**Verdict: needed, precisely because diarisation makes mistakes.**

Today naming works per *voice*: label S2 as Alan and every S2 line becomes Alan. That
does not help when the engine splits one person across two voices, or merges two
people into one — which happens with a far-field mic.

So this needs a per-line override stored alongside the segment, layered over the
voice-level name:

- Click a line's speaker tag → pick from the roster or type a name.
- Bulk fix: *"reassign this line and the next N"*, since diarisation errors run in
  streaks rather than appearing alone.
- Works live and in review, feeds exports, and feeds the prompts so the copilot
  attributes correctly from then on.
- Overrides must survive an accepted speaker-name suggestion — a manual correction
  outranks a guess.

**Effort**: 4–6 hours. **Risk**: none.

---

## 7. Filter by speaker — Easy

**Verdict: trivial, and pairs naturally with 6.**

Speaker chips become toggles; clicking filters the transcript to that person. Also
useful as *"show only what I said"*.

**One rule**: filtering is a *view*, never a change to the data. The copilot must
keep seeing the whole conversation regardless of what is on screen, or the advice
quietly degrades whenever you filter. Same in review mode: filter the display, ask
questions against everything.

**Effort**: 2 hours. **Risk**: none — as long as that rule holds.

---

## 8. Teams and Zoom — Medium to hard, with real constraints

**Verdict: possible, and there are three routes with genuinely different trade-offs.
This is the one where the platforms constrain us rather than the code.**

The problem: the browser microphone hears the room, not the people on the call.

### Route A — Chrome screen-share audio (free, no install)

Chrome's `getDisplayMedia` can capture system audio on Windows when you share a
screen, and tab audio when you share a tab. Pick the Teams window, tick "share
system audio", done.

- **Pros**: no install, works with the Teams and Zoom desktop apps, ~2 hours of work.
- **Cons**: an extra share prompt each meeting; captures *their* audio, so it needs
  mixing with your microphone to catch your side too; Chrome/Edge only.

### Route B — Windows loopback capture (best experience, more work)

Capture the system audio device directly in Python (WASAPI loopback), mix with the
microphone, and stream the result. No virtual cable needed.

- **Pros**: no prompts, works with every app, catches both sides properly, and you
  can pick devices in the UI.
- **Cons**: audio capture moves from the browser into Python for this mode — a real
  addition, though the STT layer already takes raw PCM so it plugs straight in.
  Windows-specific. About a day.

### Route C — A meeting bot that joins the call

A service joins as a participant and returns per-speaker audio streams.

- **Pros**: perfect speaker separation with real names from the platform, no local
  audio at all.
- **Cons**: costs meaningfully more per hour, sends your meeting to another
  third party, and everyone sees a bot join. Against the spirit of a private local
  tool.

**My recommendation**: Route A first because it is cheap and proves the value, then
Route B if you end up doing online meetings regularly. Skip C.

**Bonus**: online meetings usually give *better* accuracy than your room, because each
person speaks into their own headset instead of one mic across a table.

---

## Suggested order

Grouped so related work lands together, cheapest value first.

**Batch 1 — quick wins (about 1 day)**
Items 1, 4, 7, and 2. Small, independent, immediately useful. Item 4 alone should
noticeably sharpen the advice.

**Batch 2 — transcript accuracy (about 1 day)**
Item 6, plus item 3 (Speechmatics) so you can compare engines on your own audio.

**Batch 3 — the review workspace (3–4 days)**
Item 5, reusing the filtering and speaker editing from batches 1 and 2.

**Batch 4 — online meetings (1–2 days)**
Item 8, Route A first.

---

## What I need from you

1. **Reports for the review workspace** — minutes, exec summary, action items and a
   follow-up email, or something else? This shapes item 5.
2. **Teams or Zoom, and how often?** If it is occasional, Route A is enough. If it is
   most of your meetings, Route B is worth the extra day.
3. **Do you want Speechmatics as a permanent alternative, or as a one-off comparison?**
   Permanent means a provider dropdown; comparison could be a simpler switch.
4. **Batch order** — happy with the sequence above, or is the review workspace the
   urgent one?
