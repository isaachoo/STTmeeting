# Cantonese Meeting Copilot

A private, locally-run web app that listens to a live in-person meeting, shows a
real-time Cantonese transcript, and continuously tells you what matters and what
you might want to say or ask next.

Live only — there is no file upload. You press Start, it listens; you press
Stop, it writes the final notes.

```
┌─────────────────────────────┬─────────────────────────────┐
│  Live transcript            │  Copilot — private to you   │
│  one voice per speaker,     │  key points, questions      │
│  named, streaming           │  worth asking, watch-outs,  │
│                             │  answers to what you ask    │
├─────────────────────────────┼─────────────────────────────┤
│  AI attendee                │  Notes                      │
│  speaks up like a           │  decisions, action items,   │
│  participant: raises        │  open questions, topics,    │
│  questions, answers the     │  plus your own notes        │
│  room's, with sources       │                             │
└─────────────────────────────┴─────────────────────────────┘
```

Cost, exports and history live in the **Session** drawer in the title bar.

## How it works

```
browser mic ──AudioWorklet──▶ 16-bit PCM over a WebSocket ──▶ Flask
                                                             │
                                          ┌──────────────────┴─────────────────┐
                                          ▼                                    ▼
                                 Deepgram streaming WS              copilot orchestrator
                                 (zh-HK, diarised)                  (OpenRouter + web search)
                                          │                                    │
                                          └──────────▶ WebSocket ◀─────────────┘
                                                          │
                                                    the four panels
```

API keys stay on the server; the browser never sees them. The server owns the
transcript, so the copilot works on it directly and everything persists to
SQLite as the meeting happens — a refresh at hour four loses nothing.

## Setup

### Windows

Double-click **`run.bat`**. On the first run it builds the virtual environment,
installs the dependencies, and opens `.env` in Notepad for your API keys; run it
again once those are saved.

Or by hand, in PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env    # then fill in DEEPGRAM_API_KEY and OPENROUTER_API_KEY
.venv\Scripts\python app.py
```

Needs Python 3.10 or newer, installed with "Add python.exe to PATH" ticked.

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then fill in DEEPGRAM_API_KEY and OPENROUTER_API_KEY
.venv/bin/python app.py
```

### API keys

Two ways, and you only need one:

- **In the app** — start it, then click **Session → API keys**, paste them in and
  press Save. Takes effect immediately, no restart. Saved to
  `data/settings.json`, which is not in git.
- **In `.env`** — as before. A key entered in the app overrides the one in
  `.env`, and the form shows you which is in force.

Either way the keys are stored in plain text on your machine, exactly as `.env`
always was. The app never sends a key back to the browser — the form shows only
the last four characters.

### Then

Open <http://127.0.0.1:5000> in Chrome or Edge. Use `127.0.0.1`, not a LAN IP:
browsers only grant microphone access on a secure origin, and `localhost` counts
as one while a bare IP over plain HTTP does not.

The first Start click raises a microphone permission prompt — allow it, and tick
"remember" so it does not ask again mid-meeting.

Fill in the brief before you start — the agenda, what you want out of the
meeting, who is in the room, and the jargon and project names the transcriber
will mangle. Every suggestion is conditioned on it, and it is the single biggest
lever on whether the advice is useful or generic. The names also feed speaker
identification and get boosted in the transcriber.

## Configuration

Everything lives in `.env`; see `.env.example` for the full list. The ones worth
knowing about:

| Variable | Default | What it does |
|---|---|---|
| `STT_PROVIDER` | `deepgram` | `deepgram`, `speechmatics`, or `local` — also selectable per meeting |
| `DEEPGRAM_LANGUAGE` | `zh-HK` | Cantonese Traditional |
| `DEEPGRAM_MODELS` | `nova-3,nova-2` | Tried in order; falls back automatically if a model will not accept the language |
| `OPENROUTER_MODEL` | `deepseek/deepseek-v3.2` | The advisor. Cheap and fast matters more than clever here |
| `OPENROUTER_NOTES_MODEL` | same as above | Set a stronger model if you want better notes |
| `TAVILY_API_KEY` | unset | Without it the copilot answers from model knowledge and says so |
| `THINK_MIN_INTERVAL` | `15` | Seconds between think cycles — the main cost dial |
| `THINK_URGENT_INTERVAL` | `5` | Shorter floor when someone in the room just asked a question |
| `NOTES_INTERVAL` | `90` | Seconds between note-taking passes |
| `ATTENDEE_MODE` | `normal` | `quiet` answers direct questions only, `normal` also raises its own, `active` contributes more freely |
| `ATTENDEE_ENABLED` | `1` | Set to `0` to turn the AI attendee off entirely |
| `DEEPGRAM_KEYTERMS` | `1` | Boost the glossary and attendee names in the transcriber |
| `SPEAKER_GUESS_INTERVAL` | `120` | Seconds between attempts to work out which voice is whom |
| `SAVE_AUDIO` | `0` | Write the raw meeting audio to `data/audio/` for later tuning |

## Three transcribers

Chosen per meeting in the form, or set a default with `STT_PROVIDER`. The `stt/`
package hides the differences, so everything downstream — naming, filtering,
notes, exports — works the same whichever one is running.

| | Deepgram | Speechmatics | Local (sherpa-onnx) |
|---|---|---|---|
| Cost per 5-hour meeting | ~$2.31 | ~$5.20 | **free** |
| Audio leaves your machine | yes | yes | **no** |
| Works offline | no | no | **yes** |
| Speaker diarization | yes | yes | **no** |
| Punctuation | yes | yes | with the extra model |
| Extra setup | none | none | one download |

### Running transcription locally

```bash
pip install -r requirements-local.txt
python scripts/download_models.py --punct
```

Then pick **Local** in the form, or set `STT_PROVIDER=local`.

The model is
`sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en` — Mandarin,
Cantonese and English in one pass, which is how people actually speak in a Hong
Kong office. It runs on the CPU at roughly **eleven times faster than real
time** on two threads, so keeping up is not a concern and no GPU is needed.

Two things it does that the cloud engines do not, both handled automatically:

- It emits **no punctuation**, so a punctuation model runs over each finished
  utterance (that is what `--punct` fetches).
- It writes **simplified characters even for Cantonese speech**, so OpenCC maps
  the result to Hong Kong traditional. That is a deterministic mapping, unlike
  asking an LLM to guess at it.

And one thing it cannot do: **it does not separate speakers.** Every line
arrives unattributed, so click a line's speaker tag to name it. In exchange, the
audio never leaves your laptop and transcription costs nothing.

## What a meeting costs

Measured live in the Session panel. For a five-hour meeting on the defaults:

| | |
|---|---|
| Deepgram streaming, 300 min @ $0.0077/min | ~$2.31 |
| LLM advisor + notes + summaries (DeepSeek V3.2) | ~$0.60–1.00 |
| Web search (Tavily free tier covers this) | $0 |
| **Total** | **~$3–4** |

Speech-to-text dominates and is fixed by the clock. The LLM side is nearly free,
and the cost shown in the UI is the real figure OpenRouter reports per call, not
an estimate from a price table.

Two dials if you want it lower: raise `THINK_MIN_INTERVAL`, and keep the brief
tight — it sits in the prefix of every call. Note that the coaching panel and the
AI attendee come out of one LLM call, not two, so switching the attendee off
saves nothing.

## Design notes

**One call, two outputs.** The private coaching and the AI attendee's turn come
from a single think cycle. That halves the cost against running them separately
and means the two panels can never contradict each other.

**Cost per think cycle stays flat.** The transcript grows all meeting but the
prompt does not: recent speech goes in verbatim and older material is folded
into a rolling summary. Hour four costs the same per call as minute ten.

**The copilot never blocks transcription.** Every LLM call runs on a worker
thread, and at most one of each kind (think, notes, summary, speakers) is ever in
flight. A slow model response throttles the copilot instead of building a backlog
that lands all at once ten minutes later.

**Thinking is rate-limited by new speech, not just by the clock.** Silence costs
nothing. When someone asks a question the floor drops to `THINK_URGENT_INTERVAL`
so the attendee can answer while the room is still waiting — but it never goes to
zero, or a fast back-and-forth would spam the model.

**Repetition is guarded twice.** The prompts are shown what has already been
advised and already said out loud, and a near-duplicate attendee turn is dropped
server-side even if the model produces one. A participant that says the same
sentence three times is the worst failure this panel has.

**A correction on one line beats a name on the whole voice.** Diarisation splits
one person across two voices and merges two people into one, which renaming a
voice cannot fix, so any line's speaker tag can be clicked and set on its own.
That override then survives a voice-level rename and an accepted suggestion —
the user telling us about a specific line outranks anything inferred.

**Speaker names are suggested, never assumed.** Diarisation separates the voices;
the copilot proposes which voice is which person from self-introductions and
names used in the room, and you accept with one click. It will not rename a voice
on its own, will not invent a name that is not on your roster, and will not
contradict a name you set — a wrong name you trust is worse than an unnamed
voice.

**Transcription errors are expected.** Cantonese mixed with English is the hard
case for any STT. The prompts tell the model to read through homophone errors
and mangled English terms using the brief as context, and never to comment on
transcript quality.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

122 tests, no network and no API keys needed.

`test_offline.py` covers the parts that fail silently in a live meeting: JSON
coercion around model output, the rate limiting that decides when the copilot
thinks, the attendee's speak-or-stay-quiet handling and its repeat guard, speaker
naming, the rolling-summary window, the Deepgram message parser and keyterm
fallback, the exports, and the cost arithmetic.

`test_socket_flow.py` boots the real server on a real port and drives it over a
real WebSocket the way the browser does — binary audio frames in, transcript,
coaching, attendee turns and exports out — with Deepgram and OpenRouter faked.

## Limits worth being honest about

- **"Private" means not-a-SaaS-notetaker, not local.** Audio goes to Deepgram and
  the transcript goes to OpenRouter. For genuinely confidential meetings, swap
  in local `faster-whisper` and a local model — the `stt/` package and
  `copilot/llm.py` are the only two places that need to change.
- **One meeting at a time, one user.** This is a local tool, not a server.
- **In-person meetings.** The browser mic captures the room. Capturing a Zoom or
  Teams call needs a system-audio loopback device, which is not wired up here.
- **Diarisation splits voices, it does not recognise people.** Deepgram groups the
  audio by voice; nothing enrolls a voiceprint, so the names come from you or
  from an accepted suggestion. Expect it to merge two similar voices or split one
  person across two tags occasionally, especially with a far-away mic.
- **Speaking aloud is off by default.** The attendee panel can read a turn out
  through your speakers, but the microphone will hear it and transcribe it as
  meeting content. Useful for a quick test, awkward in a real room.
- **A refreshed tab stops sending audio.** The meeting keeps running on the
  server and the page repaints from it, but the microphone belonged to the old
  page. The status bar says so when it happens; stop and restart to resume
  capture. A socket that merely drops and reconnects is different — the page
  keeps its microphone and carries on sending.
