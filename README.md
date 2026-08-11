# Cantonese Meeting Copilot

A private, locally-run web app that listens to a live in-person meeting, shows a
real-time Cantonese transcript, and continuously tells you what matters and what
you might want to say or ask next.

Live only — there is no file upload. You press Start, it listens; you press
Stop, it writes the final notes.

```
┌─────────────────────────────┬─────────────────────────────┐
│  Live transcript            │  Copilot                    │
│  speaker-tagged, streaming  │  key points, questions to   │
│                             │  ask, answers to questions  │
│                             │  raised (web search)        │
├─────────────────────────────┼─────────────────────────────┤
│  Session                    │  Notes                      │
│  running cost, ask box,     │  decisions, action items,   │
│  exports, rolling summary   │  open questions + your own  │
└─────────────────────────────┴─────────────────────────────┘
```

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

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then fill in DEEPGRAM_API_KEY and OPENROUTER_API_KEY
.venv/bin/python app.py
```

Open <http://127.0.0.1:5000>. Use `127.0.0.1`, not a LAN IP: browsers only grant
microphone access on a secure origin, and `localhost` counts as one while a bare
IP over plain HTTP does not.

Write a brief before you start — who is in the room, what you want out of the
meeting, and any jargon or names the transcriber will mangle. Every suggestion
the copilot makes is conditioned on it, and it is the single biggest lever on
whether the advice is useful or generic.

## Configuration

Everything lives in `.env`; see `.env.example` for the full list. The ones worth
knowing about:

| Variable | Default | What it does |
|---|---|---|
| `DEEPGRAM_LANGUAGE` | `zh-HK` | Cantonese Traditional |
| `DEEPGRAM_MODELS` | `nova-3,nova-2` | Tried in order; falls back automatically if a model will not accept the language |
| `OPENROUTER_MODEL` | `deepseek/deepseek-v3.2` | The advisor. Cheap and fast matters more than clever here |
| `OPENROUTER_NOTES_MODEL` | same as above | Set a stronger model if you want better notes |
| `TAVILY_API_KEY` | unset | Without it the copilot answers from model knowledge and says so |
| `ADVICE_MIN_INTERVAL` | `15` | Seconds between advisor calls — the main cost dial |
| `NOTES_INTERVAL` | `90` | Seconds between note-taking passes |
| `SAVE_AUDIO` | `0` | Write the raw meeting audio to `data/audio/` for later tuning |

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

Two dials if you want it lower: raise `ADVICE_MIN_INTERVAL`, and keep the brief
tight — it sits in the prefix of every advisor call.

## Design notes

**Cost per advisor call stays flat.** The transcript grows all meeting but the
prompt does not: recent speech goes in verbatim and older material is folded
into a rolling summary. Hour four costs the same per call as minute ten.

**The copilot never blocks transcription.** Every LLM call runs on a worker
thread, and at most one of each kind (advice, notes, summary) is ever in flight.
A slow model response throttles the copilot instead of building a backlog that
lands all at once ten minutes later.

**Advice is rate-limited by new speech, not just by the clock.** Silence costs
nothing, and the advisor is shown what it already told you so it does not
restate the same point in new words.

**Transcription errors are expected.** Cantonese mixed with English is the hard
case for any STT. The prompts tell the model to read through homophone errors
and mangled English terms using the brief as context, and never to comment on
transcript quality.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

60 tests, no network and no API keys needed.

`test_offline.py` covers the parts that fail silently in a live meeting: JSON
coercion around model output, the rate limiting that decides when the copilot
thinks, the rolling-summary window, the Deepgram message parser, and the cost
arithmetic. `test_socket_flow.py` boots the real server on a real port and drives
it over a real WebSocket the way the browser does — binary audio frames in,
transcript and advice out — with Deepgram and OpenRouter faked.

## Limits worth being honest about

- **"Private" means not-a-SaaS-notetaker, not local.** Audio goes to Deepgram and
  the transcript goes to OpenRouter. For genuinely confidential meetings, swap
  in local `faster-whisper` and a local model — the `stt/` package and
  `copilot/llm.py` are the only two places that need to change.
- **One meeting at a time, one user.** This is a local tool, not a server.
- **In-person meetings.** The browser mic captures the room. Capturing a Zoom or
  Teams call needs a system-audio loopback device, which is not wired up here.
- **A refreshed tab stops sending audio.** The meeting keeps running on the
  server and the page repaints from it, but the microphone belonged to the old
  page. The status bar says so when it happens; stop and restart to resume
  capture. A socket that merely drops and reconnects is different — the page
  keeps its microphone and carries on sending.
