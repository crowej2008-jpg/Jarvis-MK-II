# JARVIS

A local-first voice assistant that can see the screen, act on it, and remember
what it learned. Ollama runs the brain, faster-whisper does the ears, Tesseract
reads the screen, and a local voice does the mouth. No cloud services.

## What it does

- **Voice and text** — wake word, speech recognition, and speech out.
- **Sees the screen** — live window title, OCR text, clickable elements, and
  a visual memory of every screen it has looked at.
- **Acts on the screen** — click, move, hover, scroll, drag, type, press keys.
  Every action is gated by an autonomy policy and verified against a second
  observation.
- **Remembers** — conversation, facts, notes, learned UI coordinates, successful
  procedures, and per-app habits.
  - **72 tools** across screen, files, web, system, memory, home automation, and
    MQTT. All 72 are registered, and the 10 most relevant to what you just said
    are sent each turn, with `list_more_tools` always offered as a way back to
    the rest.

## Requirements

- Windows 10 or 11
- Python 3.10 or newer (3.12 recommended)
- [Ollama](https://ollama.com/download), running
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki), installed and
  on `PATH`

## Install

```powershell
pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct
ollama pull moondream
ollama pull qwen2.5vl:3b
python -m jarvis --doctor
```

`--doctor` checks the Python version, the brain, OCR, the screen, seeing,
autonomy, voice, and every tool. It exits non-zero if something real is broken.
Unconfigured optional integrations report `SKIP`, not `FAIL`.

## Run

```powershell
python -m jarvis              # voice, with wake word
python -m jarvis --text       # type instead
python -m jarvis --no-wake    # voice, push to talk
python -m jarvis "what time is it"   # one-shot, then exit
python -m jarvis --tools      # list every tool
python -m jarvis --show-config
python -m jarvis --list-audio # input devices, to pick a microphone
python -m jarvis --device 7   # use a specific microphone
```

### Which microphone it listens to

With no `--device`, JARVIS picks a real microphone rather than blindly taking
the system default. On Windows the first input endpoint is usually
"Microsoft Sound Mapper - Input", and several others are loopback taps
("Stereo Mix", "PC Speaker ... output") that capture system audio or nothing at
all. Listening to one of those gives you a running assistant that never hears a
word, so the default prefers a device that looks like an actual microphone and
`--doctor` warns you when the chosen one looks virtual. Pass `--device <index>`
from `--list-audio` to override.

## Models

| Role | Model | Notes |
| --- | --- | --- |
| Brain | `qwen2.5:3b-instruct` | Default. Direct tool calls, ~9s warm. |
| Pointing | `moondream` | Finds a described object and returns coordinates. |
| Describing | `qwen2.5vl:3b` | Reads a real window. ~3 GB, see the note on cost below. |

`qwen3:4b` is installed as a fallback but is not the default: it narrates tool
calls and took 220–250s per turn on this machine.

### Measured speed

The brain prefills at roughly **40 tokens/s** on 10 CPU cores. The system prompt
plus the 38-tool working set is **3,869 prompt tokens**, and evaluating it costs
about 92 seconds.

| Measurement | Prompt | Time |
| --- | --- | --- |
| Cold call, 38 tools | 3,869 tokens | **95–110s** (≈92–103s prefill) |
| Cold call, no tools | 865 tokens | **13s** (≈12s of that is loading the model) |
| Repeat call, identical prefix | 3,869 tokens | **0.2–3s** (cache hit, 0.1s prefill) |

So a cold turn is expensive and a warm one is not, and the whole game is
whether Ollama is allowed to reuse the cache.

**Generation is the other half, and it is not negligible.** Measured at
**6.5–8.9 tokens/s**, falling only about 25% as the context grows from 862 to
4,780 tokens — so context length is not what makes it slow, the CPU is. A short
reply costs 2–3s, but the model writing 1,560 characters takes ~56s, and turn
time tracks reply length almost exactly:

| Reply | Turn time |
| --- | --- |
| 31 chars | 20.7s |
| 67 chars | 25.0s |
| 367 chars | 47.1s |
| 1,560 chars | 70.7s |

The 1,560-character case is the model enumerating all 38 tools, which is a real
cost of a real answer rather than a fault. The lever is reply length, not
context.

**The clock used to stop it, on every turn.** The system prompt carried the
current time, and the system prompt is rendered *before* the tool schemas, so
those few characters sat at the very front of the token stream. Ollama can only
reuse a cached prompt when the new one starts with the same tokens, so a clock
there invalidated the entire prefix — persona, routing notes, and all 38 tool
schemas, 3,869 tokens — as soon as the minute changed. Three consecutive turns
measured **96.8s, 105.2s and 113.2s**: every one of them a full re-prefill.
Repeating the identical request straight afterwards took 0.2s, which is what
finally showed the prompt was not the problem.

The fix is `Brain.clock_note()`: the time is now sent on the user's message, so
the varying text lands *after* the stable prefix and the cache survives. The
same three turns now measure:

| Turn | Before | After |
| --- | --- | --- |
| 1, cold | 96.8s | 112.8s |
| 2 | 105.2s | **7.7s** |
| 3 | 113.2s | **14.9s** |

`test_system_prompt_carries_no_clock` guards this, because putting a timestamp
back into the system prompt is exactly the kind of change that looks harmless.

**The second cache-breaker was the conversation history, and it was the expensive
one.** `recent()` returns the *newest* `keep_last_messages` rows, so the replayed
window slides forward on every turn and the message sitting immediately after the
system prompt is a different one each time. Ollama can only reuse a prompt that
*starts* with the same tokens, so a sliding window makes the entire history block
unreusable while the system prompt and tool schemas in front of it stay cached.
The prefix that survives is ~3,900 tokens; the ~1,600 that move are re-evaluated
from scratch, every turn.

It looked harmless, too — more context for the model, no obvious cost. Measured
mean warm prefill on real turns:

| `keep_last_messages` | Prompt tokens | Mean warm prefill |
| --- | --- | --- |
| 40 (was) | 5,552 | **62.4s** |
| 8 (now) | 3,970 | **3.7s** |
| 0 | 3,916 | **0.7s** |

A 17x improvement, and it was the single largest latency bug in the app. Worth
being precise about how it was found, because the obvious theories were all
wrong: `num_ctx` was innocent, prompt size was innocent, streaming was innocent
(blocking and streamed both hit the cache at 1.11s on an identical prefix), and
the stored rows were never rewritten. What actually mattered was that the *first
replayed message* moved, so no faithful replica of a *fixed* history reproduced
it — only real, sliding turns did.

Durable knowledge does not live in the replay. Facts, notes and visual memory all
reach the model through the system prompt, which is stable and cached, so
trimming the replay costs anaphora — "what about the second one?" — and not the
assistant's memory. 8 keeps the last four exchanges for that, at 3.7s.
`test_the_replayed_history_window_slides` asserts the mechanism (the system
prompt stays identical, position 1 does not) rather than a timing, so it still
holds on hardware too fast to measure this, and
`test_the_default_history_window_is_small` pins the value.

Two things still cost a slow turn, both rare and both correct to pay for:
saving a fact or a note changes the system prompt, and `list_more_tools`
widening the working set changes it too. That is the main reason a learned tool
is kept from one turn to the next: a working set that grows and shrinks
invalidates the cache every time.

**How often you pay it is a setting.** The cache dies with the model, so any
gap longer than `keep_alive` means a full 92s prefill on the next question. It
holds the model for `30m`, which costs 3.4 GB resident (measured as the
`llama-server` working set) on a 15.7 GB machine and saves the cold start
across an ordinary pause. Lower `keep_alive` to `5m` if that RAM is worth more
to you than the latency.

Dropping tools from the front of the set was measured and rejected. It does cut
the cold turn, but routing falls off a cliff, because this model will not
reliably reach for `list_more_tools` when the tool it wants is missing:

| Core set | Schema | Right tool | Wrong | No tool |
| --- | --- | --- | --- | --- |
| **38, unselected** | 14,133 chars | **26/36** | 4 | 4 |
| 16 | 6,980 | 16/36 | 10 | 8 |
| 10 | 4,796 | 12/36 | 15 | 6 |
| 6 | 3,068 | 7/36 | 18 | 9 |

Tightening the tool *descriptions* instead of dropping tools looked like the
obvious next saving, so the schema was measured instead of guessed at. It does
not hold up:

| Part of the 38-tool schema | Chars |
| --- | --- |
| Tool descriptions | 4,650 |
| Parameter descriptions | 1,447 |
| Property names, `type`/`properties`/`required` scaffolding | ~8,000 |
| **Total** | **14,133** |

Only 6,097 characters are prose, and the wording that earns the 26/36 is exactly
the part that has to stay. The remaining 8,000 is JSON structure that vanishes
only if tools are removed, which the table above shows is a bad trade.

**So the tools are chosen per turn instead of cut from the list.**
`Brain.select_tools()` scores all 72 registered tools against the utterance — a
category keyword from the tool's tags is worth most, a word of its own name next,
and a description word least, because descriptions share vocabulary with each
other and a single common word means nothing — and sends the best `tool_select_max`
of them. Every tool stays registered and reachable; this only decides what the
model is *shown*, which is the difference that matters.

The measured result, with the model unloaded before each call so both sides pay
full prefill:

| Working set | Tools | Schema | Prompt tokens | Cold prefill |
| --- | --- | --- | --- | --- |
| Every core tool (before) | 38 | 14,241 chars | 3,054 | **93.5s** |
| Selected for the turn (after) | 11 | 5,148 chars | 1,195 | **34.5s** |

**2.6x fewer prompt tokens and 2.7x less cold prefill, about 59s saved on a cold
first turn.** It is not free, though, and the cost lands in the wrong place.
Ollama can only reuse a cached prompt when the new one starts with the same
tokens, and a tool set chosen per question guarantees it does not. Measured in
one process with the model already loaded, four questions:

| | cold | 2nd question | 3rd question | repeat |
| --- | --- | --- | --- | --- |
| `tool_select_max=10` | 44.6s | **36.3s** | **14.5s** | 0.13s |
| `tool_select_max=0` | 68.1s | **0.49s** | **0.62s** | 0.61s |

With a fixed working set the prompt is byte-identical every turn, so only the
short user message is new and everything else comes from the cache — about half
a second, for *any* question. Selection wins the cold turn by 23s and loses every
warm turn by 15–40s.

It is also **more accurate**, which is the part that was not expected. Twenty
requests, each naming one tool it needs:

| Working set | Found the tool |
| --- | --- |
| Every core tool (before) | **13/20** |
| Selected, cap 4–8 | 19/20 |
| Selected, cap 10 | **20/20** |
| Selected, cap 24 | 20/20 |

Showing every core tool found only 13 of 20, because seven of them — `open_app`,
`take_screenshot`, `power_action`, `list_processes`, `media_control`,
`list_displays`, `delete_note` — are not core tools and were reachable *only* via
`list_more_tools`, which this model called zero times in 21 requests. Sending
fewer, better-chosen tools beats sending all the core ones.

**So it ships switched off**, because the cold turn is the wrong thing to
optimise: it is paid once per `keep_alive` window, while the warm turn is paid on
every question, and on this hardware the fixed working set wins that trade
outright. `tool_select_max` defaults to `0`, the old behaviour. The setting is
kept rather than deleted, because it is the better answer to a better
tool-routing story, and the balance above could flip on a machine with faster
prefill or a larger cache. Re-measure both rows before changing it back.

Two safeguards keep a mis-scored tool from becoming a missing one. `list_more_tools`
is in the selection unconditionally, as `TOOL_SELECTION_FLOOR`, so the route back
to the full catalogue is always offered; and scoring is deliberately generous, so
the cap — not the threshold — is what decides between plausible candidates.

The category table used for scoring (`TOOL_TAGS`) is separate from the one
`preroute()` uses (`KEYWORD_TAGS`) on purpose. `preroute` adds a whole category to
the working set permanently, so its table stays narrow; scoring picks at most ten
tools, so its table covers all 20 tags. Sharing them would have let `preroute`
drag the working set back up.

Replies stream, so text appears as it is generated.

`request_timeout` is 420s rather than something tighter precisely because the
cold first turn is legitimately slow, and a timeout there is a false failure.

### Answering fast: the no-tools path

Sending the 38-tool working set is the most expensive thing JARVIS does, and most
questions need none of it. Measured on a warm session, same questions both ways:

| Question | Tools sent | Time to first output |
| --- | --- | --- |
| How do I bake sourdough bread | 0 | **0.95s** |
| Tell me a short joke about a duck | 0 | **0.39s** |
| How do I bake sourdough bread | 38 | 8.97s |
| Tell me a short joke about a duck | 38 | 7.65s |

The tool schemas are most of the prompt, so leaving them out is worth roughly an
order of magnitude. A question that shows no sign of wanting a tool is asked
without any.

**The gate leans deliberately towards the tools**, because the two failure modes
are not symmetric — a slow correct answer is fine, a fast wrong one is not. A
question takes the slow path if any of these hold:

- it contains a word from the tool tag vocabulary (`TOOL_TAGS`);
- it contains an action verb (`FAST_PATH_BLOCKING_VERBS`);
- an image is attached;
- the request is empty.

The verb list exists because the tag vocabulary has a gap that is only visible in
use. None of "open notepad", "launch the browser", "install firefox" or "find my
keys" contains a single tag word, and all four are obvious tool calls. Verbs are
matched as whole words, so "opening hours" and "running water" stay on the fast
path.

Politeness is deliberately *not* in that list. "Can you", "could you" and "please"
appear in almost every request, including ones that need nothing, so blocking on
them would send nearly all traffic down the slow path and the fast path would
never run.

**And the guess can admit it lost.** If a no-tools answer comes back as a refusal
— "I don't have access to that" — the gate misfired, and JARVIS spends the slow
call after all rather than telling the user it cannot do the thing it just did
twenty times a day. `fast_path_refused()` is that check, and
`test_a_refusal_gets_retried_with_tools_instead_of_returned` holds it in place.

One thing that needed fixing once measured: tag matching was plain substring
matching, so "sourdough **bread**" contained the file tag "**read**" and sent a
baking question to the filesystem tools. Single-word tags are now matched on word
boundaries; multi-word ones are phrases and are matched as they are.

Set `fast_path = false` in the config to always send the tools.

### Hearing you back quickly: the acknowledgement cue

A warm turn is 15–30s, and silence is what makes a long wait feel broken rather
than slow. So JARVIS acknowledges the moment it has a transcript, before it starts
thinking.

| Step | Cost |
| --- | --- |
| Build the 45ms tone | 0.03ms |
| Write it to an already-open output stream | **0.22–2.52ms** |
| **Total before the cue is audible** | **under 3ms** |

That clears the 50ms bar, and the only reason it can is that there is no file to
read and no codec to run — the tone is generated into a NumPy buffer
(`ack_tone`), so the whole cost is one `stream.write`.

**It is a tone, not a phrase, on purpose.** Spoken acknowledgements were rejected
for a specific reason: the same synthesiser that would speak the acknowledgement
would then speak the answer, and queueing one behind the other delays the answer —
the part the user is actually waiting for. A cue cannot do that, and does not read
as a fake reply.

**The output stream is opened at startup, and primed.** This was the real
finding. Opening a `sounddevice.OutputStream` costs ~200ms here, and the *first
write* costs a further ~220ms while the driver primes — measured, not estimated.
Paid at reply time, the first cue of a session lands 923ms after the user stopped
talking, which is worse than the 236ms it was meant to cover. So `VoiceLoop.run()`
calls `warm_ack_stream()` while idle, which opens the stream and writes 20ms of
silence. Inaudible, and it moves the whole cost off the critical path:

| | Cue latency |
| --- | --- |
| Opened at reply time | 923ms first, then 0.26–0.32ms |
| Warmed at startup | 2.52ms first, then 0.22–0.28ms |

A zero-length write does not prime the driver, which is why the warm-up writes
real samples of silence.

The stream is process-wide and shared, because TTS and the cue want the same
output device and two independent `OutputStream`s on one Windows device reliably
produces exclusive-mode errors. If the device cannot be opened at all, that
failure is latched: a missing sound card costs one attempt, not one per turn,
forever, and the turn continues silently. There is deliberately no `sd.play()`
fallback, because it opens and closes its own stream and would cost the ~200ms
this cue exists to avoid, on every turn, forever.

Set `acknowledge_sound = ""` to turn the cue off, or pick another of `tick`,
`chime`, `blip`, `soft`.

### What 50ms is and is not

Worth being precise, because "respond in 50ms" has a floor under it. Measured on
this machine:

| | Time |
| --- | --- |
| Localhost round trip, no model work | 8.7–16.2ms |
| Fastest possible model output (perfect cache hit, one token) | **236ms** |
| Time to first output, no tools | 0.33s |
| Time to first output, 38 tools, cold | 103.0s |

A complete model-generated answer in 50ms is not reachable by tuning: the fastest
single token ever observed here was 236ms, so 50ms is 5–10x below the floor. A
smaller model does not fix it either — at 100 tok/s, 10ms per token, this hardware
generates at 6.5–8.9 tok/s.

What 50ms *can* cover is acknowledging you, which is what the cue is for, plus
genuinely fast answers on the no-tools path, which measures 0.39–1.69s.

### Choosing a model

`qwen2.5:3b-instruct` is the default because it is the only installed model that
emits tool calls directly and reliably enough. Its weakness is tool selection,
which was measured rather than guessed at. A 21-request benchmark, each asked
for exactly one round with no tool executed, scored **14/21** before any fixes.
The failures were specific, and each is now handled:

| Failure | Fix |
| --- | --- |
| "click save" went to `find_on_screen`, which only reports coordinates | its description now says it does not click, and routing says to use `click_on` |
| "turn on the living room lights" went to `open_app` | `open_app` says it opens programs only; `home` tools are pre-loaded from keywords |
| "send an mqtt message…" returned nothing at all | `mqtt` tools are pre-loaded from keywords |
| "what do you remember about X" was answered from imagination | answered in code from the database, no model involved |

That took the benchmark to **20/21**, and the remaining case is now answered
without a round trip as well. Three of the four fixes are deliberate routing
rather than prompting, because this model follows a short list of literal
examples but keeps ignoring instructions to distrust itself.

Two routes are pre-computed rather than chosen, both in `brain.py`:

- **Keyword pre-loading** (`KEYWORD_TAGS`) puts whole tool categories in the
  working set before the schemas are built, because `list_more_tools` was never
  called for smart-home or MQTT requests even when the question named the
  category. `home` adds 4 tools, `mqtt` adds 2, against 38 already loaded.
- **Memory questions** (`memory_topic`) are answered directly from the database.
  Asked "what do you remember about the ferret", the model replied "I don't have
  any saved information" while the detail was in the prompt, because it was
  copying its own earlier wrong answer out of the history. Hiding the other
  tools, using a separate system message, and appending to the system prompt
  were all measured and all still wrong. Answering in code is correct, and takes
  0.0s instead of 150s.

Because a wrong reply of its own was the cause, JARVIS does not treat its
statements as evidence. A reply counts only once you have confirmed it, with an
explicit verdict such as "that's right" — vague agreement like "ok thanks" is
ignored, since accepting something by habit is not verification. "No, that's
wrong" withdraws it again, so one mistaken confirmation does not become a
permanent wrong answer. The model cannot set this flag itself; only your
agreement or disagreement reaches it.

A common request also failed silently: `brain._tools()` returned whatever was
registered, which is a single tool if the tool modules have not been imported,
and the model replied with nothing at all. It now loads the registry itself and
warns if any tool in the working set has no schema.

The brain is text-only, so the screen reaches it as OCR text and element lists,
never as image bytes. JARVIS probes the model's capabilities at startup, and if
you swap in a vision-capable model it will pass real screenshots instead.

Set `vlm_describe_model` in your config to use a different model for describing
than for pointing.

## How seeing works

`perception.py` captures the foreground window with MSS, reads text with
Tesseract, and finds clickable elements. Nothing is captured in the background:
the screen is only photographed when a tool asks for it.

OCR is the primary channel because it is fast and exact. The vision model is a
fallback for what OCR cannot see, and there are two of them because pointing and
describing are different jobs:

| Job | Model | Why |
| --- | --- | --- |
| Point at a target | `moondream` (896px) | It can point and return coordinates. |
| Describe the screen | `qwen2.5vl:3b` (448px) | It can actually read a screen. |

`moondream` cannot describe a real window. Measured on a 1920x1200 OpenCode
window, asked what application was shown, it answered
`urn:jira:issue/uid:urn:jira:issue/uid:...`, and on a later pass
`1. Read the instructions.` — a plausible sentence about a window that was never
on screen. `qwen2.5vl:3b` read the same window back correctly and verbatim.
Set `vlm_describe_model` to `""` to go back to using `moondream` for both, but
expect invented answers.

Two things protect you from a small model inventing something convincing:

- **The degenerate-output filter** rejects repetition loops, consonant soup, and
  answers that are mostly a URL. A URL is only treated as a confabulation when
  little else is said, because a correct read of a terminal legitimately
  contains `http://localhost:11434/api/tags` and rejecting that threw away a
  good answer.
- **Corroboration against OCR** (`Vision.supported_by`). The model's answer is
  checked against the text actually on screen, which costs nothing because the
  OCR has already been done. An answer with no overlap is reported with
  `vision_unverified` and an instruction not to state it as fact. This is what
  catches the dangerous case: a fluent wrong answer, not an obvious one.

When OCR has already read the screen, the model is asked one question about the
window rather than asked to transcribe text, since OCR is the better reader and
asking both only invites the model to contradict a correct reading.

**Cost, and what does not help.** With `qwen2.5vl` on CPU a describe call is
78-93s, and about **97% of that is prefilling the image**: the model reports
1099 prompt tokens at 96px, 128px, 168px, 224px, 336px, 448px, 560px, 672px,
896px and 1120px alike, because it pads whatever image it is given to a fixed
token grid. Cropping does not help either, for the same reason. So shrinking the
picture saves nothing at all, and the earlier note here blaming image size was
wrong. `vlm_describe_max_side` is kept at 448 only to keep the payload small;
pointing keeps 896px because there the detail is the point.

**The cost is per image, not per question.** This is the part the old
description of this code had backwards, and it is worth stating precisely,
because it is what makes the cache below work:

| | Prefill rate |
| --- | --- |
| First question about a given image | ~78-88ms per token |
| Any later question about the *same bytes* | ~0.5ms per token |

Three questions against one screenshot took **71.74s, then 6.28s, then 3.11s**.
Shift one region of that image by 40 levels and the first question is back to
**91.66s**. The first call is what pays for ~1100 image tokens; `ollama` keeps
the vision-encoder result and reuses it for identical bytes. The prompt text
does not matter — a completely different question against the same image still
costs 8.10s, then 2.55s.

So `describe()` caches descriptions keyed on the encoded payload plus the
question. Measured against the real model on a real screen:

| | Time |
| --- | --- |
| First description of a screen | 81.89s |
| Same screen, same question, asked again | **0.01s** |
| Screen changed by one region | 86.18s |

The key deliberately hashes the *encoded payload* rather than the array, so this
cache and the server's own agree about what "the same image" means — a hit here
predicts a cheap call there rather than the two disagreeing. The question is
part of the key, because answering the wrong question quickly is worse than
answering slowly. Only a *usable* answer is stored: caching a confabulation
would leave that screen permanently undescribable, since every later call would
be a hit on the junk and the miss would never be retried.

Set `vlm_describe_cache_entries` to 0 to disable it, which is the honest setting
if the screen is expected to change between every question anyway.

**What still does not help, re-measured.** Swapping in the faster model is a
false economy. `moondream` prefills 748 tokens in ~14s where `qwen2.5vl` takes
~80s, so it looks like a 6x win — but it is not a usable substitute. Asked about
a deliberately photo-like screen with no text on it, `moondream` answered
`urn of water` where `qwen2.5vl` answered `a window showing a beach`, and on a
blank screen it returned nothing at all. Using it as a cheap first pass and
escalating would cost 14s + 80s and end up slower than simply waiting, with a
worse answer on the way there.

Two things that do help are enforced:

- `vlm_describe_num_predict` (96) caps generation. Good answers are 15-25
  tokens, so the cap costs nothing there, but `moondream` once spent all 256
  tokens emitting `urn:jars:li:9:0:0:0...` for 16s of generation and no answer.
- `vlm_describe_budget_seconds` (150) bounds a whole describe. It is protection
  against a cold image or a model that is slow for some other reason, not
  against three questions each costing 80-90s, which is what the old comment
  here claimed. Once the budget is spent the remaining questions are skipped and
  the partial answer is returned, so the answer degrades instead of the user
  hanging. A specific question you asked directly is never gated by it.

Repeats are genuinely cheap: the same screenshot bytes hit Ollama's prompt
cache, so an identical call drops from 97.8s to 3.0s. That is a cache effect,
not a smaller workload, and a new screenshot misses it.

`moondream` is not a faster way to describe. On the same prompt and image it
prefilled in 14s against qwen's 90s, but emitted `urn:jars:li:9:0:0:0...` at
448px and returned nothing at all at 320px and 224px. Keep
`vlm_describe_model` pointed at a model that can read a window.

If the model does not support images, the turn still completes: the tool result
is rewritten into a short text caption so the brain is told what happened
instead of receiving a wall of base64.

## How acting works

Three layers gate every action.

**Autonomy** decides whether an app may be operated at all. Credential managers
and login, lock, and payment screens are denied outright and cannot be
overridden with `/allow`. Anything that looks like a credential field blocks
typing even inside an allowed app.

`/deny` outranks the config allowlist and survives a restart, because the
allowlist is rebuilt from config on every launch and a denial that quietly came
back would be worse than no denial at all. `grant()` refuses a denylisted app
outright, so a new tool cannot route around the gate, and `/allow` checks the
denylist before it asks for confirmation rather than after.

**Confirmation** is requested for dangerous tools unless you pass
`--confirmation off`.

**Verification** re-observes the screen after the action. A no-change result
records a failure, warns against repeating it, and demotes the confidence of
the remembered coordinates until later successes restore it.

Learned coordinates are stored as element centres, not top-left corners, so
clicking a remembered button lands on the button even if the window moves.

**What the learning systems actually do**, all three measured rather than
assumed. Reinforcement and demotion behave as intended: confidence rises with
hits, falls monotonically with misses, a control that keeps failing stops being
offered after four misses, and later hits rescue it. Three records were wrong
about what they were a record *of*, and are now fixed:

- A hit or miss describes a **position**, so a control that moves starts
  unproven. It used to keep its tally across a move, which left a Save button
  that had shifted from (100,200) to (999,888) still reporting 0.857
  confidence about where it used to be — and the locator clicks that coordinate
  at 0.9x whenever OCR cannot see the control. The threshold scales with the
  control, because the map is fed from every OCR match and one-pixel jitter must
  not reset learning.
- A procedure's successes score its **steps**, not its goal. Re-teaching a goal
  with different steps cleared the tally; before, steps JARVIS had never run
  reported themselves as proven five times over.
- `likely_open_now` has to mean now. Habit counts never decayed, so an app last
  used a year ago outranked one opened seconds earlier. Volume is now scaled by
  recency on a 30-day half-life, and each entry reports its age.
- `recall_procedure` without an app used to return the top word-overlap match as a
  confident hit, and the overlap threshold was only 0.1. Asking for "open the
  settings page" therefore returned a Notepad procedure called "open the file
  menu", steps included, and those steps are clicks and typed text that then get
  replayed in whatever window happens to be focused. An identical goal still
  counts with no app named; anything else is reported as a near miss, with the app
  named, so the model can ask again with it. Naming an app was always exact.
- The keyword half of visual recall never worked. `observations_fts` was declared
  as an FTS5 *external content* table over `observations`, which tells FTS5 to read
  the indexed columns out of that table — but the index's one column is `body`,
  and `observations` has no such column, only `text` and `window_title`. So every
  read raised `no such column: T.body`. `recall` caught `OperationalError` and
  moved on, which is why this was invisible: keyword search silently contributed
  nothing to any search, and the vector half carried the result alone. `prune`
  warned on every startup for the same reason, and because it threw before
  `commit()`, the pruning it was trying to do was rolled back every time. The
  index is contentless now (`content=''`) and fed by the triggers that already
  existed, which removes the content table it was disagreeing with; FTS5's own
  `'rebuild'` is gone with it, replaced by a refit from `observations`. An
  existing database is detected on open by its stored declaration and remade,
  because FTS5 only creates its table when absent and would otherwise leave the
  broken one in place. All 13 observations were kept. `TestVisionMemoryFts`
  covers the read, the record, the retraction on delete, the prune, and the
  migration, including a check that the old declaration really does fail.
- **The second monitor was unreachable, and the reason was DPI.** Three separate
  faults, all silent, all found by extending the desktop to a 1920x1080 monitor
  beside the 1920x1200 laptop panel (which runs at 150% scaling, so the two
  screens were at different scales — the hard case):
  - `Mouse.size()` returned `pyautogui.size()`, which only ever describes the
    *primary* monitor. `validate()` then treated anything past x=1919 as off the
    edge of a 1920-wide screen and refused it, so the mouse could not go to the
    second display at all — while `mss`, which takes the screenshots those
    coordinates get matched against, cheerfully reported that display at
    (1920, 0). The bounds are now the joined virtual desktop, from
    `env.virtual_desktop()`.
  - The per-step clamp in the humanised move had the same primary-only bounds, so
    a move *across* onto the second display was pinned to the first screen for
    its whole duration and only arrived at the end.
  - Neither process DPI mode was set, so the coordinate space depended on
    **import order**. `jarvis.mouse` imports pyautogui lazily and pyautogui calls
    `SetProcessDPIAware()` — system-wide, not per-monitor — on import. After that,
    mss reported the 100% second monitor as **2880x1620 at an offset of 2880**,
    because the 150% primary's scale got applied to every screen. So the very
    first mouse action silently corrupted every later screenshot and coordinate.
    `env.claim_dpi_awareness()` now claims per-monitor v2 before any desktop
    library loads, in `jarvis/env.py`, which is the module that already existed to
    make those libraries safe. It is cached, because awareness is a
    once-per-process property and a second call would fail every entry point and
    then misreport whichever fallback happened to succeed.

  Verified afterwards, with the model unloaded and nothing clicked: the mouse
  reaches the centre and far corner of both displays and reads back exactly, a
  humanised move crosses screens, `validate()` accepts (0,0) through (3839,1199)
  and rejects (3840,0), and a region capture of the second display is
  **pixel-identical** to the same rectangle cropped out of a full-desktop
  capture. `mouse_position` reports the full desktop and which display the cursor
  is on — it used to report a 1920-wide "screen" for a cursor at x=2880, and
  answered "display 0", which is the union rather than a display.
  `TestMultiMonitorGeometry` stubs a two-screen desktop so all of this runs on a
  single-screen machine too. `--doctor` now reports the display count, the
  desktop size and the DPI mode.

## How the web tools behave

`get_weather` uses Open-Meteo, not wttr.in. wttr.in now answers every request,
including `?format=3`, with its HTML landing page; the original parser did not
notice and handed the model a page of CSS as a "forecast", which it then
correctly reported as missing data. Open-Meteo is a real JSON API and needs no
API key. Postcodes are not supported, because its geocoder is name-only, and
the tool says so plainly instead of claiming the place does not exist.

Every JSON fetch goes through a guard that rejects a response whose content
type is not JSON. A free endpoint answering with a web page is a common failure
and should surface as an error, never as a confident wrong answer.

## Tests

```powershell
python -m unittest jarvis_tests -v
```

349 tests, about 26 seconds, no external network and no model needed. They
cover the autonomy gate, tool schema validation, argument coercion, VLM output
filtering, coordinate parsing, vision-model corroboration and sizing,
visual-memory confidence, the visual-memory keyword index, multi-monitor
geometry, microphone selection, the no-tools fast path, the acknowledgement cue,
and the prompt helpers.
The routing added for tool selection is covered the same way: keyword
pre-loading, the memory-question patterns, and the evidence filter that stops a
search result from counting the question itself or the model's own past wrong
replies as evidence.

Relevance-based tool selection has its own class, `TestToolSelection`, because
a scoring change that quietly drops the tool a request needs is invisible until
the assistant says it cannot do something it plainly could. It asserts the tool
each of 20 requests needs survives selection, that `list_more_tools` is offered
whatever was said, that the cap is respected, that a cap of 0 restores the old
behaviour, that a tool learned from `list_more_tools` is not dropped on the next
turn, and that **the default is 0** — so the regression above cannot be
reintroduced by changing a default without re-running the measurement. It also
pins the reason selection is better than sending all 38 core tools, by asserting
those seven non-core tools really are unreachable when it is off.

Selection is deterministic, but that only makes a *repeated* question hit the
cache — it does not rescue a new one, because a different question picks a
different tool set. That asymmetry is the whole reason the default is 0, and no
amount of test coverage can make it go away.

The no-tools fast path has two classes, split by what kind of bug they are
looking for. `TestFastPathGate` is about the *judgement*: that a plain question
is let through, that a tool-ish one is not, that the verb check catches the gap
the tag vocabulary leaves ("open notepad"), that it does not fire on substrings
("opening hours", "running water"), that politeness does not force every request
down the slow path, and that a refusal is detected without mistaking an ordinary
answer for one — a false positive there throws away a good answer and pays the
slow call for nothing. `TestFastPathRouting` is about the *wiring*: that a simple
question is answered with zero tools sent, that a tool request is offered the
tools, that a refusal triggers the retry, and that `fast_path = false` and an
attached image both take the slow way.

The acknowledgement cue is covered by `TestAcknowledgeCue`, which pins the claims
this README makes: that the buffer is short, quiet and zero at both ends so it
cannot click; that it actually oscillates rather than being a window of silence;
that the stream is written to rather than played synchronously; that
`VoiceLoop.run()` warms it and `_run_turn()` fires the cue *before* calling the
model; that the stream is shared across cues rather than reopened per turn; that
a dead output device is latched instead of retried every turn; that the warm-up
primes the driver with real samples of silence; and that queueing a cue costs
under 50ms.

Prompt-cache stability is covered too, because it is invisible until it is
expensive: that `system_prompt()` does not change when the clock moves, that the
clock still reaches the model on the user's message rather than in the system
prompt, and that stored history does not accumulate clock notes, which would
break the cache again a turn later.

The describe cache has its own class, `TestVlmDescribeCache`, because a stale
description served for a screen that has moved on is worse than no description
at all. It asserts that the same screen is described once, that a changed screen
is described again, that a different question is a different answer, that the
multi-question path is cached as one unit rather than three independent
questions, that the cache is bounded and evicts least-recently-used, that
disabling it always asks, and — the one the code got wrong first time — that an
*unusable* answer is not stored, since caching a confabulation would leave that
screen permanently undescribable. It also pins that the key follows the bytes
actually sent to the model, so a hit here predicts a cheap call to the server.

Confirmation is covered as well: that a reply starts unconfirmed, that user
rows cannot be marked confirmed, that a confirmed reply becomes evidence while
an unconfirmed one does not, that agreement and disagreement each do what they
should, that vague agreement is refused, that a verdict reaches back only one
reply, and that an existing database gains the column without losing rows.

The learning stores are covered from the same angle: that a moved control loses
its old record, that re-detection jitter keeps it, that the move threshold scales
with the control's size, that a re-taught procedure's tally is cleared but a
same-steps re-save keeps it, and that recent app use outranks old volume while a
genuine current habit still wins.

Procedure recall is covered for the app confusion it caused: that another
program's steps are not returned as a hit when only a goal is given, that the
suggestion still names the app, that an identical goal is still a hit with no app
named, that naming the right app finds the right one when two apps share a goal,
and that naming an app which has none does not fall back to a different one.

**The `dangerous` flag enforces nothing, so the tests check the tools instead.**
`dangerous=True` is read in exactly one place in the package: `__main__.py`, to
print "[needs approval]" in a listing. What actually stops a tool is a
hand-written `_approve()` call inside its handler, so the flag and the call are
two pieces of information kept in different places by hand. A test reads the
source with `ast` and asserts that every flagged tool asks before it acts,
either directly or by delegating to another flagged tool that does
(`ha_run_script` hands off to `ha_call_service`, which is where the confirmation
lives). It was verified to fail when `_approve` is removed from `power_action`,
so it is not a test that passes by construction. `kill_process` and
`abort_shutdown`, which force things and had no coverage, now also check that a
declined confirmation reaches neither `kill()` nor a command.

**Approval is recorded, not just answered.** The confirmation used to be a
boolean that vanished once the tool returned, so "what did I just agree to, and
was that the same thing it went on to do?" had no answer. Every decision now
goes through `jarvis/approval.py` and lands in one process-wide log: the tool,
the exact wording shown, the outcome, how it was reached, and when. That last
part matters because `--confirmation off` also produces a yes, and a reader of
the trail must be able to see nobody was asked. A decision is also fingerprinted
against its tool *and* its detail, so a yes for closing Notepad is visibly a
different decision from a yes for restarting the machine.

For a front end that wants "don't ask me again for this", `grant_token()`
returns the fingerprint plus a deadline and `check_token()` refuses it once
stale, so a remembered answer cannot quietly become a permanent one. Every call
still prompts by default; skipping a prompt is a convenience, and a convenience
on by default eventually skips the one prompt that mattered. A stale or
mismatched token falls back to asking rather than being trusted.

**Schema enums are enforced, not just documented.** Every tool schema declares
enums, and nothing checked them, so `button="leftt"` reached the handler
unchanged. `mouse.click` then handed the string to pyautogui, which raised a
bare `ValueError` *after* the autonomy check and after a before/after
screenshot had been written to visual memory. The registry now walks the
arguments against the schema - including enums nested in objects and on array
items - and raises a `ToolError` naming the field and the allowed values, which
is something a 3B model can correct itself from. Matching ignores case and
stray spaces and rewrites the value to the canonical member, because handlers
index tables with the exact string.

`power_action` is the interesting case: the prompt advertises five actions but
the handler understands 36 words, because a small model sends "reboot" and "log
off" regardless. Listing all 36 in the schema would cost about 150 prompt
tokens and turn a 5-way choice into a 36-way one, so the short list stays in
the prompt and the wider vocabulary is declared separately as `accepts`, used
for validation only and never sent to the model. A test asserts the two lists
cannot drift apart.

**Speech is tested with a synthesized voice.** The only parts of the voice path
that genuinely need a person are the microphone and the wake word, so
everything after them is covered: Windows SAPI speaks a known phrase, and it
goes through the same 16 kHz-mono-to-Whisper path a recorded utterance takes.
Measured on this machine, "what time is it right now" comes back as
"What time is it right now?" in 1.0s. The quiet-audio guards are checked with a
spy model so they cannot pass by dropping everything. That last test needs
faster-whisper and skips cleanly when it is absent; the guards do not.

**Home Assistant and MQTT are tested against real servers on loopback**, in
`jarvis_tests/fakes.py`: an HTTP server speaking the Home Assistant API, and a
small MQTT 3.1.1 broker. Mocking `requests` or `paho` would only have proved the
mock worked. This is where the five IoT bugs came from, and the suite is slower
for it - most of the twenty seconds is real socket connect and retry time
against a port with nothing listening.

What that found, all of which had been invisible without a server to talk to:

- paho only reads its socket from the network thread it starts itself.
  `mqtt_publish` never started one, so **every QoS 1 and QoS 2 publish could not
  complete its handshake** and reported a delivery the broker had never
  acknowledged. QoS 2 never sent its `PUBREL`, so those messages were simply
  lost.
- A refused connection is a paho **return code, not an exception**, and bad
  credentials are not even known until `CONNACK` arrives. Both were ignored, so
  publishing against a broker that had rejected the login **reported success**.
- `mqtt_listen` subscribed even after the broker refused the login, and a
  refused subscription is indistinguishable from a silent sensor. It now says
  the broker rejected the connection.
- `ha_run_script` read `friendly_name` with an empty default, so **any script
  without a friendly name could not be found by its own name**.
- `_entity_id` promised in its docstring that the model could say "the kitchen
  lights" instead of "light.kitchen", and actually produced `the_kitchen_lights`,
  which is not an entity id. Names are now resolved against the entities Home
  Assistant reports, an ambiguous name is reported rather than guessed, and an
  unresolvable one is an error rather than a malformed request.

One latency fix came out of it too: paho's network thread blocks on `loop_stop`
until its `select` times out, so stopping the loop before disconnecting cost a
full second on **every** publish and listen. The twenty-second runtime is mostly
the two tests that confirm a dead broker and a dead host are reported clearly.

## What acting on the machine actually does

`power_action` was the most dangerous thing found in this project, and none of
it was visible without checking what command each input produced.

Its schema listed an `enum` of the five supported actions. Nothing enforces
enums: `registry.call` type-checks arguments and binds them to the signature,
and that is all. The body was then a chain of equality tests ending in a
shutdown branch, so **any word the chain did not recognise became a shutdown**.
Asking JARVIS to reboot powered the machine off instead of restarting it.
Asking it to log off did the same instead of signing out. "hibernate", "suspend"
and "wipe" were all shutdowns too.

Actions now resolve through an explicit table before anything is approved or
run, so an unrecognised word is an error rather than a default, and the user is
never asked to confirm a request that cannot be carried out. The test suite
asserts that every value in the schema's enum is one the table understands, so
the two lists cannot drift apart again, and that nothing reaches `Popen` for an
action the table rejects. `kill_process` was checked for the same shape of bug
and is fine: it matches a process name exactly and errors if it matched nothing.

`list_displays` was broken in a smaller way. It called `mss.mss()` on the object
`_open()` had already returned, so the tool raised `AttributeError` every time
it was used - which is the only reason anyone would have found out, since a
single-monitor machine has nothing to discover. Bad screenshot regions now fail
as a readable `ToolError` instead of leaking whatever the capture library
raises. `abort_shutdown` was run against the live machine with nothing pending
and correctly reported `aborted: false` along with Windows' own explanation.

Only one display is attached here, so the multi-monitor cases are covered
against a fake session rather than a second screen: that a secondary display
reports its virtual-desktop offset, including the negative origin a monitor to
the left of or above the primary has, and that the capture session is closed.

Which display is primary was a separate bug. The tool hardcoded `primary` to
index 1, which happens to be right on a single monitor and wrong on a real
multi-monitor machine, because `mss` lists the virtual desktop union at index 0
and then the real displays in whatever order Windows reports them. It now reads
`is_primary` from the capture library's own monitor entry, and falls back to the
old index only for `mss` builds too old to carry the flag. The panel name is
included when the library provides it, which is how the attached display here
identifies itself as a 1920x1200 FlexView.

Asking Windows directly with `GetMonitorInfoW` was the obvious alternative and is
wrong. That call returns DPI-virtualised coordinates unless the process has
opted into DPI awareness, so on this machine it reports the primary as
1280x800 while the display is actually 1920x1200 at 150% scaling. Matching
rects between the two APIs would have silently failed on exactly the
high-DPI displays most likely to be multi-monitor.

Note the command: the suite lives in `jarvis_tests/__init__.py`, so the usual
`unittest discover` pattern matches nothing and reports `Ran 0 tests`.

The suite never touches the real screen, the real mouse, the network, or your
actual database. That is deliberate: the failures worth guarding against here
are the ones where JARVIS types into the wrong window, and a test that clicks
something is no use as a guard.

It has already earned its keep. Writing it turned up a crash in the nested
screenshot path, two dangerous window titles that were not being blocked, and
argument coercion that mangled list arguments a model sends as JSON strings.

## Slash commands

| Command | Effect |
| --- | --- |
| `/vision` | Look at the screen now and print what is there |
| `/vision-memory` | Show learned controls, confidence, and procedures |
| `/forget-screen <app\|all>` | Erase learned screen memory |
| `/allow <app>` | Trust an app to click and type unattended |
| `/deny <app>` | Remove trust, permanently, even if it is on the allowlist |
| `/trusted` | List trusted, allowlisted, and denied apps |
| `/tools [tag]` | List tools |
| `/schema <tool>` | Show a tool's JSON schema |
| `/remember`, `/recall`, `/facts`, `/forget` | Facts |
| `/note`, `/notes` | Notes |
| `/history`, `/forget-all` | Conversation |
| `/status` | Model, tool, memory, and vision status |
| `/speaker`, `/voices` | Speech |
| `/quit` | Exit |

## Configuration

Settings live in `~/.jarvis/config.json`. Inspect with `--show-config` and
persist with `--write-config`.

Useful ones:

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | `qwen2.5:3b-instruct` | Brain model |
| `vlm_model` | `moondream` | Model used for pointing |
| `vlm_describe_model` | `qwen2.5vl:3b` | Model used for describing. `""` reuses `vlm_model` |
| `vlm_max_side` | `896` | Longest side sent to the pointing model |
| `vlm_describe_max_side` | `448` | Longest side sent to the describing model. Affects payload size, not latency: the model pads to a fixed token grid at every size tried |
| `vlm_num_predict` | `256` | Generation cap for pointing |
| `vlm_describe_num_predict` | `96` | Generation cap for describing. Bounds degenerate output |
| `vlm_describe_budget_seconds` | `150.0` | Wall-clock ceiling for a whole describe, across all its questions. `0` disables. Guards a cold image, not the per-question cost — only the first question is expensive |
| `vlm_describe_cache_entries` | `24` | How many screen descriptions to keep, keyed on the image bytes and the question. Least-recently-used. A repeat description of an unchanged screen measured 0.01s against 81.89s cold; `0` disables |
| `vlm_timeout` | `300.0` | Seconds before a vision call fails |
| `request_timeout` | `420.0` | Seconds before a brain call fails |
| `keep_alive` | `30m` | How long Ollama holds the model and its prompt cache. Costs 3.4 GB while resident; shortening it brings the 92s cold prefill back |
| `keep_last_messages` | `8` | How much raw conversation to replay each turn. **The most expensive setting here, and it looks free.** `recent()` takes the *newest* n rows, so the window slides every turn, the token after the system prompt changes, and the whole history block becomes unreusable. Mean warm prefill: 62.4s at 40, 3.7s at 8, 0.7s at 0. Facts and notes reach the model through the cached system prompt, so this only buys anaphora — raise it only with a measurement |
| `num_predict` | `400` | Reply cap. Generation runs at 6.5–8.9 tok/s on CPU, so this is also the main lever on how long a turn feels: 400 tokens is ~50s |
| `tool_select_max` | `0` | How many tools, chosen by relevance to the utterance, reach the prompt per turn. All 72 stay registered; this only narrows what is shown. `0`, the default, sends every core tool every turn, which keeps the prompt byte-identical so the cache keeps hitting. **Leave it at 0:** raising it cuts the cold turn from 68s to 45s but costs 15–40s on every *new* question, which is the common case. It does improve routing, 20/20 requests finding their tool against 13/20, so it is worth revisiting on hardware with faster prefill |
| `fast_path` | `true` | Ask questions that show no sign of wanting a tool without attaching any tool schemas. Worth about an order of magnitude: 0.39–1.69s to first output against 7.65–8.97s with the 38-tool block. The gate is lopsided towards the tools on purpose, and a refusal from a no-tools answer is retried properly rather than returned. `false` always sends the tools |
| `acknowledge_sound` | `tick` | Short tone played once the transcript exists, before thinking starts, so a 15–30s wait does not feel broken. Under 3ms from cue to audible. `""` disables it; `chime`, `blip` and `soft` also work. The output stream is opened and primed at startup, because opening it costs ~200ms and the driver's first write a further ~220ms |
| `confirm_destructive` | `true` | Ask before dangerous tools. `--confirmation` overrides per run |
| `autonomy_allowlist` | see config | Apps trusted by default |
| `autonomy_denylist` | see config | Never touch, cannot be overridden |

## Notes on this machine

The project directory contains `pyautogui.py` and `mss.py` shims that exist to
prove the imports are safe. `env.py` removes them from `sys.path` at startup, so
the real site-packages versions always win. `--doctor` verifies this rather
than assuming it.

## Layout

| File | Role |
| --- | --- |
| `assistant.py` | Wires every subsystem together |
| `brain.py` | Ollama tool-calling loop, dynamic tool selection |
| `perception.py` | Screen capture, OCR, elements |
| `vision_memory.py` | Observations, UI map, procedures, habits |
| `autonomy.py` | Allow, deny, credential safety |
| `locator.py` | Resolves a target from memory, OCR, or vision |
| `tools/mouse_tools.py` | Verified, gated screen actions |
| `tools/sight_tools.py` | Screen inspection and learning tools |
| `vlm.py` | Vision model calls, output filtering, describe budget |
| `approval.py` | Approval decisions: provenance, fingerprinting, expiry |
| `env.py` | Stops the project-root `pyautogui`/`mss` shims shadowing the real packages, and claims per-monitor DPI awareness before either loads |
| `tools/` | 72 registered tools |
| `jarvis_tests/fakes.py` | Loopback Home Assistant and MQTT servers for tests |
