"""Command line entry point.

    python -m jarvis                  voice mode with the wake word
    python -m jarvis --text           type instead of talking
    python -m jarvis --hud            the heads-up display, type into it
    python -m jarvis "what's the time"   one question, then exit
    python -m jarvis --doctor         check the whole install

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import __version__
from .config import available_models, load_config, pick_model, save_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jarvis",
        description="A local-first voice assistant. Ollama for the brain, "
        "faster-whisper for ears, a local voice for the mouth.",
    )
    p.add_argument("prompt", nargs="*", help="Ask one question and exit.")
    p.add_argument("--text", action="store_true", help="Type instead of talking.")
    p.add_argument("--hud", action="store_true",
                   help="Open the JARVIS heads-up display and type into it.")
    p.add_argument("--voice", action="store_true", help="Force voice mode.")
    p.add_argument("--no-wake", action="store_true", help="Disable the hotword.")
    p.add_argument("--no-speak", action="store_true", help="Do not speak replies.")
    p.add_argument("--doctor", action="store_true", help="Check the install and exit.")
    p.add_argument("--tools", action="store_true", help="List all tools and exit.")
    p.add_argument("--model", help="Override the Ollama model.")
    p.add_argument("--stt-model", help="Whisper size: tiny, base, small, medium.")
    p.add_argument("--device", type=int, help="Microphone device index.")
    p.add_argument("--list-audio", action="store_true", help="List microphones and exit.")
    p.add_argument("--confirmation", choices=["typed", "auto", "off"], default="typed",
                   help="How to gate dangerous tools. Default: typed.")
    p.add_argument("--write-config", action="store_true",
                   help="Write the current settings to your config file and exit.")
    p.add_argument("--show-config", action="store_true", help="Print settings and exit.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    p.add_argument("--version", action="version", version=f"jarvis {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)-16s %(message)s",
        stream=sys.stderr,
    )
    # PortAudio and Whisper are chatty at INFO.
    for noisy in ("jarvis.audio", "faster_whisper", "urllib3", "numba", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(
        model=args.model,
        stt_model=args.stt_model,
        audio_device=args.device,
        wake_enabled=False if args.no_wake else None,
    )
    if args.confirmation == "off":
        cfg.confirm_destructive = False
    elif args.confirmation == "typed":
        cfg.confirm_destructive = True

    if args.show_config:
        _print_config(cfg)
        return 0
    if args.write_config:
        path = save_config(cfg)
        print(f"wrote {path}")
        return 0
    if args.list_audio:
        return _list_audio()
    if args.tools:
        return _list_tools()
    if args.doctor:
        return _doctor(cfg)

    prompt = " ".join(args.prompt).strip()

    try:
        return _run(cfg, prompt, text_mode=args.text, force_voice=args.voice,
                    no_speak=args.no_speak, hud=args.hud)
    except KeyboardInterrupt:
        print("\n  offline.")
        return 130


def _run(cfg, prompt: str, text_mode: bool, force_voice: bool, no_speak: bool,
         hud: bool = False) -> int:
    from .assistant import Assistant
    from .brain import Brain, BrainUnavailable
    from .memory import Memory
    from .tts import TTSUnavailable, make_speaker
    from .tools import load_all

    load_all()  # registers every tool handler
    memory = Memory(cfg.resolved_db_path)
    print(f"  memory: {cfg.resolved_db_path}", file=sys.stderr)

    try:
        brain = Brain(cfg, memory)
    except BrainUnavailable as exc:
        print(f"\n  Brain unavailable: {exc}\n", file=sys.stderr)
        print("  Install and start Ollama, then pull a tool-capable model:\n", file=sys.stderr)
        print("    ollama serve", file=sys.stderr)
        print(f"    ollama pull {cfg.model}", file=sys.stderr)
        memory.close()
        return 2

    speaker = None
    if not no_speak:
        try:
            speaker = make_speaker(
                cfg.tts_engine, cfg.tts_rate, cfg.tts_voice,
                cfg.piper_binary, cfg.piper_model,
            )
        except TTSUnavailable as exc:
            print(f"  speech unavailable: {exc}", file=sys.stderr)
            speaker = None
    else:
        from .tts import SilentSpeaker

        speaker = SilentSpeaker()

    assistant = Assistant(cfg, memory, brain, speaker)
    assistant.maintain()
    print(f"  vision: {assistant.vision_status()}", file=sys.stderr)
    # Load the model and prefill the stable prefix in the background. A cold
    # first turn otherwise pays ~2.4 GB of weights plus ~92s of tool-schema
    # prefill, and neither depends on the question. Backgrounded so startup
    # itself stays about a second; the first turn waits on it at worst.
    if not hud:
        assistant.warm(on_done=lambda note: print(f"  warm: {note}", file=sys.stderr))

    # Everything past this point can raise or be interrupted, and the visual
    # memory holds an open SQLite connection that should not be leaked.
    try:
        if hud:
            # The display reports the warm into its own ring, so it is left to
            # start it rather than being warmed twice from here.
            from .hud import run_hud

            return run_hud(assistant, speaker, echo=not no_speak, initial=prompt)

        if prompt:
            if text_mode:
                from .cli import TextLoop

                loop = TextLoop(assistant, speaker, echo=not no_speak)
                loop._turn(prompt)  # noqa: SLF001 - deliberate single-shot use
            else:
                assistant.say(assistant.respond(prompt).spoken)
            return 0

        if text_mode:
            from .cli import TextLoop

            TextLoop(assistant, speaker, echo=not no_speak).run()
            return 0

        if force_voice or sys.stdin.isatty() or not cfg.wake_enabled:
            return _run_voice(cfg, assistant, speaker)

        from .cli import TextLoop

        TextLoop(assistant, speaker, echo=not no_speak).run()
        return 0
    finally:
        assistant.close()
        memory.close()


def _run_voice(cfg, assistant, speaker) -> int:
    from .audio import AudioUnavailable, Mic
    from .stt import Listener, STTUnavailable
    from .voice import VoiceLoop
    from .wake import WakeWord, WakeWordUnavailable

    try:
        listener = Listener(
            cfg.stt_model, cfg.stt_device, cfg.stt_compute, cfg.stt_beam_size
        )
    except STTUnavailable as exc:
        print(f"  speech recognition unavailable: {exc}", file=sys.stderr)
        print("  falling back to text mode.\n", file=sys.stderr)
        from .cli import TextLoop

        TextLoop(assistant, speaker).run()
        return 0

    mic = Mic(
        sample_rate=cfg.sample_rate,
        device=cfg.audio_device,
        vad_threshold=cfg.vad_threshold,
        vad_silence_ms=cfg.vad_silence_ms,
        vad_max_utterance_s=cfg.vad_max_utterance_s,
        vad_preroll_ms=cfg.vad_preroll_ms,
    )
    wake = WakeWord(cfg.wake_model, cfg.wake_threshold, cfg.sample_rate) if cfg.wake_enabled else None

    loop = VoiceLoop(assistant, cfg, listener, mic, wake, speaker)
    try:
        loop.run()
    except AudioUnavailable as exc:
        print(f"\n  microphone unavailable: {exc}\n", file=sys.stderr)
        print("  Try --list-audio to see the devices, then --device <index>.\n",
              file=sys.stderr)
        return 3
    return 0


def _list_tools() -> int:
    from .tools import load_all

    registry = load_all()

    by_tag: dict[str, list[str]] = {}
    for tool in registry._tools.values():  # noqa: SLF001
        for tag in tool.tags or ("other",):
            by_tag.setdefault(tag, []).append(tool.name)

    print(f"\n  {len(registry)} tools\n")
    for tag in sorted(by_tag):
        print(f"  {tag}:")
        for name in sorted(by_tag[tag]):
            tool = registry.get(name)
            danger = "  [needs approval]" if tool and tool.dangerous else ""
            print(f"    {name}{danger}")
    print()
    return 0


def _list_audio() -> int:
    try:
        from .audio import list_input_devices
    except ImportError as exc:
        print(f"  sounddevice missing: {exc}")
        return 1
    try:
        devices = list_input_devices()
    except Exception as exc:  # noqa: BLE001
        print(f"  could not query audio: {exc}")
        return 1
    print("\n  input devices\n")
    for d in devices:
        print(f"    [{d['index']}] {d['name']} - {d['channels']}ch @ {d['default_rate']}Hz")
    print("\n  use: jarvis --device <index>\n")
    return 0


def _print_config(cfg) -> None:
    print(f"\n  config file: {Path.home() / '.jarvis' / 'config.json'}")
    for key, value in sorted(cfg.to_dict().items()):
        if key in {"persona"} and isinstance(value, str) and len(value) > 60:
            value = value[:57] + "..."
        print(f"    {key:<22} {value}")
    print()


def _doctor(cfg) -> int:
    """Check every moving part and report what is and is not working."""
    ok = True
    print(f"\n  JARVIS {__version__} - install check\n")

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        if not good:
            ok = False
        mark = "PASS" if good else "FAIL"
        print(f"  [{mark}] {label:<24}{detail}")

    def skip(label: str, detail: str, configured: bool) -> None:
        """Report an optional integration without failing the install check."""
        mark = "PASS" if configured else "SKIP"
        print(f"  [{mark}] {label:<24}{detail if not configured else 'configured'}")

    print("  python")
    check("version", sys.version_info >= (3, 10),
          f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    print("\n  desktop automation")
    import importlib

    from .env import GUARDED, describe, shadowed_modules

    # Shims sitting next to the project are expected: env.py exists to make
    # sure they lose. So the thing worth asserting is that the real package
    # won, not that no shim file exists.
    shadowed = shadowed_modules()
    hijacked = []
    for name in GUARDED:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except ImportError:
                continue
        origin = getattr(module, "__file__", "") or ""
        if not origin or not os.path.isfile(origin):
            continue
        if "site-packages" not in os.path.realpath(origin).replace("\\", "/"):
            hijacked.append(f"{name} -> {origin}")
    detail = ", ".join(hijacked) if hijacked else (
        f"bypassed {len(shadowed)} local shim(s)" if shadowed else "no local shims"
    )
    check("real packages win", not hijacked, detail)
    check("pyautogui", True, describe().split(";")[0].replace("pyautogui -> ", ""))

    print("\n  brain")
    models = available_models(cfg.ollama_host)
    check("ollama reachable", bool(models),
          f"{len(models)} models" if models else f"nothing at {cfg.ollama_host}")
    if models:
        chosen, note = pick_model(cfg)
        check("tool-capable model", bool(chosen), f"{chosen} ({note})")
        print(f"         installed: {', '.join(models[:8])}")

    print("\n  memory")
    try:
        from .memory import Memory

        mem = Memory(cfg.resolved_db_path)
        stats = mem.stats()
        mem.close()
        check("database", True,
              f"{cfg.resolved_db_path} "
              f"({stats['messages']} msgs, {stats['facts']} facts, {stats['notes']} notes)")
    except Exception as exc:  # noqa: BLE001
        check("database", False, str(exc))

    print("\n  ears")
    try:
        from .audio import default_input_device, list_input_devices, mic_rank

        devices = list_input_devices()
        chosen = default_input_device(cfg.audio_device)
        if devices:
            name = next(
                (d["name"] for d in devices if d["index"] == chosen), f"index {chosen}"
            )
            detail = f"{len(devices)} found, using {chosen}: {name}"
        else:
            detail = f"no input devices (resolved {chosen})"
        check("microphone", bool(devices), detail)
        if devices and mic_rank(name) > 0:
            print(
                f"  [WARN] {'microphone' if cfg.audio_device >= 0 else 'system default'}"
                f" looks like a virtual/loopback device, so it may capture nothing useful."
                f"\n         Run --list-audio and pass --device <index> for a real mic."
            )
    except Exception as exc:  # noqa: BLE001
        check("microphone", False, str(exc))

    try:
        from .stt import _get_model

        _get_model(cfg.stt_model, cfg.stt_device, cfg.stt_compute)
        check(f"whisper '{cfg.stt_model}'", True, "loaded")
    except Exception as exc:  # noqa: BLE001
        check(f"whisper '{cfg.stt_model}'", False, str(exc)[:90])

    print("\n  wake word")
    if not cfg.wake_enabled:
        check("hotword", True, "disabled by config")
    else:
        try:
            from .wake import WakeWord

            WakeWord(cfg.wake_model, cfg.wake_threshold).load()
            check(f"'{cfg.wake_model}'", True, "loaded")
        except Exception as exc:  # noqa: BLE001
            check(f"'{cfg.wake_model}'", False, str(exc)[:90])

    print("\n  mouth")
    try:
        from .tts import make_speaker

        speaker = make_speaker(cfg.tts_engine, cfg.tts_rate, cfg.tts_voice,
                               cfg.piper_binary, cfg.piper_model)
        check("speech engine", True, getattr(speaker, "name", "?"))
    except Exception as exc:  # noqa: BLE001
        check("speech engine", False, str(exc)[:90])

    print("\n  screen")
    try:
        import pytesseract

        check("ocr", True, f"tesseract {pytesseract.get_tesseract_version()}")
    except Exception as exc:  # noqa: BLE001
        check("ocr", False, str(exc)[:90])
    try:
        from .env import DPI_AWARENESS, virtual_desktop
        from .tools.screen_tools import list_displays

        real = [m for m in list_displays()["monitors"] if m["index"] != 0]
        desk = virtual_desktop()
        extent = f"desktop {desk[0]}x{desk[1]}" if desk else "desktop unknown"
        check("displays", True, f"{len(real)} attached, {extent}, dpi {DPI_AWARENESS}")
    except Exception as exc:  # noqa: BLE001
        check("displays", False, str(exc)[:90])
    try:
        from .tools.screen_tools import take_screenshot

        shot = take_screenshot()
        check("screenshot", True, f"{shot['width']}x{shot['height']} -> {shot['path']}")
    except Exception as exc:  # noqa: BLE001
        check("screenshot", False, str(exc)[:90])

    print("\n  seeing")
    try:
        from .perception import Perception

        obs = Perception(thumb_width=cfg.vision_thumb_width).observe(force=True)
        check("perception", True,
              f"{len(obs.elements)} elements, window={obs.window_title[:40]!r}")
    except Exception as exc:  # noqa: BLE001
        check("perception", False, str(exc)[:90])

    try:
        from .vlm import Vision

        vision = Vision(cfg)
        check(f"vision '{cfg.vlm_model}'", vision.available(),
              "installed, used for pointing"
              if vision.available()
              else "not pulled; run: ollama pull " + cfg.vlm_model)
        # The describing model is a separate install and does the work that
        # matters most, so reporting only the pointer hid a missing 3 GB
        # download behind a green line.
        if vision.describe_model != cfg.vlm_model:
            check(f"describe '{vision.describe_model}'", vision.describe_available(),
                  f"installed, used for describing at {vision.describe_max_side}px"
                  if vision.describe_available()
                  else "not pulled; run: ollama pull " + vision.describe_model)
    except Exception as exc:  # noqa: BLE001
        check(f"vision '{cfg.vlm_model}'", False, str(exc)[:90])

    try:
        from .embeddings import make_embedder

        embedder = make_embedder(cfg)
        check("embeddings", True, f"{embedder.name}, {embedder.dim} dims")
    except Exception as exc:  # noqa: BLE001
        check("embeddings", False, str(exc)[:90])

    try:
        from .autonomy import Autonomy

        autonomy = Autonomy(cfg)
        trusted = len([k for k, v in autonomy.known_apps().items() if v != "DENIED"])
        check("autonomy", True, f"{trusted} apps trusted, {len(autonomy.deny)} denied")
    except Exception as exc:  # noqa: BLE001
        check("autonomy", False, str(exc)[:90])

    # Both are genuinely optional, so an unconfigured one is a skip, not a
    # failure. Counting them as FAIL made a healthy install look broken.
    print("\n  home automation")
    skip("home assistant", "not set (optional)", bool(cfg.home_assistant_url))
    skip("mqtt", "not set (optional)", bool(cfg.mqtt_host))

    print("\n  tools")
    from .tools import load_all

    check("registered", len(load_all()) > 0, f"{len(load_all())} tools")

    print(f"\n  {'all good' if ok else 'some checks failed - see FAIL above'}")
    print("  Start with: ollama serve   (in another window)\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
