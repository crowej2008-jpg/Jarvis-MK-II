"""The vision model, used only when OCR is not enough.

moondream is small enough to run on CPU and, unlike most vision models, it can
*point* at a described object and return coordinates. That is what this module
exists for: OCR gives boxes for text, and this gives coordinates for icons,
images, and buttons that carry no readable text.

Coordinates coming back from a vision model are estimates. They are used as a
starting point and then verified against a screen diff, never trusted blindly.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

log = logging.getLogger("jarvis.vlm")

# moondream answers with a bracketed list of coordinates. It is usually a
# normalised bounding box [x1, y1, x2, y2] in 0..1, but a bare [x, y] point and
# pixel-space answers both show up depending on the prompt, so all three are
# accepted rather than assumed.
_BRACKETED = re.compile(r"[\[\(]\s*([\d.\-]+(?:\s*,\s*[\d.\-]+){1,3})\s*[\]\)]")
_POINT_PATTERNS = (
    re.compile(r"<\s*point\s*>\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"),
    re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"),
    re.compile(r"x\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*,?\s*y\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.I),
)


@dataclass
class Fix:
    """A located region, in real screen coordinates."""

    x: int
    y: int
    confidence: float = 0.5
    source: str = "vlm"
    note: str = ""
    box: tuple[int, int, int, int] | None = None

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0] if self.box else 0

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1] if self.box else 0

    def as_dict(self) -> dict[str, Any]:
        out = {
            "x": self.x,
            "y": self.y,
            "confidence": round(self.confidence, 2),
            "source": self.source,
            "note": self.note,
        }
        if self.box:
            out["box"] = list(self.box)
            out["size"] = [self.width, self.height]
        return out


class VLMUnavailable(RuntimeError):
    pass


class Vision:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.vlm_model
        # Describing and pointing want different things from a model, so they
        # can be served by different ones. Falls back to the pointer model.
        self.describe_model = (
            getattr(cfg, "vlm_describe_model", "") or self.model
        )
        self.host = cfg.ollama_host.rstrip("/")
        self.timeout = float(getattr(cfg, "vlm_timeout", 120.0))
        self.max_side = int(getattr(cfg, "vlm_max_side", 896))
        # Describing sends a smaller image than pointing, but the size is close
        # to irrelevant to latency: measured with qwen2.5vl on a real 1920x1200
        # window, a describe call reported 1099 prompt tokens at 96px, 128px,
        # 168px, 224px, 336px, 448px, 560px, 672px, 896px and 1120px alike,
        # because the model pads whatever it gets to a fixed token grid.
        # Cropping does not help either, for the same reason. So the small size
        # is kept only to keep the payload small. Pointing keeps full
        # resolution, where detail matters.
        #
        # That cost is paid once per distinct image, not once per question.
        # See _describe_cache: re-asking identical bytes costs 0.5ms per token
        # rather than 78ms, so an unchanged screen is cheap after the first ask.
        self.describe_max_side = int(getattr(cfg, "vlm_describe_max_side", 448) or 448)
        self.num_predict = int(getattr(cfg, "vlm_num_predict", 256) or 256)
        self.describe_num_predict = int(
            getattr(cfg, "vlm_describe_num_predict", 96) or 96
        )
        self.describe_budget = float(
            getattr(cfg, "vlm_describe_budget_seconds", 150.0) or 0.0
        )
        self._client = None
        self._installed_cache: dict[str, bool] = {}
        # Descriptions of screens already described, keyed on the image bytes.
        #
        # The cost of describing is paid once per *distinct* image, not once per
        # question, and the measurement was unambiguous. Same screenshot, three
        # questions: 71.74s, then 6.28s, then 3.11s. Change one region by 40
        # levels and it goes back to 91.66s. The first call prefills ~1100 image
        # tokens at ~78ms each; every later call on identical bytes prefills at
        # 0.5ms/token, because ollama caches the vision encoder result against
        # the image.
        #
        # So an unchanged screen is already cheap to re-ask, but only inside one
        # process, and only while the model is still loaded. Caching it here
        # makes a repeat description free and independent of both, which is the
        # difference between "the screen is the same as last time" costing 85
        # seconds and costing nothing.
        self._describe_cache: "OrderedDict[str, str]" = OrderedDict()
        # Question text is part of the key: the same screen asked a different
        # question is a different answer, and answering the wrong question
        # quickly would be worse than answering slowly. 0 disables the cache,
        # which is the honest setting when a screen is expected to change
        # between every question anyway.
        self._describe_cache_max = max(
            0, int(getattr(cfg, "vlm_describe_cache_entries", 24) or 0)
        )

    def _image_key(self, image: np.ndarray, question: str, model: str) -> str:
        """A cache key over the exact bytes the model would be sent.

        Deliberately hashes the encoded payload rather than the array: the
        bytes are what ollama's own cache keys on, so this cache and the
        server's agree about what "the same image" means. That means a hit here
        predicts a fast call there, rather than the two disagreeing.
        """
        payload, _scale, _size = self._prepare(image, self.describe_max_side)
        digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
        return f"{model}|{question.strip()}|{digest}"

    def _describe_cached(
        self, key: str, produce: Callable[[], str]
    ) -> tuple[str, bool]:
        """Return (answer, was_hit) for `key`, computing it if absent.

        Only a usable answer is stored. The usability check lives here rather
        than in the callers so that a confabulation cannot be cached: `_ask`
        returns text that is frequently junk, and caching that would leave the
        screen permanently undescribable, because every later call would be a hit
        on the junk and the miss would never be retried.
        """
        if self._describe_cache_max <= 0:
            return produce(), False
        hit = self._describe_cache.get(key)
        if hit is not None:
            self._describe_cache.move_to_end(key)
            log.info("describe cache hit for %r", key[:12])
            return hit, True
        answer = produce()
        if answer and self._usable(answer):
            self._describe_cache[key] = answer
            # Bounded, and least-recently-used evicted. A long session on a
            # screen that changes constantly should not grow without limit.
            while len(self._describe_cache) > self._describe_cache_max:
                self._describe_cache.popitem(last=False)
        elif answer:
            log.info("describe answer for %r unusable, not caching it", key[:12])
        return answer, False

    # -- plumbing --------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host, timeout=self.timeout)
        return self._client

    def available(self) -> bool:
        """Is the pointing model installed?"""
        return self._installed(self.model)

    def describe_available(self) -> bool:
        """Is the description model installed?"""
        return self._installed(self.describe_model)

    def _installed(self, model: str) -> bool:
        cache = self._installed_cache
        if model in cache:
            return cache[model]
        ok = False
        try:
            import requests

            resp = requests.get(f"{self.host}/api/tags", timeout=3.0)
            resp.raise_for_status()
            names = {m.get("name", "").split(":")[0] for m in resp.json().get("models", [])}
            ok = model.split(":")[0] in names
        except Exception:  # noqa: BLE001
            ok = False
        cache[model] = ok
        if not ok:
            log.info("vision model %s not available", model)
        return ok

    def _prepare(
        self, image: np.ndarray, max_side: int | None = None
    ) -> tuple[str, float, tuple[int, int]]:
        """Downscale for the model. Returns (base64 png, scale, original size)."""
        import cv2

        side = int(max_side or self.max_side)
        height, width = image.shape[:2]
        longest = max(height, width)
        scale = 1.0
        if longest > side:
            scale = side / longest
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise VLMUnavailable("could not encode the screenshot for the vision model")
        return base64.b64encode(buf.tobytes()).decode("ascii"), scale, (width, height)

    def _ask(
        self,
        image: np.ndarray,
        prompt: str,
        model: str = "",
        max_side: int | None = None,
        num_predict: int | None = None,
    ) -> str:
        model = model or self.model
        if not self._installed(model):
            raise VLMUnavailable(
                f"the vision model {model!r} is not installed. "
                f"Run: ollama pull {model}"
            )
        payload, _scale, _size = self._prepare(image, max_side)
        cap = self.num_predict if num_predict is None else int(num_predict)
        try:
            response = self._get_client().generate(
                model=model,
                prompt=prompt,
                images=[payload],
                options={"temperature": 0.0, "num_predict": cap},
                keep_alive="5m",
            )
        except Exception as exc:  # noqa: BLE001
            raise VLMUnavailable(f"the vision model failed: {exc}") from exc
        return (response.get("response") or "").strip()

    # -- capabilities ----------------------------------------------------
    # Moondream answers concrete questions well and rambles when asked to
    # "describe" a whole screen unprompted. Empirically, one narrow question
    # per call beats one broad one, so these are deliberately narrow.
    #
    # Order matters. The first is the easiest, and on a screen this model
    # cannot actually read, every one of these comes back as a confabulation.
    # Asking three questions then costs three times the wait for three times
    # the nothing, so the first unusable answer ends the sequence.
    _DETAIL_PROMPTS = (
        "What application is shown? Answer with just the name.",
        "List the text and buttons you can read, one per line. No commentary.",
        "What is the most prominent thing on screen? One short sentence.",
    )

    # Used when OCR already read the screen. Asking a model to re-read text
    # that Tesseract has already extracted is strictly worse than useless: at
    # this scale the glyphs do not resolve, so the model invents something
    # fluent. Measured on a real 1920x1200 window, moondream answered
    # "1. Read the instructions." and a later pass answered
    # "urn:jira:issue/uid:...". The OCR text already has the real content, so
    # the model's job here is only what OCR cannot see: which application this
    # is, and what the window is doing.
    #
    # One question, not three. A screen-reading model costs 30-70s per call on
    # this CPU, so three passes would make the user wait minutes for one
    # description.
    _SINGLE_PROMPT = (
        "What is this window, and what is it showing? "
        "Answer in one short sentence. Do not list text you can read."
    )

    def describe(
        self,
        image: np.ndarray,
        question: str = "",
        ocr_text: str = "",
    ) -> str:
        """A plain-language description of what is on screen.

        With a question, it is passed straight through. Without one, this asks
        the model a few narrow questions and stitches the answers together,
        because the model reliably degrades into noise on a single broad
        "describe this screen" instruction.

        `ocr_text` is the text already extracted from this same screenshot. When
        it is substantial, the model is asked one question about the window
        rather than being asked to transcribe text OCR has already done
        correctly.

        Returns "" when the model produced nothing usable. Callers must treat
        that as "no visual information", never as an empty description.
        """
        if question.strip():
            key = self._image_key(image, question, self.describe_model)
            answer, _hit = self._describe_cached(key, lambda: self._ask(
                image, self._question_prompt(question),
                model=self.describe_model,
                max_side=self.describe_max_side,
                num_predict=self.describe_num_predict,
            ))
            return answer if self._usable(answer) else ""

        deadline = time.monotonic() + self.describe_budget

        if self._ocr_is_rich(ocr_text):
            key = self._image_key(image, self._SINGLE_PROMPT, self.describe_model)
            try:
                answer, _hit = self._describe_cached(key, lambda: self._ask(
                    image, self._SINGLE_PROMPT,
                    model=self.describe_model,
                    max_side=self.describe_max_side,
                    num_predict=self.describe_num_predict,
                ))
            except VLMUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001
                log.info("describe pass failed: %s", exc)
                return ""
            return answer if self._usable(answer) else ""

        key = self._image_key(image, "|".join(self._DETAIL_PROMPTS),
                              self.describe_model)
        answer, _hit = self._describe_cached(
            key, lambda: self._multi_pass(image, deadline)
        )
        return answer

    def _multi_pass(self, image: np.ndarray, deadline: float) -> str:
        """The three-question path, factored out so it caches as one unit.

        Only the *first* question is expensive. Measured on a real screen with
        qwen2.5vl: 71.74s, then 6.28s, then 3.11s against the same image bytes,
        because the first call is what pays the ~1100 image tokens at ~78ms each
        and ollama keeps the vision-encoder result for the rest. Change one
        region by 40 levels and it is back to 91.66s.

        So the cost is per *image*, not per question. The budget below therefore
        protects against a genuinely cold image or a model that is slow for some
        other reason, not against three questions each costing 80-90s as the
        old comment here claimed.
        """
        parts: list[str] = []
        for prompt in self._DETAIL_PROMPTS:
            # The budget turns "wait longer" into "answer with less", which is
            # the right way round: a partial description arrives, instead of
            # nothing arriving eventually.
            if time.monotonic() > deadline:
                log.info("describe gave up after %d of %d questions: "
                         "budget of %.0fs spent",
                         len(parts), len(self._DETAIL_PROMPTS),
                         self.describe_budget)
                break
            try:
                answer = self._ask(image, prompt, model=self.describe_model,
                                   max_side=self.describe_max_side,
                                   num_predict=self.describe_num_predict)
            except VLMUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001
                log.info("describe pass failed: %s", exc)
                break
            if not self._usable(answer):
                log.info(
                    "describe pass unusable (%d of %d asked); stopping early",
                    len(parts) + 1,
                    len(self._DETAIL_PROMPTS),
                )
                break
            parts.append(answer)
        return "\n".join(f"- {p}" for p in parts)

    @staticmethod
    def _ocr_is_rich(ocr_text: str) -> bool:
        """Is there already enough text on screen that OCR is the reader?

        Roughly a dozen real words. Below that the screen may be an image, a
        diagram, or a game, which is exactly the case the vision model exists
        to handle, so the threshold is deliberately low.
        """
        return len(Vision.content_words(ocr_text)) >= 12

    @staticmethod
    def _question_prompt(question: str) -> str:
        return (
            f"Answer this question about the screen in one or two sentences, "
            f"using only what is visible: {question}"
        )

    # Words that carry no evidence: they appear in plausible filler and in
    # almost any OCR result, so they cannot corroborate anything.
    _STOPWORDS = frozenset(
        """a an the and or but of to in on at for with from by is are was were be
        been being this that these those it its as if then than so such there here
        what which who whom whose when where why how do does did done not no yes
        you your we our they their he she his her i me my can could should would
        will shall may might must have has had about into over under again once
        all any both each few more most other some only own same too very just
        up down out off between through during before after above below""".split()
    )

    @staticmethod
    def content_words(text: str) -> set[str]:
        """Lowercase words that could plausibly appear on screen."""
        words = re.findall(r"[a-z0-9']+", (text or "").lower())
        return {w for w in words if len(w) > 2 and w not in Vision._STOPWORDS}

    @classmethod
    def supported_by(cls, answer: str, ocr_text: str) -> bool:
        """Does the OCR text back up what the model just claimed?

        A tiny vision model handed a text-dense screenshot does not fail by
        going quiet, it fails by inventing something plausible: on a real
        OpenCode window it answered "1. Read the instructions." That reads
        like a real observation, so the degenerate-output filter lets it
        through and the chat model repeats it as fact. Checking the answer
        against text that is genuinely on screen catches exactly that, with
        no extra model call.

        Returns False only when the answer is a *reading* claim that the screen
        does not support. Purely visual statements ("a code editor is open")
        legitimately share no words with the OCR text, so callers that ask
        about appearance should not treat this as a verdict on the picture.
        """
        claimed = cls.content_words(answer)
        if not claimed:
            # Nothing checkable, e.g. "Firefox." - leave it to _usable.
            return True
        on_screen = cls.content_words(ocr_text)
        if not on_screen:
            # No OCR to check against, so nothing is contradicted. Saying the
            # screen was empty would be a claim this function cannot support.
            return True
        return bool(claimed & on_screen)

    @staticmethod
    def _is_periodic(text: str, min_repeats: int = 3) -> bool:
        """True if the end of `text` is one short chunk repeated.

        The tail is checked rather than the whole string because these loops
        often start after a one-off prefix, e.g. "urn:jira:issue/uid:" followed
        by "jarvis:issue/uid:" over and over, where the period does not divide
        the string from index 0.
        """
        length = len(text)
        for period in range(1, length // min_repeats + 1):
            if text[-period:] * min_repeats == text[-period * min_repeats:]:
                return True
        return False

    @staticmethod
    def _usable(answer: str, want_prose: bool = True) -> bool:
        """Reject the degenerate outputs a tiny VLM falls into.

        Observed failures on a real text-dense window: a repeating token loop
        with no spaces at all, a coordinate list in reply to a request for
        prose, and a hallucinated URI. All three would reach the chat model
        dressed up as observations, so they are caught here rather than trusted.
        """
        text = (answer or "").strip()
        if len(text) < 3:
            return False

        # Character-level loop. Word counting misses these, because a run-on
        # loop like "urn:jira:issue/uid:urn:jira:issue/uid:" is a single token.
        if len(text) >= 24 and Vision._is_periodic(text):
            return False

        if want_prose:
            # A bare coordinate list, id or number is not an answer to a
            # question. Single words are fine, so this tests for letters rather
            # than for spaces.
            if not any(c.isalpha() for c in text.strip(".,:;!?()[]")):
                return False
            # A URL or URN is a confabulation when it is all the model had to
            # say. It is not one when the screen really does show a link: a
            # verbatim read of a terminal legitimately contains
            # "http://localhost:11434/api/tags" alongside the surrounding
            # commands, and rejecting that threw away a correct answer.
            # So the test is how much is left once the URI is removed.
            if re.search(r"\b(urn|https?|ftp|file|mailto)\s*:", text, re.IGNORECASE):
                residue = re.sub(
                    r"\b(?:urn|https?|ftp|file|mailto)\s*:\s*\S+", " ", text, flags=re.IGNORECASE
                )
                if len(Vision.content_words(residue)) < 4:
                    return False

        letters = [c for c in text if c.isalpha()]
        if letters:
            vowels = sum(c in "aeiouAEIOU" for c in letters)
            if vowels / len(letters) < 0.12:
                return False
        return True

    def point_at(self, image: np.ndarray, target: str) -> Fix | None:
        """Ask the model where `target` is. Returns a real-coordinate estimate."""
        prompt = (
            f"Point to the {target} on this screen. "
            "Answer with only the coordinates as [[x, y]]."
        )
        raw = self._ask(image, prompt)
        return self._parse_point(raw, image.shape[1], image.shape[0], target)

    @staticmethod
    def _parse_point(raw: str, width: int, height: int, target: str) -> Fix | None:
        """Turn a model reply into real screen coordinates.

        Handles the three shapes seen in practice: a normalised box, a
        normalised point, and pixel coordinates. A box is preferred because it
        also tells us how big the target is, which raises confidence.
        """
        if not raw:
            return None

        bracketed = _BRACKETED.search(raw)
        if bracketed:
            try:
                values = [float(v) for v in re.split(r"\s*,\s*", bracketed.group(1))]
            except ValueError:
                values = []
            if len(values) in (2, 4) and all(math.isfinite(v) for v in values):
                fix = Vision._from_values(values, width, height, target, raw)
                if fix is not None:
                    return fix

        for pattern in _POINT_PATTERNS:
            match = pattern.search(raw)
            if not match:
                continue
            try:
                values = [float(match.group(1)), float(match.group(2))]
            except ValueError:
                continue
            fix = Vision._from_values(values, width, height, target, raw)
            if fix is not None:
                return fix

        log.info("could not read coordinates out of: %r", raw[:120])
        return None

    @staticmethod
    def _from_values(
        values: list[float], width: int, height: int, target: str, raw: str
    ) -> Fix | None:
        note = f"{target!r} located by vision model: {raw[:60]}"

        # Normalised 0..1 box: the common case.
        if len(values) == 4 and all(0.0 <= v <= 1.0 for v in values):
            x1, y1, x2, y2 = values
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            left, right = int(x1 * (width - 1)), int(x2 * (width - 1))
            top, bottom = int(y1 * (height - 1)), int(y2 * (height - 1))
            if right - left < 1 or bottom - top < 1:
                return None
            return Fix(
                (left + right) // 2,
                (top + bottom) // 2,
                confidence=0.6,
                note=note,
                box=(left, top, right, bottom),
            )

        # Normalised 0..1 point.
        if len(values) == 2 and all(0.0 <= v <= 1.0 for v in values):
            x = int(values[0] * (width - 1))
            y = int(values[1] * (height - 1))
            return Fix(x, y, confidence=0.45, note=note, box=(x, y, x, y))

        if len(values) == 2:
            x, y = values
            # Already in pixels. Tried before the 0..1000 reading because on a
            # large display those two interpretations are indistinguishable, and
            # moondream emits 0..1, so anything that fits the screen is far
            # more likely to be literal pixel coordinates.
            if 0 <= x < width and 0 <= y < height:
                return Fix(
                    int(x), int(y), confidence=0.5, note=note,
                    box=(int(x), int(y), int(x), int(y)),
                )
            # Legacy 0..1000 normalised space.
            if 1.0 < x <= 1000.0 and 1.0 < y <= 1000.0:
                return Fix(
                    int(x / 1000.0 * (width - 1)),
                    int(y / 1000.0 * (height - 1)),
                    confidence=0.45,
                    note=note,
                )
        return None

    def locate(self, image: np.ndarray, target: str) -> Fix | None:
        """point_at with one retry, since a vision model misses fairly often."""
        first = self.point_at(image, target)
        if first is not None:
            return first
        log.info("retrying locate for %r", target)
        return self.point_at(image, f"the location of the {target}")
