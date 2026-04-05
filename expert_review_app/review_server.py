#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm_client_factory import build_chat_client
from backend.llm_clients import LLMError
from backend.runtime_config import LLMConfig

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "review_samples_assigned_12x20.json"
REVIEWS_DIR = ROOT / "data" / "reviews"
FRONTEND_DIR = ROOT / "frontend"
TRANSLATION_CACHE_PATH = ROOT / "data" / "translation_cache.json"

BLIND_FIELDS_10 = [
    "expert_harm_1_10",
    "expert_negative_emotion_1_10",
    "expert_inappropriate_1_10",
    "expert_empathic_language_1_10",
]
BLIND_FIELDS_5 = [
    "expert_anthro_q1",
    "expert_anthro_q2",
    "expert_anthro_q3",
    "expert_anthro_q4",
    "expert_anthro_q5",
]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sanitize_reviewer(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip())
    return cleaned[:80].strip("._-")


class TranslationStore:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}

        self.client = build_chat_client(
            LLMConfig(
                provider=os.environ.get("EXPERT_REVIEW_TRANSLATOR_PROVIDER", "deepseek"),
                model=os.environ.get("EXPERT_REVIEW_TRANSLATOR_MODEL", "deepseek-chat"),
                api_key_env=os.environ.get("EXPERT_REVIEW_TRANSLATOR_API_KEY_ENV", "DEEPSEEK_API_KEY"),
                base_url=os.environ.get("EXPERT_REVIEW_TRANSLATOR_BASE_URL", "https://api.deepseek.com/v1"),
                temperature=0.0,
                top_p=1.0,
                seed=None,
            )
        )

    def _cache_key(self, sample_id: str, field_name: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{sample_id}:{field_name}:{digest}"

    def _save(self) -> None:
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _translate_text(self, text: str) -> str:
        if not text.strip():
            return ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise translator. Translate the user's input into natural, fluent Simplified Chinese. "
                    "Preserve meaning, tone, formatting, and line breaks. Do not summarize. Do not explain. "
                    "Output only the Chinese translation."
                ),
            },
            {"role": "user", "content": text},
        ]
        result = self.client.chat(messages, timeout_seconds=120)
        return result.text.strip()

    def translate_sample(self, sample_id: str, user_text: str, model_reply: str) -> dict:
        output = {}
        cache_miss = False
        for field_name, text in (("user_text_zh", user_text), ("model_reply_zh", model_reply)):
            key = self._cache_key(sample_id, field_name, text)
            cached = self.cache.get(key)
            if cached is None:
                translated = self._translate_text(text)
                self.cache[key] = translated
                output[field_name] = translated
                cache_miss = True
            else:
                output[field_name] = cached
        if cache_miss:
            self._save()
        return output


class ReviewStore:
    def __init__(self, data_path: Path, reviews_dir: Path) -> None:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        self.samples = payload["samples"]
        self.sample_map = {item["sample_id"]: item for item in self.samples}
        self.assignments = payload.get("assignments", {})
        self.reviewer_meta = payload.get("reviewer_meta", {})
        self.reviews_dir = reviews_dir
        self.reviews_dir.mkdir(parents=True, exist_ok=True)

    def assigned_samples(self, reviewer: str) -> list[dict]:
        if self.assignments:
            safe = sanitize_reviewer(reviewer)
            if safe not in self.assignments:
                raise ValueError("unknown reviewer account")
            allowed = set(self.assignments[safe])
            return [sample for sample in self.samples if sample["sample_id"] in allowed]
        return self.samples

    def review_path(self, reviewer: str) -> Path:
        safe = sanitize_reviewer(reviewer)
        if not safe:
            raise ValueError("reviewer is required")
        return self.reviews_dir / f"{safe}.json"

    def load_review_state(self, reviewer: str) -> dict:
        path = self.review_path(reviewer)
        if not path.exists():
            return {
                "reviewer": reviewer,
                "created_at": now_utc(),
                "updated_at": now_utc(),
                "reviews": {},
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def save_review_state(self, reviewer: str, state: dict) -> None:
        state["updated_at"] = now_utc()
        self.review_path(reviewer).write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def build_session_summary(self, reviewer: str) -> dict:
        assigned_samples = self.assigned_samples(reviewer)
        state = self.load_review_state(reviewer)
        reviews = state.get("reviews", {})
        completed = sum(1 for item in reviews.values() if item.get("blind_submitted"))
        next_sample_id = None
        for sample in assigned_samples:
            status = reviews.get(sample["sample_id"], {})
            if not status.get("blind_submitted"):
                next_sample_id = sample["sample_id"]
                break
        return {
            "reviewer": reviewer,
            "sample_count": len(assigned_samples),
            "completed_count": completed,
            "next_sample_id": next_sample_id,
            "reviewer_meta": self.reviewer_meta.get(sanitize_reviewer(reviewer), {}),
            "sample_status": [
                {
                    "sample_id": sample["sample_id"],
                    "domain": sample["domain"],
                    "condition": sample["condition"],
                    "phase": sample["phase"],
                    "blind_submitted": bool(reviews.get(sample["sample_id"], {}).get("blind_submitted")),
                }
                for sample in assigned_samples
            ],
        }

    def get_sample_payload(self, reviewer: str, sample_id: str) -> dict:
        assigned = {sample["sample_id"] for sample in self.assigned_samples(reviewer)}
        if sample_id not in assigned:
            raise ValueError("sample not assigned to this reviewer")
        state = self.load_review_state(reviewer)
        review = state.get("reviews", {}).get(sample_id, {})
        sample = self.sample_map[sample_id]
        payload = {
            key: sample[key]
            for key in [
                "sample_id",
                "domain",
                "condition",
                "phase",
                "dialogue_id",
                "turn_index",
                "user_text",
                "model_reply",
            ]
        }
        payload["review_state"] = {
            "blind_submitted": bool(review.get("blind_submitted")),
            "blind_scores": review.get("blind_scores"),
        }
        return payload

    def submit_blind(self, reviewer: str, sample_id: str, blind_scores: dict, blind_notes: str) -> dict:
        state = self.load_review_state(reviewer)
        reviews = state.setdefault("reviews", {})
        review = reviews.setdefault(sample_id, {})
        review["blind_submitted"] = True
        review["blind_submitted_at"] = now_utc()
        review["blind_scores"] = blind_scores
        review["blind_notes"] = blind_notes
        review["final_submitted"] = True
        review["final_submitted_at"] = review["blind_submitted_at"]
        self.save_review_state(reviewer, state)
        return self.get_sample_payload(reviewer, sample_id)

    def export_reviews(self, reviewer: str) -> dict:
        assigned_samples = self.assigned_samples(reviewer)
        state = self.load_review_state(reviewer)
        rows = []
        for sample in assigned_samples:
            review = state.get("reviews", {}).get(sample["sample_id"], {})
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "domain": sample["domain"],
                    "condition": sample["condition"],
                    "phase": sample["phase"],
                    "dialogue_id": sample["dialogue_id"],
                    "turn_index": sample["turn_index"],
                    "user_text": sample["user_text"],
                    "model_reply": sample["model_reply"],
                    "blind_submitted": bool(review.get("blind_submitted")),
                    "blind_scores": review.get("blind_scores"),
                    "blind_notes": review.get("blind_notes"),
                }
            )
        return {
            "reviewer": reviewer,
            "exported_at": now_utc(),
            "rows": rows,
        }


STORE: ReviewStore | None = None
TRANSLATIONS: TranslationStore | None = None


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def _json_response(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw or "{}")

    def _require_store(self) -> ReviewStore:
        if STORE is None:
            raise RuntimeError("store not initialized")
        return STORE

    def _require_translations(self) -> TranslationStore:
        if TRANSLATIONS is None:
            raise RuntimeError("translations not initialized")
        return TRANSLATIONS

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self._json_response({"ok": True})
        if parsed.path == "/api/session":
            reviewer = parse_qs(parsed.query).get("reviewer", [""])[0]
            try:
                payload = self._require_store().build_session_summary(reviewer)
            except ValueError as exc:
                return self._json_response({"error": str(exc)}, status=400)
            return self._json_response(payload)
        if parsed.path == "/api/sample":
            qs = parse_qs(parsed.query)
            reviewer = qs.get("reviewer", [""])[0]
            sample_id = qs.get("sample_id", [""])[0]
            try:
                payload = self._require_store().get_sample_payload(reviewer, sample_id)
            except Exception as exc:  # noqa: BLE001
                return self._json_response({"error": str(exc)}, status=400)
            return self._json_response(payload)
        if parsed.path == "/api/export":
            reviewer = parse_qs(parsed.query).get("reviewer", [""])[0]
            try:
                payload = self._require_store().export_reviews(reviewer)
            except ValueError as exc:
                return self._json_response({"error": str(exc)}, status=400)
            return self._json_response(payload)
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except Exception as exc:  # noqa: BLE001
            return self._json_response({"error": f"invalid json: {exc}"}, status=400)

        if parsed.path == "/api/review/blind":
            reviewer = str(payload.get("reviewer", ""))
            sample_id = str(payload.get("sample_id", ""))
            blind_scores = payload.get("blind_scores") or {}
            blind_notes = str(payload.get("blind_notes", ""))
            try:
                for field in BLIND_FIELDS_10:
                    value = int(blind_scores[field])
                    if value < 1 or value > 10:
                        raise ValueError(f"{field} out of range")
                for field in BLIND_FIELDS_5:
                    value = int(blind_scores[field])
                    if value < 1 or value > 5:
                        raise ValueError(f"{field} out of range")
                result = self._require_store().submit_blind(reviewer, sample_id, blind_scores, blind_notes)
            except Exception as exc:  # noqa: BLE001
                return self._json_response({"error": str(exc)}, status=400)
            return self._json_response(result)

        if parsed.path == "/api/translate":
            reviewer = str(payload.get("reviewer", ""))
            sample_id = str(payload.get("sample_id", ""))
            try:
                sample = self._require_store().get_sample_payload(reviewer, sample_id)
                translations = self._require_translations().translate_sample(
                    sample_id=sample["sample_id"],
                    user_text=sample["user_text"],
                    model_reply=sample["model_reply"],
                )
            except (ValueError, LLMError, RuntimeError) as exc:
                return self._json_response({"error": str(exc)}, status=400)
            except Exception as exc:  # noqa: BLE001
                return self._json_response({"error": str(exc)}, status=500)
            return self._json_response({"sample_id": sample_id, **translations})

        return self._json_response({"error": "not found"}, status=404)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the expert review app.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8787, help="Bind port.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Sample dataset JSON.")
    return parser.parse_args()


def main() -> int:
    global STORE, TRANSLATIONS
    args = parse_args()
    STORE = ReviewStore(Path(args.data), REVIEWS_DIR)
    TRANSLATIONS = TranslationStore(TRANSLATION_CACHE_PATH)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Expert review app serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
