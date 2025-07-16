#!/usr/bin/env python3
"""
Qwen Proxy Concurrency Bench
============================

Concurrent load / soak tester for the Qwen2.5-32B FastAPI service.

See header doc in previous revisions for full description.
"""

from __future__ import annotations

import os
import sys
import json
import csv
import asyncio
import time
import uuid
import random
import statistics
import signal
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    import websockets  # type: ignore
except ImportError:  # optional dependency
    websockets = None  # type: ignore


# ---------------------------------------------------------------------------
# Defaults from env (overridable via CLI)
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "supersecret")
WS_URL = f"{BASE_URL.replace('http', 'ws')}/ws"

DEFAULT_CHAT_WEIGHT = float(os.getenv("CHAT_WEIGHT", "0.4"))
DEFAULT_STREAM_WEIGHT = float(os.getenv("STREAM_WEIGHT", "0.4"))
DEFAULT_SUMMARY_WEIGHT = float(os.getenv("SUMMARY_WEIGHT", "0.2"))

CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "64"))
STREAM_MAX_TOKENS = int(os.getenv("STREAM_MAX_TOKENS", "256"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "300"))

THINK_LOW = float(os.getenv("THINK_LOW", "0.25"))
THINK_HIGH = float(os.getenv("THINK_HIGH", "1.25"))

HEALTH_INTERVAL = float(os.getenv("HEALTH_INTERVAL", "2.0"))
METRICS_INTERVAL = float(os.getenv("METRICS_INTERVAL", "5.0"))

OUT_JSON = os.getenv("OUT_JSON", "bench_results.json")
OUT_CSV = os.getenv("OUT_CSV", "bench_results.csv")


# ---------------------------------------------------------------------------
# Message templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "Tu es un bot de test. Réponds très brièvement."

USER_MSGS = [
    "Bonjour",
    "Peux-tu me dire quelque chose de drôle?",
    "Raconte une anecdote médicale courte",
    "J'ai mal aux dents",
    "Quel est mon prochain rendez-vous?",
    "Merci au revoir",
]

SUMMARY_SYSTEM = (
    'Tu es un système d\'extraction. Retourne JSON strict: {"nom":..., "prenom":..., "resume":...}.'
)

SUMMARY_CONVOS = [
    [
        ("user", "Bonjour j'ai mal aux dents"),
        ("assistant", "Quel est votre nom de famille ?"),
        ("user", "Dubois"),
        ("assistant", "Quel est votre prénom ?"),
        ("user", "Marie"),
    ],
    [
        ("user", "Urgence abcès"),
        ("assistant", "Quel est votre nom ?"),
        ("user", "Durand Pierre"),
    ],
]


# ---------------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------------

@dataclass
class BenchCfg:
    chat_max_tokens: int = CHAT_MAX_TOKENS
    stream_max_tokens: int = STREAM_MAX_TOKENS
    summary_max_tokens: int = SUMMARY_MAX_TOKENS


# ---------------------------------------------------------------------------
# Metrics data structures
# ---------------------------------------------------------------------------

@dataclass
class LatencySample:
    kind: str  # chat|stream|summary|health|metrics
    start: float
    end: float
    ok: bool
    status: Optional[int] = None
    tokens: int = 0  # approx tokens generated (stream)
    bytes: int = 0
    user_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class BenchStats:
    samples: List[LatencySample] = field(default_factory=list)
    metrics_samples: List[Dict[str, Any]] = field(default_factory=list)  # scraped metrics
    start_wall: float = field(default_factory=time.time)
    end_wall: float = 0.0

    def record(self, sample: LatencySample) -> None:
        self.samples.append(sample)

    def record_metrics(self, m: Dict[str, Any]) -> None:
        self.metrics_samples.append(m)

    def close(self) -> None:
        self.end_wall = time.time()

    # --- summary helpers ---
    def _by_kind(self, kind: str) -> List[LatencySample]:
        return [s for s in self.samples if s.kind == kind]

    @staticmethod
    def _pct(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        k = max(0, min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1)))))
        return sorted(values)[k]

    def _lat_summary(self, kind: str) -> Dict[str, Any]:
        data = self._by_kind(kind)
        durs = [s.dur for s in data if s.ok]
        errs = [s for s in data if not s.ok]
        return {
            "count": len(data),
            "errors": len(errs),
            "p50_ms": round(self._pct(durs, 50) * 1000, 1) if durs else None,
            "p90_ms": round(self._pct(durs, 90) * 1000, 1) if durs else None,
            "p95_ms": round(self._pct(durs, 95) * 1000, 1) if durs else None,
            "p99_ms": round(self._pct(durs, 99) * 1000, 1) if durs else None,
        }

    def summary(self) -> Dict[str, Any]:
        total_tokens = sum(s.tokens for s in self.samples if s.tokens)
        wall = (self.end_wall or time.time()) - self.start_wall
        qvals = [
            m.get("fastapi_inference_queue_size")
            for m in self.metrics_samples
            if m.get("fastapi_inference_queue_size") is not None
        ]
        gpu_util = [
            m.get("gpu_utilization_percent")
            for m in self.metrics_samples
            if m.get("gpu_utilization_percent") is not None
        ]
        return {
            "wall_s": round(wall, 2),
            "total_requests": len(self.samples),
            "total_tokens_streamed": total_tokens,
            "tps_effective": round(total_tokens / wall, 2) if wall > 0 else None,
            "chat": self._lat_summary("chat"),
            "stream": self._lat_summary("stream"),
            "summary": self._lat_summary("summary"),
            "health": self._lat_summary("health"),
            "queue_max": max(qvals) if qvals else None,
            "queue_mean": round(sum(qvals) / len(qvals), 2) if qvals else None,
            "gpu_util_max": max(gpu_util) if gpu_util else None,
            "gpu_util_mean": round(sum(gpu_util) / len(gpu_util), 1) if gpu_util else None,
        }

    def dump_json(self, path: str) -> None:
        data = {
            "summary": self.summary(),
            "samples": [asdict(s) for s in self.samples],
            "metrics_samples": self.metrics_samples,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def dump_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["kind", "ok", "status", "dur_s", "tokens", "bytes", "error"])
            for s in self.samples:
                w.writerow(
                    [s.kind, int(s.ok), s.status, f"{s.dur:.4f}", s.tokens, s.bytes, s.error or ""]
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_headers(token: str, user_id: Optional[str] = None) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {token}"}
    if user_id is not None:
        h["X-User-ID"] = user_id
    return h


def _rand_user_id(prefix: str = "vu") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


def _rand_messages(n: int = 1) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for _ in range(n):
        msgs.append({"role": "user", "content": random.choice(USER_MSGS)})
    return msgs


def _summary_payload() -> List[Dict[str, str]]:
    convo = random.choice(SUMMARY_CONVOS)
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SUMMARY_SYSTEM}]
    msgs.extend({"role": r, "content": c} for (r, c) in convo)
    return msgs


async def _sleep_jitter(low=THINK_LOW, high=THINK_HIGH):
    await asyncio.sleep(random.uniform(low, high))


# ---------------------------------------------------------------------------
# REST: chat (non-stream)
# ---------------------------------------------------------------------------

async def do_chat(
    client: httpx.AsyncClient,
    stats: BenchStats,
    token: str,
    user_id: str,
    cfg: BenchCfg,
) -> None:
    payload = {
        "model": "qwen2.5-32b",
        "messages": _rand_messages(n=1 + random.randint(0, 2)),
        "stream": False,
        "max_tokens": cfg.chat_max_tokens,
    }
    headers = _auth_headers(token, user_id)
    headers["Content-Type"] = "application/json"
    url = f"{BASE_URL}/v1/chat/completions"
    start = time.time()
    ok = True
    status = None
    err = None
    try:
        r = await client.post(url, headers=headers, json=payload, timeout=None)
        status = r.status_code
        if r.status_code != 200:
            ok = False
            err = f"HTTP {r.status_code}"
        _ = r.text  # consume
    except Exception as e:  # pragma: no cover
        ok = False
        err = str(e)
    end = time.time()
    stats.record(
        LatencySample(kind="chat", start=start, end=end, ok=ok, status=status, error=err, user_id=user_id)
    )


# ---------------------------------------------------------------------------
# REST: chat stream (SSE)
# ---------------------------------------------------------------------------

async def do_stream(
    client: httpx.AsyncClient,
    stats: BenchStats,
    token: str,
    user_id: str,
    cfg: BenchCfg,
) -> None:
    payload = {
        "model": "qwen2.5-32b",
        "messages": _rand_messages(n=1 + random.randint(0, 2)),
        "stream": True,
        "max_tokens": cfg.stream_max_tokens,
    }
    headers = _auth_headers(token, user_id)
    headers["Content-Type"] = "application/json"
    url = f"{BASE_URL}/v1/chat/completions"
    start = time.time()
    ok = True
    status = None
    tokens = 0
    bytes_rcv = 0
    err = None
    try:
        async with client.stream("POST", url, headers=headers, json=payload, timeout=None) as resp:
            status = resp.status_code
            if status != 200:
                ok = False
            async for line in resp.aiter_lines():
                if line is None:
                    break
                bytes_rcv += len(line)
                if not line:
                    continue
                if line.startswith("data: "):
                    if line.strip() == "data: [DONE]":
                        break
                    tokens += 1
    except Exception as e:  # pragma: no cover
        ok = False
        err = str(e)
    end = time.time()
    stats.record(
        LatencySample(
            kind="stream",
            start=start,
            end=end,
            ok=ok,
            status=status,
            tokens=tokens,
            bytes=bytes_rcv,
            error=err,
            user_id=user_id,
        )
    )


# ---------------------------------------------------------------------------
# REST: summary extraction
# ---------------------------------------------------------------------------

async def do_summary(
    client: httpx.AsyncClient,
    stats: BenchStats,
    token: str,
    user_id: str,
    cfg: BenchCfg,
) -> None:
    payload = {"messages": _summary_payload()}
    headers = _auth_headers(token, user_id)
    headers["Content-Type"] = "application/json"
    url = f"{BASE_URL}/v1/summary"
    start = time.time()
    ok = True
    status = None
    err = None
    try:
        r = await client.post(url, headers=headers, json=payload, timeout=None)
        status = r.status_code
        if status != 200:
            ok = False
            err = f"HTTP {status}: {r.text[:100]}"
        else:
            _ = r.json()  # parse to surface JSON errors
    except Exception as e:  # pragma: no cover
        ok = False
        err = str(e)
    end = time.time()
    stats.record(
        LatencySample(kind="summary", start=start, end=end, ok=ok, status=status, error=err, user_id=user_id)
    )


# ---------------------------------------------------------------------------
# WebSocket roundtrip (optional)
# ---------------------------------------------------------------------------

async def do_ws_roundtrip(
    stats: BenchStats,
    token: str,
    user_id: str,
    cfg: BenchCfg,
) -> None:
    if websockets is None:
        return
    url = f"{WS_URL}?token={token}"
    start = time.time()
    ok = True
    err = None
    tokens = 0
    try:
        async with websockets.connect(url, extra_headers={"X-User-ID": user_id}, open_timeout=30) as ws:
            # handshake
            try:
                _ = await asyncio.wait_for(ws.recv(), timeout=10)
            except Exception:
                pass

            req_id = f"bench_{uuid.uuid4().hex[:6]}"
            payload = {
                "request_id": req_id,
                "messages": _rand_messages(n=2),
                "temperature": 0.1,
                "max_tokens": cfg.stream_max_tokens,
                "top_p": 0.9,
                "top_k": 40,
            }
            await ws.send(json.dumps(payload))

            # collect tokens
            end_token = False
            while not end_token:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    err = "ws timeout"
                    ok = False
                    break
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "stream_token":
                    tokens += 1
                elif t == "stream_end":
                    end_token = True
    except Exception as e:  # pragma: no cover
        ok = False
        err = str(e)
    end = time.time()
    stats.record(
        LatencySample(kind="stream", start=start, end=end, ok=ok, status=None, tokens=tokens, error=err, user_id=user_id)
    )


# ---------------------------------------------------------------------------
# Health & Metrics pollers
# ---------------------------------------------------------------------------

async def poll_health(
    client: httpx.AsyncClient,
    stats: BenchStats,
    stop_evt: asyncio.Event,
) -> None:
    url = f"{BASE_URL}/health"
    headers: Dict[str, str] = {}
    while not stop_evt.is_set():
        start = time.time()
        ok = True
        status = None
        err = None
        try:
            r = await client.get(url, headers=headers, timeout=10)
            status = r.status_code
            if status != 200:
                ok = False
                err = f"HTTP {status}"
        except Exception as e:  # pragma: no cover
            ok = False
            err = str(e)
        end = time.time()
        stats.record(LatencySample(kind="health", start=start, end=end, ok=ok, status=status, error=err))
        await asyncio.sleep(HEALTH_INTERVAL)


async def poll_metrics(
    client: httpx.AsyncClient,
    stats: BenchStats,
    stop_evt: asyncio.Event,
) -> None:
    url = f"{BASE_URL}/metrics"
    headers: Dict[str, str] = {}
    while not stop_evt.is_set():
        try:
            r = await client.get(url, headers=headers, timeout=10)
            raw = r.text
            m: Dict[str, Any] = {}
            def scrape(name: str) -> Optional[float]:
                for line in raw.splitlines():
                    if line.startswith(name + " "):
                        try:
                            return float(line.split()[1])
                        except Exception:  # pragma: no cover
                            return None
                return None
            m["fastapi_inference_queue_size"] = scrape("fastapi_inference_queue_size")
            m["gpu_utilization_percent"] = scrape("gpu_utilization_percent")
            m["cpu_usage_percent"] = scrape("cpu_usage_percent")
            m["ts"] = time.time()
            stats.record_metrics(m)
        except Exception:  # pragma: no cover
            pass
        await asyncio.sleep(METRICS_INTERVAL)


# ---------------------------------------------------------------------------
# Virtual user worker
# ---------------------------------------------------------------------------

async def vu_worker(
    idx: int,
    stats: BenchStats,
    token: str,
    weights: Tuple[float, float, float],
    stop_evt: asyncio.Event,
    cfg: BenchCfg,
    ws_mode: bool = False,
) -> None:
    """Loop issuing mixed requests until stop_evt set."""
    chat_w, stream_w, summary_w = weights
    total_w = chat_w + stream_w + summary_w
    chat_w /= total_w
    stream_w /= total_w
    summary_w /= total_w

    user_id = _rand_user_id(f"vu{idx}")

    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        while not stop_evt.is_set():
            r = random.random()
            if r < chat_w:
                await do_chat(client, stats, token, user_id, cfg)
            elif r < chat_w + stream_w:
                if ws_mode and websockets is not None and random.random() < 0.5:
                    await do_ws_roundtrip(stats, token, user_id, cfg)
                else:
                    await do_stream(client, stats, token, user_id, cfg)
            else:
                await do_summary(client, stats, token, user_id, cfg)
            await _sleep_jitter()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

async def smoke_test(token: str, cfg: BenchCfg) -> None:
    print("Running smoke tests...")
    stats = BenchStats()
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        await do_chat(client, stats, token, _rand_user_id("smoke"), cfg)
        await do_stream(client, stats, token, _rand_user_id("smoke"), cfg)
        await do_summary(client, stats, token, _rand_user_id("smoke"), cfg)
        stop = asyncio.Event()
        stop.set()
        await poll_health(client, stats, stop)  # minimal single call
    stats.close()
    print(json.dumps(stats.summary(), indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Bench runner
# ---------------------------------------------------------------------------

async def run_bench(
    clients: int,
    duration: float,
    token: str,
    weights: Tuple[float, float, float],
    cfg: BenchCfg,
    ws_mode: bool = False,
) -> BenchStats:
    stats = BenchStats()
    stop_evt = asyncio.Event()

    # pollers (health + metrics)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    poll_client = httpx.AsyncClient(limits=limits, timeout=None)
    poll_health_task = asyncio.create_task(poll_health(poll_client, stats, stop_evt))
    poll_metrics_task = asyncio.create_task(poll_metrics(poll_client, stats, stop_evt))

    # VU workers
    tasks = [
        asyncio.create_task(vu_worker(i, stats, token, weights, stop_evt, cfg, ws_mode=ws_mode))
        for i in range(clients)
    ]

    # run for duration
    await asyncio.sleep(duration)
    stop_evt.set()

    # await tasks
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0.1)  # flush pollers
    poll_health_task.cancel()
    poll_metrics_task.cancel()
    try:
        await poll_health_task
    except asyncio.CancelledError:
        pass
    try:
        await poll_metrics_task
    except asyncio.CancelledError:
        pass
    await poll_client.aclose()

    stats.close()
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Qwen concurrency bench")
    p.add_argument("--base-url", default=BASE_URL, help="Service base URL (default env BASE_URL)")
    p.add_argument("--token", default=API_TOKEN, help="Bearer token")
    p.add_argument("--clients", type=int, default=4, help="Concurrent virtual users")
    p.add_argument("--duration", type=float, default=30.0, help="Duration seconds")
    p.add_argument("--chat-weight", type=float, default=DEFAULT_CHAT_WEIGHT)
    p.add_argument("--stream-weight", type=float, default=DEFAULT_STREAM_WEIGHT)
    p.add_argument("--summary-weight", type=float, default=DEFAULT_SUMMARY_WEIGHT)
    p.add_argument("--ws-mode", action="store_true", help="Include WebSocket streams in stream mix")
    p.add_argument("--smoke", action="store_true", help="Run quick 1x each request for validation")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="max_tokens demandé côté API pour chat & stream (overrides defaults).")
    p.add_argument("--stream-max-tokens", type=int, default=None,
                   help="Override only stream max_tokens (default: same as --max-tokens).")
    p.add_argument("--summary-max-tokens", type=int, default=SUMMARY_MAX_TOKENS,
                   help="Reserved (API /v1/summary uses fixed 300 today; future use).")
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-csv", default=OUT_CSV)
    return p.parse_args()


async def _amain():
    args = _parse_args()

    # update globals for convenience (used in helpers)
    global BASE_URL, API_TOKEN, WS_URL
    BASE_URL = args.base_url.rstrip("/")
    API_TOKEN = args.token
    WS_URL = f"{BASE_URL.replace('http', 'ws')}/ws"

    # build runtime config
    cfg = BenchCfg(
        chat_max_tokens=args.max_tokens,
        stream_max_tokens=args.stream_max_tokens if args.stream_max_tokens is not None else args.max_tokens,
        summary_max_tokens=args.summary_max_tokens,
    )

    print("=== BENCH CONFIG ===")
    print(json.dumps({
        "base_url": BASE_URL,
        "clients": args.clients,
        "duration_s": args.duration,
        "weights": {
            "chat": args.chat_weight,
            "stream": args.stream_weight,
            "summary": args.summary_weight,
        },
        "chat_max_tokens": cfg.chat_max_tokens,
        "stream_max_tokens": cfg.stream_max_tokens,
        "summary_max_tokens": cfg.summary_max_tokens,
        "ws_mode": args.ws_mode,
    }, indent=2, ensure_ascii=False))

    if args.smoke:
        await smoke_test(API_TOKEN, cfg)
        return

    weights = (args.chat_weight, args.stream_weight, args.summary_weight)
    stats = await run_bench(args.clients, args.duration, API_TOKEN, weights, cfg, ws_mode=args.ws_mode)

    # print summary
    summ = stats.summary()
    print("\n=== BENCH SUMMARY ===")
    print(json.dumps(summ, indent=2, ensure_ascii=False))

    # dump files
    stats.dump_json(args.out_json)
    stats.dump_csv(args.out_csv)
    print(f"Results saved: {args.out_json}, {args.out_csv}")


def main():
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:  # pragma: no cover
        print("Interrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
