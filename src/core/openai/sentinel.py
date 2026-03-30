"""Helpers for OpenAI Sentinel proof-of-work tokens."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Sequence

logger = logging.getLogger(__name__)


DEFAULT_SENTINEL_DIFF = "0fffff"
DEFAULT_MAX_ITERATIONS = 500_000
_SCREEN_SIGNATURES = (3000, 3120, 4000, 4160)
_LANGUAGE_SIGNATURE = "en-US,es-US,en,es"
_NAVIGATOR_KEYS = ("location", "ontransitionend", "onprogress")
_WINDOW_KEYS = ("window", "document", "navigator")


class SentinelPOWError(RuntimeError):
    """Raised when a Sentinel proof-of-work token cannot be solved."""


def _format_browser_time() -> str:
    """Match the browser-style timestamp used by public Sentinel solvers."""
    browser_now = datetime.now(timezone(timedelta(hours=-5)))
    return browser_now.strftime("%a %b %d %Y %H:%M:%S") + " GMT-0500 (Eastern Standard Time)"


def build_sentinel_config(user_agent: str) -> list:
    """Build a browser-like fingerprint payload for the Sentinel PoW solver."""
    perf_ms = time.perf_counter() * 1000
    epoch_ms = (time.time() * 1000) - perf_ms
    return [
        random.choice(_SCREEN_SIGNATURES),
        _format_browser_time(),
        4294705152,
        0,
        user_agent,
        "",
        "",
        "en-US",
        _LANGUAGE_SIGNATURE,
        0,
        random.choice(_NAVIGATOR_KEYS),
        "location",
        random.choice(_WINDOW_KEYS),
        perf_ms,
        str(uuid.uuid4()),
        "",
        8,
        epoch_ms,
    ]


def _encode_pow_payload(config: Sequence[object], nonce: int) -> bytes:
    prefix = (json.dumps(config[:3], separators=(",", ":"), ensure_ascii=False)[:-1] + ",").encode("utf-8")
    middle = (
        "," + json.dumps(config[4:9], separators=(",", ":"), ensure_ascii=False)[1:-1] + ","
    ).encode("utf-8")
    suffix = ("," + json.dumps(config[10:], separators=(",", ":"), ensure_ascii=False)[1:]).encode("utf-8")
    body = prefix + str(nonce).encode("ascii") + middle + str(nonce >> 1).encode("ascii") + suffix
    return base64.b64encode(body)


def solve_sentinel_pow(
    seed: str,
    difficulty: str,
    config: Sequence[object],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> str:
    """Solve the Sentinel PoW challenge and return the base64 payload."""
    seed_bytes = seed.encode("utf-8")
    target = bytes.fromhex(difficulty)
    prefix_length = len(target)

    for nonce in range(max_iterations):
        encoded = _encode_pow_payload(config, nonce)
        digest = hashlib.sha3_512(seed_bytes + encoded).digest()
        if digest[:prefix_length] <= target:
            return encoded.decode("ascii")

    raise SentinelPOWError(f"failed to solve sentinel pow after {max_iterations} attempts")


def build_sentinel_pow_token(
    user_agent: str,
    difficulty: str = DEFAULT_SENTINEL_DIFF,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> str:
    """Build the `p` token required by the Sentinel request endpoint."""
    config = build_sentinel_config(user_agent)
    seed = format(random.random())
    solution = solve_sentinel_pow(seed, difficulty, config, max_iterations=max_iterations)
    return f"gAAAAAC{solution}"


def solve_pow_challenge(seed: str, difficulty: str, user_agent: str) -> str:
    """
    根据服务端返回的 seed 和 difficulty 求解 PoW

    Args:
        seed: 服务端提供的随机种子
        difficulty: 难度值（如 "0fffff"）
        user_agent: 浏览器 User-Agent

    Returns:
        PoW token 字符串
    """
    config = build_sentinel_config(user_agent)
    solution = solve_sentinel_pow(seed, difficulty, config, max_iterations=DEFAULT_MAX_ITERATIONS)
    return f"gAAAAAC{solution}"


# --------------- Turnstile 求解框架 ---------------

import abc
from typing import Optional as _Optional


class TurnstileSolver(abc.ABC):
    """Turnstile 求解器抽象基类"""

    @abc.abstractmethod
    def solve(self, site_key: str, page_url: str, proxy: _Optional[str] = None) -> _Optional[str]:
        """
        求解 Turnstile 挑战

        Args:
            site_key: Turnstile site key
            page_url: 当前页面 URL
            proxy: 代理 URL（可选）

        Returns:
            Turnstile token (cf_clearance 或类似值)
        """
        pass


class DummyTurnstileSolver(TurnstileSolver):
    """占位 Turnstile 求解器（始终返回 None，需接入真实服务）"""

    def solve(self, site_key: str, page_url: str, proxy: _Optional[str] = None) -> _Optional[str]:
        logger.warning(f"Turnstile 挑战未实现真实求解器，site_key={site_key}, page_url={page_url}")
        return None


# 全局求解器实例（可通过配置替换）
_turnstile_solver: TurnstileSolver = DummyTurnstileSolver()


def set_turnstile_solver(solver: TurnstileSolver):
    """设置 Turnstile 求解器（用于接入 capsolver 等服务）"""
    global _turnstile_solver
    _turnstile_solver = solver


def solve_turnstile_challenge(site_key: str, page_url: str, proxy: _Optional[str] = None) -> _Optional[str]:
    """
    求解 Turnstile 挑战

    Args:
        site_key: Turnstile site key
        page_url: 页面 URL
        proxy: 代理 URL

    Returns:
        Turnstile token
    """
    return _turnstile_solver.solve(site_key, page_url, proxy)
