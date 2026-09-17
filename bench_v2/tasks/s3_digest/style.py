"""s3's own agentic wording -- the news-digest framing.

This is the wording s3 shipped with, which was also the shared fallback until
2026-09-16, so every s3 run up to v5 recorded exactly these strings. v4 and v5
pass this explicitly, which is what keeps their transcripts reproducible now that
the fallback is generic.

The strings themselves live in ``helpers/system_prompt.LEGACY_NEWS_DIGEST_STYLE``,
frozen, because s1 v2 recorded them too and its records have to stay readable. s3
owns the *decision* to use them; the helper owns the historical record.
"""

from __future__ import annotations

from typing import Any, Dict

from bench_v2.helpers.system_prompt import LEGACY_NEWS_DIGEST_STYLE

NEWS_DIGEST_STYLE: Dict[str, Any] = dict(LEGACY_NEWS_DIGEST_STYLE)
