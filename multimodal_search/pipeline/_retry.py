import logging
import random
import time
from typing import Callable, TypeVar

from multimodal_search.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    max_attempts: int | None = None,
    label: str = "operation",
) -> T:
    """
    Execute *fn* with exponential backoff + jitter on failure.

    Args:
        fn:           Thunk to execute (zero-argument callable).
        max_attempts: Override settings.max_attempts (default: None -> use setting).
        label:        Human-readable label for log messages.

    Returns:
        Whatever *fn* returns on success.

    Raises:
        The last exception if all attempts are exhausted.
    """
    max_attempts = max_attempts or settings.max_attempts
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "%s attempt %d/%d failed: %s",
                label,
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts:
                backoff = (2**attempt) + random.uniform(0, 1)
                logger.debug("Retrying %s in %.1fs…", label, backoff)
                time.sleep(backoff)

    raise last_exc  # type: ignore[misc]
