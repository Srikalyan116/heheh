import sys
from loguru import logger

def configure_stdout_logging(level: str = "INFO") -> None:
    # idempotent: safe to call in parent and child
    try:
        logger.remove()
    except Exception:
        pass

    # Force immediate flushing to the pod's stdout
    try:
        # Python 3.7+: make stdout line-buffered so logs appear immediately
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    logger.add(
        sys.stdout,
        level=level,
        enqueue=True,          # non-blocking, safe across processes
        backtrace=False,
        diagnose=False,
    )