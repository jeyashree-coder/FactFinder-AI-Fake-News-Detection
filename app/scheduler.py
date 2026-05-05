from __future__ import annotations

import threading
import time

import schedule

from app.ml import model_manager
from app.scraper import collect_bbc_news

_scheduler_started = False


def scheduled_collection() -> None:
    result = collect_bbc_news()
    if result.get("status") == "completed" and result.get("inserted", 0) > 0:
        model_manager.train()


def _scheduler_loop() -> None:
    while True:
        schedule.run_pending()
        time.sleep(30)


def start_scheduler() -> None:
    global _scheduler_started
    if _scheduler_started:
        return
    schedule.every().day.at("09:00").do(scheduled_collection)
    threading.Thread(target=_scheduler_loop, daemon=True).start()
    _scheduler_started = True
