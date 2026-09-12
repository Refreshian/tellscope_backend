# -*- coding: utf-8 -*-
"""Прогресс агентного запуска: шаги, проценты, оценка остатка и heartbeat.

Зачем модуль: длинные шаги (чтение текстов локальной моделью, вызовы модели) занимают
минуты, а в интерфейсе был виден только статичный текст — понять, работает задача или
зависла, было нельзя. Здесь считается единое состояние прогресса, из которого в поток
запуска (WebSocket ``/ws/agent-run/{id}`` и ``GET /agent/run/{id}``) уходят события:

  * ``progress``  — что идёт сейчас, сколько шагов сделано из плана, процент и оценка остатка;
  * ``heartbeat`` — «запуск жив» во время длинной операции (не реже, чем раз в 10 секунд).

Состояние живёт в контексте запуска (``ctx.progress``), поэтому его видят и цикл агента
(``loop.py``), и конструктор шагов (``pipeline.py``), и инструменты (через ``registry.execute``,
например самый долгий шаг — ``analyze_texts``).

Оценка остатка (ETA) считается честно и уточняется по ходу:
  * база — средняя длительность уже завершённых шагов этого запуска;
  * до первого завершённого шага — история: среднее время шага по последним запускам
    того же режима (её кладёт ``runs.py``);
  * внутри шага с под-прогрессом (пачки ``analyze_texts``) — по фактическому темпу пачек.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import Any, Dict, List, Optional

# Как часто во время длинной операции отправлять heartbeat (секунды).
# 8 секунд: между переходами шагов пауза не превышает 15 секунд, и в интерфейсе
# всегда видно, что запуск жив.
HEARTBEAT_SEC = float(os.environ.get("TELLSCOPE_PROGRESS_HB") or 8.0)

# Название этапа, пока запуск ещё не назвал шаг.
START_STAGE = "подготовка запуска"


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


class ProgressTracker:
    """Состояние прогресса одного запуска: шаги, под-шаги, ETA, heartbeat."""

    def __init__(self, ctx: Any, *, total: int = 0, history_step_sec: float = 0.0, mode: str = "",
                 history_run_sec: float = 0.0) -> None:
        self.ctx = ctx
        self.started = time.time()
        self.total = max(0, _int(total))
        self.mode = str(mode or "")
        self.history_step_sec = float(history_step_sec or 0.0)
        self.history_run_sec = float(history_run_sec or 0.0)
        # сколько шагов уже завершено (для процента) и идёт ли шаг прямо сейчас
        self.step = 0
        self.in_stage = False
        self.stage = START_STAGE
        self.detail = ""
        # под-прогресс внутри шага (пачки чтения текстов) — доля текущего шага
        self.sub_stage = ""
        self.sub_done = 0
        self.sub_total = 0
        self.sub_units_done = 0
        self.sub_units_total = 0
        self.sub_units_parallel = 0
        self.sub_started = 0.0
        self._last_sub_eta = 0.0
        self._durations: List[float] = []
        self._stage_started = 0.0
        self._last_emit = 0.0
        # Сколько heartbeat-обёрток открыто сейчас: вложенные не создают вторую задачу.
        self._beats = 0

    # ------------------------------------------------------------------ расчёты

    def avg_step_sec(self) -> float:
        """Средняя длительность шага: факт этого запуска, иначе история того же режима."""
        if self._durations:
            value = sum(self._durations) / float(len(self._durations))
            # Один быстрый шаг (обзор датасета за 8 мс) ещё не даёт права оценивать остаток:
            # пока выборки мало, опираемся на историю запусков того же режима.
            if value >= 2.0 or len(self._durations) >= 3:
                return value
            return self.history_step_sec
        return self.history_step_sec

    def current_step(self) -> int:
        number = self.step + (1 if self.in_stage else 0)
        if self.total:
            number = min(number, self.total)
        return number

    def percent(self) -> Optional[int]:
        if not self.total:
            return None
        frac = 0.0
        if self.in_stage and self.sub_total > 0:
            frac = min(1.0, float(self.sub_done) / float(self.sub_total))
        value = (self.step + frac) / float(self.total)
        return max(0, min(100, int(round(value * 100))))

    def eta_seconds(self) -> Optional[int]:
        """Оценка остатка. None — если считать пока не из чего (свободный цикл без плана)."""
        now = time.time()
        avg = self.avg_step_sec()
        known = False
        current = 0.0
        if self.in_stage:
            if self.sub_total > 0 and self.sub_done <= 0:
                # Пачки ещё не вернулись: темпа нет, честной оценки пока быть не может.
                return None
            if self.sub_total > 0 and self.sub_done > 0:
                current = self._sub_estimate(now)
                known = current > 0
                if not known and self._last_sub_eta:
                    # Чтение закончилось, а шаг ещё идёт (сводка по темам, разбор моделью):
                    # держим последнюю оценку, а не показываем «осталось 0 секунд».
                    current = self._last_sub_eta
                    known = True
                elif known and not self.total:
                    self._last_sub_eta = current
            elif avg:
                # Шаг без под-прогресса: опираемся на среднюю длительность шага.
                current = avg
                known = True
        rest_steps = max(0, self.total - self.current_step()) if self.total else 0
        if rest_steps and avg:
            current += rest_steps * avg
            known = True
        if not known:
            return None
        if not self.total and not (self.in_stage and self.sub_total):
            # свободный агентный цикл: общая длительность заранее неизвестна
            return None
        return int(round(current))

    def _sub_estimate(self, now: float) -> float:
        """Оценка остатка внутри шага по под-прогрессу.

        Пачки чтения текстов идут параллельно (три сразу), поэтому «последовательная»
        оценка по числу прочитанных сообщений завышает остаток, а оценка «остальные пачки
        закончатся вместе с уже завершённой самой медленной» — занижает. Берём среднее двух
        границ, чтобы ошибка оставалась в пределах полуминуты, а не минут.
        """
        elapsed = max(0.0, now - self.sub_started)
        if elapsed <= 0:
            return 0.0
        if self.sub_units_total > 1 and 0 < self.sub_units_done < self.sub_units_total:
            rate = elapsed / float(self.sub_units_done)
            upper = max(0.0, float(self.sub_units_total - self.sub_units_done) * rate)
            unit_time = elapsed / float(self.sub_units_done)
        elif self.sub_done < self.sub_total:
            rate = elapsed / float(self.sub_done)
            upper = max(0.0, float(self.sub_total - self.sub_done) * rate)
            unit_time = elapsed
        else:
            return 0.0
        if upper <= 0:
            return 0.0
        parallel = max(1, self.sub_units_parallel or self.sub_units_done or 1)
        units_total = self.sub_units_total or 1
        waves = -(-units_total // parallel)  # ceil: сколько «волн» пачек нужно всего
        lower = max(0.0, unit_time * waves - elapsed)
        return (upper + min(upper, lower)) / 2.0

    def eta_scope(self) -> str:
        if self.total:
            return "run"
        return "stage" if self.in_stage and self.sub_total else "unknown"

    def payload(self, type_: str = "progress", **extra: Any) -> Dict[str, Any]:
        """Формат события для фронтенда.

        Ключи ``stage/step/total/percent/eta_seconds/detail`` — контракт из задачи,
        остальное (``elapsed``, ``done``, ``eta_scope``, ``sub``) добавлено для интерфейса.
        """
        eta = self.eta_seconds()
        data: Dict[str, Any] = {
            "type": type_,
            "stage": self.sub_stage or self.stage or START_STAGE,
            "step": self.current_step(),
            "done": self.step,
            "total": self.total or None,
            "percent": self.percent(),
            "eta_seconds": eta,
            "eta_scope": self.eta_scope(),
            "elapsed": int(max(0.0, time.time() - self.started)),
            "detail": self.detail,
        }
        if self.sub_total:
            data["sub"] = {
                "done": self.sub_done,
                "total": self.sub_total,
                "percent": int(round(100.0 * self.sub_done / float(self.sub_total))),
            }
            if self.sub_units_total:
                data["sub"]["units_done"] = self.sub_units_done
                data["sub"]["units_total"] = self.sub_units_total
        if self.mode:
            data["mode"] = self.mode
        if self.history_run_sec > 0:
            # «обычно такой запуск занимает ~2 мин» — мягкая подсказка, когда плана нет
            data["history_run_sec"] = int(round(self.history_run_sec))
        for key, value in extra.items():
            if value is not None:
                data[key] = value
        return data

    # ------------------------------------------------------------------ события

    async def _emit(self, payload: Dict[str, Any]) -> None:
        self._last_emit = time.time()
        await self.ctx.event(payload)

    async def start_run(self) -> None:
        """Первый прогресс запуска: общее число шагов из плана (если план известен)."""
        if self.total:
            self.detail = f"план: {self.total} шагов"
        else:
            self.detail = "число шагов заранее неизвестно — показываю прогресс по факту"
        await self._emit(self.payload())

    async def begin_stage(self, stage: str, detail: str = "") -> None:
        """Начался новый шаг: счётчик шагов остаётся на предыдущем, пока шаг не завершится."""
        self.stage = str(stage or self.stage or START_STAGE)
        self.detail = str(detail or "выполняется")
        self.in_stage = True
        self._stage_started = time.time()
        self._reset_sub()
        await self._emit(self.payload())

    async def set_stage(self, stage: str, detail: str = "") -> None:
        """Сменить подпись «что делается сейчас», не трогая счётчик шагов (например, ответ модели)."""
        self.stage = str(stage or self.stage)
        if detail:
            self.detail = str(detail)
        await self._emit(self.payload())

    async def end_stage(self, detail: str = "", ok: bool = True) -> None:
        """Шаг завершён: инкремент, уточнение средней длительности и ETA."""
        if self._stage_started:
            self._durations.append(max(0.0, time.time() - self._stage_started))
            self._stage_started = 0.0
        self.step += 1
        self.in_stage = False
        self._reset_sub()
        self.detail = str(detail or ("готово" if ok else "шаг не удался"))
        await self._emit(self.payload())

    async def sub(self, done: int, total: int, detail: str = "", stage: Optional[str] = None,
                  units_done: Optional[int] = None, units_total: Optional[int] = None,
                  units_parallel: Optional[int] = None) -> None:
        """Под-прогресс внутри шага: пачки чтения текстов и другие длинные под-операции.

        ``done``/``total`` — в сообщениях (то, что видит пользователь), ``units_*`` — в пачках:
        пачки читаются параллельно, поэтому оценка остатка точнее по ним.
        """
        if stage:
            self.sub_stage = str(stage)
        if not self.sub_total:
            self.sub_started = time.time()
            self._last_sub_eta = 0.0
        self.sub_total = max(0, _int(total))
        self.sub_done = max(0, min(_int(done), self.sub_total or _int(done)))
        if units_total is not None:
            self.sub_units_total = max(0, _int(units_total))
        if units_done is not None:
            self.sub_units_done = max(0, _int(units_done))
        if units_parallel is not None:
            self.sub_units_parallel = max(0, _int(units_parallel))
        if detail:
            self.detail = str(detail)
        await self._emit(self.payload())

    async def note(self, detail: str) -> None:
        """Обновить пояснение («что делается сейчас») без смены шага."""
        if detail:
            self.detail = str(detail)
        await self._emit(self.payload())

    async def finish_run(self, status: str, detail: str = "") -> None:
        """Финальное событие прогресса: 100% и «выполнено за …»."""
        self.in_stage = False
        self._reset_sub()
        if self.total:
            self.step = max(self.step, self.total)
        self.detail = str(detail or status)
        await self._emit(self.payload(**{"status": status}))

    def _reset_sub(self) -> None:
        self.sub_stage = ""
        self.sub_done = 0
        self.sub_total = 0
        self.sub_units_done = 0
        self.sub_units_total = 0
        self.sub_units_parallel = 0
        self.sub_started = 0.0
        self._last_sub_eta = 0.0

    # ----------------------------------------------------------------- heartbeat

    @contextlib.asynccontextmanager
    async def heartbeat(self, stage: Optional[str] = None, interval: Optional[float] = None):
        """Пока идёт длинная операция, раз в ``interval`` секунд пишем «жив».

        Задача гасится в ``finally``, поэтому фоновых задач не остаётся и поток запуска
        не ломается; если emit недоступен (инструмент вызван вне запуска) — ничего не делаем.
        Вложенные вызовы (registry.execute → инструмент с собственным heartbeat) не создают
        второй цикл heartbeat: работает один, самый внешний.
        """
        task: Optional[asyncio.Task] = None
        self._beats += 1
        if self._beats == 1 and getattr(self.ctx, "emit", None) is not None:
            # Первый «жив» отправляем сразу: между короткими шагами (графики, отчёты) иначе
            # набегала пауза больше 15 секунд, пока новый цикл heartbeat дойдёт до первого тика.
            try:
                await self._emit(self.payload("heartbeat"))
            except Exception:
                pass
            try:
                task = asyncio.ensure_future(self._beat(stage, float(interval or HEARTBEAT_SEC)))
            except Exception:
                task = None
        try:
            yield
        finally:
            self._beats = max(0, self._beats - 1)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _beat(self, stage: Optional[str], interval: float) -> None:
        try:
            while True:
                await asyncio.sleep(max(2.0, interval))
                payload = self.payload("heartbeat")
                if stage:
                    payload["stage"] = str(stage)
                await self._emit(payload)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — heartbeat не должен ломать запуск
            return


def tracker(ctx: Any, *, total: int = 0, history_step_sec: float = 0.0, mode: str = "",
            history_run_sec: float = 0.0) -> ProgressTracker:
    """Возвращает трекер запуска, создавая его при первом обращении."""
    existing = getattr(ctx, "progress", None)
    if isinstance(existing, ProgressTracker):
        if total and not existing.total:
            existing.total = max(0, _int(total))
        return existing
    created = ProgressTracker(ctx, total=total, history_step_sec=history_step_sec, mode=mode,
                              history_run_sec=history_run_sec)
    try:
        ctx.progress = created
    except Exception:
        pass
    return created


def human_duration(seconds: Any) -> str:
    """'3 мин 20 с' — для текстов «выполнено за …»."""
    try:
        total = int(round(float(seconds or 0)))
    except Exception:
        total = 0
    if total < 60:
        return f"{total} с"
    minutes, rest = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} мин {rest} с" if rest else f"{minutes} мин"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes} мин"
