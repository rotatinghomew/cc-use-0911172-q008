from __future__ import annotations

import datetime as dt
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .model import CapacityInterval, ShiftCalendar, WaitRiskError, minutes_label

DISCIPLINES = {'fifo', 'appointment_first'}


@dataclass(frozen=True)
class Arrival:
    customer_id: str
    day: dt.date
    arrival: dt.datetime
    channel: str
    slot_minutes: int
    service_seconds: int
    evidence: str = 'empirical'
    service_evidence: str = 'empirical'
    source_date: dt.date | None = None

    @property
    def slot_label(self) -> str:
        return minutes_label(self.slot_minutes)


@dataclass(frozen=True)
class ServiceRecord:
    customer_id: str
    channel: str
    arrival: dt.datetime
    started: dt.datetime
    finished: dt.datetime
    arrival_day: dt.date
    arrival_slot_minutes: int
    arrival_shift_key: tuple[dt.date, str] | None
    served_shift_key: tuple[dt.date, str] | None
    service_seconds: int
    demand_evidence: str
    service_evidence: str

    @property
    def wait_seconds(self) -> float:
        return (self.started - self.arrival).total_seconds()

    @property
    def same_shift(self) -> bool:
        return self.arrival_shift_key is not None and self.arrival_shift_key == self.served_shift_key

    @property
    def same_day(self) -> bool:
        return self.started.date() == self.arrival.date()

    def as_dict(self) -> dict[str, Any]:
        return {'customer_id': self.customer_id, 'channel': self.channel,
                'arrival': self.arrival.isoformat(), 'started': self.started.isoformat(),
                'finished': self.finished.isoformat(), 'wait_seconds': round(self.wait_seconds, 3),
                'service_seconds': self.service_seconds, 'arrival_slot': minutes_label(self.arrival_slot_minutes),
                'arrival_shift': _shift_key_label(self.arrival_shift_key),
                'served_shift': _shift_key_label(self.served_shift_key),
                'same_shift': self.same_shift, 'same_day': self.same_day,
                'demand_evidence': self.demand_evidence, 'service_evidence': self.service_evidence}


@dataclass(frozen=True)
class SlotOutcome:
    date: str
    weekday: int
    slot: str
    slot_minutes: int
    arrivals: int
    served: int
    same_shift: int
    deferred: int
    waiting: int
    mean_wait_seconds: float
    p50_wait_seconds: float
    p90_wait_seconds: float
    p99_wait_seconds: float
    max_wait_seconds: float
    breach_target: int
    breach_promise: int
    backlog_end: int
    evidence: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {'date': self.date, 'weekday': self.weekday, 'slot': self.slot,
                'slot_minutes': self.slot_minutes, 'arrivals': self.arrivals, 'served': self.served,
                'same_shift': self.same_shift, 'deferred': self.deferred, 'waiting': self.waiting,
                'mean_wait_seconds': round(self.mean_wait_seconds, 1),
                'p50_wait_seconds': round(self.p50_wait_seconds, 1),
                'p90_wait_seconds': round(self.p90_wait_seconds, 1),
                'p99_wait_seconds': round(self.p99_wait_seconds, 1),
                'max_wait_seconds': round(self.max_wait_seconds, 1),
                'breach_target': self.breach_target, 'breach_promise': self.breach_promise,
                'backlog_end': self.backlog_end, 'evidence': dict(self.evidence)}


@dataclass(frozen=True)
class DayOutcome:
    date: dt.date
    arrivals: int
    served: int
    same_shift: int
    deferred: int
    unserved: int
    max_wait_seconds: float
    p90_wait_seconds: float
    breach_target: int
    breach_promise: int
    backlog_end: int
    channel: Mapping[str, Mapping[str, Any]]
    slots: tuple[SlotOutcome, ...]

    def as_dict(self) -> dict[str, Any]:
        return {'date': self.date.isoformat(), 'arrivals': self.arrivals, 'served': self.served,
                'same_shift': self.same_shift, 'deferred': self.deferred, 'unserved': self.unserved,
                'max_wait_seconds': round(self.max_wait_seconds, 1),
                'p90_wait_seconds': round(self.p90_wait_seconds, 1),
                'breach_target': self.breach_target, 'breach_promise': self.breach_promise,
                'backlog_end': self.backlog_end,
                'channel': {key: dict(value) for key, value in self.channel.items()},
                'slots': [item.as_dict() for item in self.slots]}


@dataclass(frozen=True)
class SimulationResult:
    records: tuple[ServiceRecord, ...]
    unserved: tuple[Arrival, ...]
    days: tuple[DayOutcome, ...]
    discipline: str

    @property
    def arrivals(self) -> int:
        return len(self.records) + len(self.unserved)

    def as_dict(self) -> dict[str, Any]:
        return {'discipline': self.discipline, 'arrivals': self.arrivals,
                'served': len(self.records), 'unserved': len(self.unserved),
                'days': [item.as_dict() for item in self.days]}


def _shift_key_label(key: tuple[dt.date, str] | None) -> str | None:
    return None if key is None else f'{key[0].isoformat()}:{key[1]}'


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _server_segments(calendar: ShiftCalendar, first_day: dt.date,
                     last_day: dt.date) -> dict[tuple[str, int], tuple[CapacityInterval, ...]]:
    """Per-seat availability. Seat ``index`` only exists on days scheduled with that many seats."""
    grouped: dict[tuple[str, int], list[CapacityInterval]] = defaultdict(list)
    day = first_day
    while day <= last_day:
        for shift in calendar.shifts:
            seats = calendar.seats_for(day, shift)
            if day.weekday() not in shift.weekdays or seats <= 0:
                continue
            start, end = shift.window(day, calendar.timezone)
            open_blocks = []
            cursor = start
            for pause in shift.breaks:
                pause_start = dt.datetime.combine(day, pause.start, calendar.timezone)
                pause_end = dt.datetime.combine(day, pause.end, calendar.timezone)
                if pause_start > cursor:
                    open_blocks.append((cursor, pause_start))
                cursor = max(cursor, pause_end)
            if cursor < end:
                open_blocks.append((cursor, end))
            for index in range(seats):
                for block_start, block_end in open_blocks:
                    grouped[(shift.shift_id, index)].append(
                        CapacityInterval(block_start, block_end, 1, shift.shift_id))
        day += dt.timedelta(days=1)
    return {key: tuple(sorted(values, key=lambda item: (item.start, item.end)))
            for key, values in grouped.items()}


def earliest_start(segments: Sequence[CapacityInterval], moment: dt.datetime) -> dt.datetime | None:
    for segment in segments:
        if segment.end <= moment:
            continue
        start = max(moment, segment.start)
        return start if start < segment.end else None
    return None


def _entitlement(calendar: ShiftCalendar, arrival: dt.datetime, last_day: dt.date) -> tuple[dt.date, str] | None:
    """Shift the customer belongs to on arrival: the on-duty shift, else the next opening shift."""
    for item in calendar.windows(arrival.date()):
        if item.start <= arrival < item.end:
            return (arrival.date(), item.shift_id)
    cursor = arrival.date()
    while cursor <= last_day:
        for item in calendar.windows(cursor):
            if item.start >= arrival:
                return (cursor, item.shift_id)
        cursor += dt.timedelta(days=1)
    return None


def simulate_arrivals(
    calendar: ShiftCalendar,
    arrivals: Sequence[Arrival],
    first_day: dt.date,
    last_day: dt.date,
    discipline: str = 'fifo',
    open_ended_days: int = 7,
    target_seconds: int = 900,
    promise_seconds: int = 1800,
) -> SimulationResult:
    """Queue simulation across recurring shifts.

    Rules:
    * only an on-duty (non-break, non-closed) minute can start a service;
    * a running service may finish after a break or close, but never starts in one;
    * a seat only exists on days its shift is scheduled with that many seats;
    * customers are not dropped within the horizon; leftovers are reported as ``unserved``;
    * a deferred customer is served by the next shift with a free seat (possibly another day);
    * FIFO within channel, appointments ahead of walk-ins when ``discipline='appointment_first'``.
    """
    if discipline not in DISCIPLINES:
        raise WaitRiskError('unknown queue discipline')
    if target_seconds <= 0 or promise_seconds < target_seconds:
        raise WaitRiskError('invalid service thresholds')
    if not arrivals:
        return SimulationResult((), (), (), discipline)
    horizon_end = dt.datetime.combine(last_day + dt.timedelta(days=open_ended_days), dt.time.max, calendar.timezone)
    segments = _server_segments(calendar, first_day, last_day + dt.timedelta(days=open_ended_days))
    servers = [{'key': key, 'next_free': None} for key in sorted(segments)]
    if not servers:
        ordered = sorted(arrivals, key=lambda item: (item.arrival, item.customer_id))
        return SimulationResult((), tuple(ordered), (), discipline)

    ordered = sorted(arrivals, key=lambda item: (item.arrival, item.customer_id))

    def best_start(moment: dt.datetime) -> tuple[dt.datetime, dict[str, Any]] | None:
        best: tuple[dt.datetime, dict[str, Any]] | None = None
        for server in servers:
            cursor = moment if server['next_free'] is None else max(moment, server['next_free'])
            start = earliest_start(segments[server['key']], cursor)
            if start is None or start >= horizon_end:
                continue
            if best is None or (start, server['key']) < (best[0], best[1]['key']):
                best = (start, server)
        return best

    def queue_key(item: Arrival) -> tuple[Any, ...]:
        priority = 0 if (discipline == 'appointment_first' and item.channel == 'appointment') else 1
        return (priority, item.arrival, item.customer_id)

    pending: list[tuple[tuple[Any, ...], Arrival]] = []
    records: list[ServiceRecord] = []
    index = 0

    while index < len(ordered) or pending:
        if not pending:
            heapq.heappush(pending, (queue_key(ordered[index]), ordered[index]))
            index += 1
        head = pending[0][1]
        candidate = best_start(head.arrival)
        if candidate is None:
            break
        start, server = candidate
        added = False
        while index < len(ordered) and ordered[index].arrival < start:
            heapq.heappush(pending, (queue_key(ordered[index]), ordered[index]))
            index += 1
            added = True
        if added and pending[0][1] is not head:
            continue
        heapq.heappop(pending)
        finish = start + dt.timedelta(seconds=head.service_seconds)
        server['next_free'] = finish
        shift_id, seat_index = server['key']
        records.append(ServiceRecord(
            customer_id=head.customer_id, channel=head.channel, arrival=head.arrival, started=start,
            finished=finish, arrival_day=head.day, arrival_slot_minutes=head.slot_minutes,
            arrival_shift_key=_entitlement(calendar, head.arrival, last_day),
            served_shift_key=(start.date(), shift_id),
            service_seconds=head.service_seconds, demand_evidence=head.evidence,
            service_evidence=head.service_evidence))
    unserved = tuple(item for _, item in sorted(pending, key=lambda entry: entry[0]))
    days = _day_outcomes(records, unserved, arrivals, first_day, last_day, target_seconds, promise_seconds)
    return SimulationResult(tuple(records), unserved, days, discipline)


def _day_outcomes(
    records: Sequence[ServiceRecord],
    unserved: Sequence[Arrival],
    arrivals: Sequence[Arrival],
    first_day: dt.date,
    last_day: dt.date,
    target_seconds: int,
    promise_seconds: int,
) -> tuple[DayOutcome, ...]:
    by_day: dict[dt.date, list[ServiceRecord]] = defaultdict(list)
    for record in records:
        by_day[record.arrival_day].append(record)
    unserved_by_day: dict[dt.date, list[Arrival]] = defaultdict(list)
    for item in unserved:
        unserved_by_day[item.day].append(item)
    arrivals_by_day: dict[dt.date, list[Arrival]] = defaultdict(list)
    for item in arrivals:
        arrivals_by_day[item.day].append(item)
    results: list[DayOutcome] = []
    cursor = first_day
    while cursor <= last_day:
        day_records = by_day.get(cursor, [])
        day_arrivals = arrivals_by_day.get(cursor, [])
        day_unserved = unserved_by_day.get(cursor, [])
        day_waits = [record.wait_seconds for record in day_records]
        backlog = sum(1 for record in day_records if record.started.date() > cursor) + len(day_unserved)
        slots: list[SlotOutcome] = []
        for slot_minutes in sorted({item.slot_minutes for item in day_arrivals}):
            slot_records = [record for record in day_records if record.arrival_slot_minutes == slot_minutes]
            slot_arrivals = [item for item in day_arrivals if item.slot_minutes == slot_minutes]
            waits = [record.wait_seconds for record in slot_records]
            evidence: dict[str, int] = defaultdict(int)
            for item in slot_arrivals:
                evidence[item.evidence] += 1
                if item.service_evidence != 'empirical':
                    evidence[f'service_{item.service_evidence}'] += 1
            slots.append(SlotOutcome(
                date=cursor.isoformat(), weekday=cursor.weekday(), slot=minutes_label(slot_minutes),
                slot_minutes=slot_minutes, arrivals=len(slot_arrivals), served=len(slot_records),
                same_shift=sum(record.same_shift for record in slot_records),
                deferred=sum(not record.same_shift for record in slot_records),
                waiting=sum(1 for value in waits if value > 0),
                mean_wait_seconds=sum(waits) / len(waits) if waits else 0.0,
                p50_wait_seconds=_quantile(waits, 0.5), p90_wait_seconds=_quantile(waits, 0.9),
                p99_wait_seconds=_quantile(waits, 0.99), max_wait_seconds=max(waits) if waits else 0.0,
                breach_target=sum(1 for value in waits if value > target_seconds),
                breach_promise=sum(1 for value in waits if value > promise_seconds),
                backlog_end=sum(1 for record in slot_records if record.started.date() > cursor),
                evidence=dict(evidence)))
        channel: dict[str, dict[str, Any]] = {}
        for name in sorted({item.channel for item in day_arrivals}):
            channel_records = [record for record in day_records if record.channel == name]
            channel_waits = [record.wait_seconds for record in channel_records]
            channel[name] = {'arrivals': sum(1 for item in day_arrivals if item.channel == name),
                             'served': len(channel_records),
                             'deferred': sum(not record.same_shift for record in channel_records),
                             'p90_wait_seconds': round(_quantile(channel_waits, 0.9), 1),
                             'max_wait_seconds': round(max(channel_waits) if channel_waits else 0.0, 1)}
        results.append(DayOutcome(
            date=cursor, arrivals=len(day_arrivals), served=len(day_records),
            same_shift=sum(record.same_shift for record in day_records),
            deferred=sum(not record.same_shift for record in day_records), unserved=len(day_unserved),
            max_wait_seconds=max(day_waits) if day_waits else 0.0,
            p90_wait_seconds=_quantile(day_waits, 0.9),
            breach_target=sum(1 for value in day_waits if value > target_seconds),
            breach_promise=sum(1 for value in day_waits if value > promise_seconds),
            backlog_end=backlog, channel=channel, slots=tuple(slots)))
        cursor += dt.timedelta(days=1)
    return tuple(results)
