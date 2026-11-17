from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..contracts import fingerprint

UTC = dt.timezone.utc


class WaitRiskError(ValueError):
    """Raised when a wait-risk analysis input cannot be interpreted."""


def parse_timezone(value: Any) -> dt.tzinfo:
    if value in (None, '', 'UTC', 'Z'):
        return UTC
    if isinstance(value, dt.tzinfo):
        return value
    text = str(value).strip()
    match = re.fullmatch(r'([+-])(\d{2}):?(\d{2})', text)
    if match:
        sign = 1 if match.group(1) == '+' else -1
        hours, minutes = int(match.group(2)), int(match.group(3))
        if hours > 23 or minutes > 59:
            raise WaitRiskError('timezone offset out of range')
        return dt.timezone(sign * dt.timedelta(hours=hours, minutes=minutes))
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(text)
    except Exception as exc:  # pragma: no cover - depends on tzdata availability
        raise WaitRiskError(f'unsupported timezone: {value!r}') from exc


def parse_time(value: Any, name: str) -> dt.time:
    if isinstance(value, dt.datetime):
        return value.timetz().replace(tzinfo=None)
    if isinstance(value, dt.time):
        return value.replace(tzinfo=None)
    text = str(value).strip()
    for pattern in ('%H:%M', '%H:%M:%S'):
        try:
            return dt.datetime.strptime(text, pattern).time()
        except ValueError:
            continue
    raise WaitRiskError(f'{name} must be HH:MM or HH:MM:SS')


def require_int(value: Any, name: str, minimum: int = 0, maximum: int = 10 ** 9) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WaitRiskError(f'{name} must be an integer')
    if not minimum <= value <= maximum:
        raise WaitRiskError(f'{name} out of range')
    return value


def require_text(value: Any, name: str, maximum: int = 160) -> str:
    if not isinstance(value, str):
        raise WaitRiskError(f'{name} must be text')
    text = ' '.join(value.split())
    if not text or len(text) > maximum:
        raise WaitRiskError(f'{name} length out of range')
    return text


CHANNEL_ALIASES = {
    'appointment': 'appointment',
    'appointments': 'appointment',
    'booked': 'appointment',
    'booking': 'appointment',
    '预约': 'appointment',
    'walk-in': 'walk_in',
    'walkin': 'walk_in',
    'walk_in': 'walk_in',
    'walk-ins': 'walk_in',
    'walkins': 'walk_in',
    '现场': 'walk_in',
    '现场到达': 'walk_in',
}


def normalize_channel(value: Any) -> str:
    text = require_text(str(value), 'channel', 40).casefold()
    return CHANNEL_ALIASES.get(text, text)


@dataclass(frozen=True)
class Break:
    start: dt.time
    end: dt.time
    reason: str = 'break'

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise WaitRiskError('break end must follow start')

    def as_dict(self) -> dict[str, Any]:
        return {'start': self.start.isoformat(timespec='minutes'), 'end': self.end.isoformat(timespec='minutes'),
                'reason': self.reason}


@dataclass(frozen=True)
class Shift:
    shift_id: str
    weekdays: frozenset[int]
    start: dt.time
    end: dt.time
    seats: int
    breaks: tuple[Break, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.shift_id, 'shift_id', 80)
        if not self.weekdays or any(day < 0 or day > 6 for day in self.weekdays):
            raise WaitRiskError('shift weekdays must be within 0..6')
        if self.end <= self.start:
            raise WaitRiskError('shift end must follow start (overnight shifts are not supported)')
        require_int(self.seats, 'seats')
        ordered = tuple(sorted(self.breaks, key=lambda item: item.start))
        for index, item in enumerate(ordered):
            if item.start < self.start or item.end > self.end:
                raise WaitRiskError(f'break outside shift {self.shift_id}')
            if index and ordered[index - 1].end > item.start:
                raise WaitRiskError(f'breaks overlap in shift {self.shift_id}')
        object.__setattr__(self, 'breaks', ordered)

    def with_seats(self, seats: int) -> 'Shift':
        return Shift(self.shift_id, self.weekdays, self.start, self.end, seats, self.breaks)

    def window(self, day: dt.date, timezone: dt.tzinfo) -> tuple[dt.datetime, dt.datetime]:
        return (dt.datetime.combine(day, self.start, timezone),
                dt.datetime.combine(day, self.end, timezone))

    def as_dict(self, include_seats: bool = True) -> dict[str, Any]:
        result = {'shift_id': self.shift_id, 'weekdays': sorted(self.weekdays),
                  'start': self.start.isoformat(timespec='minutes'), 'end': self.end.isoformat(timespec='minutes'),
                  'breaks': [item.as_dict() for item in self.breaks]}
        if include_seats:
            result['seats'] = self.seats
        return result


@dataclass(frozen=True)
class CapacityInterval:
    start: dt.datetime
    end: dt.datetime
    seats: int
    shift_id: str

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise WaitRiskError('capacity interval end must follow start')
        require_int(self.seats, 'interval seats')


def minutes_label(minutes: int) -> str:
    return f'{minutes // 60:02d}:{minutes % 60:02d}'


def parse_slot(value: Any, slot_minutes: int) -> int:
    if isinstance(value, bool):
        raise WaitRiskError('slot must be text or integer')
    if isinstance(value, int):
        if not 0 <= value <= 23:
            raise WaitRiskError('integer slot must be an hour within 0..23')
        minutes = value * 60
    else:
        text = str(value).strip()
        if re.fullmatch(r'\d{1,2}', text):
            hour = int(text)
            if not 0 <= hour <= 23:
                raise WaitRiskError('slot hour out of range')
            minutes = hour * 60
        else:
            parsed = parse_time(text, 'slot')
            minutes = parsed.hour * 60 + parsed.minute
    if minutes % slot_minutes:
        raise WaitRiskError(f'slot {minutes_label(minutes)} is not aligned to {slot_minutes}-minute grid')
    return minutes


@dataclass(frozen=True)
class ServiceProfile:
    samples_seconds: tuple[int, ...] = ()
    by_channel: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    assumed_mean_seconds: float | None = None
    source: str = 'unknown'

    def __post_init__(self) -> None:
        for value in self.samples_seconds:
            if value <= 0:
                raise WaitRiskError('service samples must be positive')
        for channel, samples in self.by_channel.items():
            normalize_channel(channel)
            for value in samples:
                if value <= 0:
                    raise WaitRiskError('service samples must be positive')
        if self.assumed_mean_seconds is not None and self.assumed_mean_seconds <= 0:
            raise WaitRiskError('assumed service mean must be positive')

    def describe(self, channel: str, minimum_samples: int) -> dict[str, Any]:
        channel_samples = tuple(self.by_channel.get(channel, ()))
        if len(channel_samples) >= minimum_samples:
            return {'channel': channel, 'status': 'empirical', 'n_samples': len(channel_samples),
                    'mean_seconds': _mean(channel_samples), 'source': self.source,
                    'note': 'channel-specific observed service durations'}
        if len(self.samples_seconds) >= minimum_samples:
            return {'channel': channel, 'status': 'empirical_pooled', 'n_samples': len(self.samples_seconds),
                    'mean_seconds': _mean(self.samples_seconds), 'source': self.source,
                    'note': 'channel sample too small; pooled observed durations used'}
        if self.assumed_mean_seconds is not None:
            return {'channel': channel, 'status': 'assumption', 'n_samples': len(channel_samples) + len(self.samples_seconds),
                    'mean_seconds': float(self.assumed_mean_seconds), 'source': self.source,
                    'note': 'insufficient observed durations; configured mean treated as an explicit assumption'}
        available = channel_samples or self.samples_seconds
        if available:
            return {'channel': channel, 'status': 'insufficient_samples', 'n_samples': len(available),
                    'mean_seconds': _mean(available), 'source': self.source,
                    'note': 'too few observed durations; mean of the small sample is an assumption, not a measured peak'}
        return {'channel': channel, 'status': 'missing', 'n_samples': 0, 'mean_seconds': None,
                'source': self.source, 'note': 'no service duration evidence'}

    def draw(self, channel: str, minimum_samples: int, rng: Any) -> tuple[int, bool]:
        description = self.describe(channel, minimum_samples)
        channel_samples = tuple(self.by_channel.get(channel, ()))
        if len(channel_samples) >= minimum_samples:
            return int(rng.choice(channel_samples)), False
        if len(self.samples_seconds) >= minimum_samples:
            return int(rng.choice(self.samples_seconds)), True
        mean = description['mean_seconds']
        if mean is None:
            raise WaitRiskError(f'no service duration evidence for channel {channel!r}')
        return max(1, int(round(mean))), True


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


class ShiftCalendar:
    """Weekly shift pattern with holidays. Breaks carve capacity out of a shift window."""

    def __init__(self, shifts: Iterable[Shift], timezone: dt.tzinfo = UTC,
                 holidays: Iterable[dt.date] = (), closures: Iterable[dt.date] = (),
                 source: str = 'calendar',
                 date_seats: Mapping[dt.date, Mapping[str, int]] | None = None):
        ordered = tuple(sorted(shifts, key=lambda item: item.shift_id))
        if not ordered:
            raise WaitRiskError('calendar requires at least one shift')
        identifiers = [item.shift_id for item in ordered]
        if len(identifiers) != len(set(identifiers)):
            raise WaitRiskError('duplicate shift_id')
        self.shifts = ordered
        self.timezone = timezone
        self.holidays = frozenset(holidays)
        self.closures = frozenset(closures)
        self.source = source
        self.date_seats = {day: dict(values) for day, values in (date_seats or {}).items()}
        known = {item.shift_id for item in ordered}
        for day, values in self.date_seats.items():
            unknown = sorted(set(values) - known)
            if unknown:
                raise WaitRiskError(f'date seat override references unknown shift(s): {", ".join(unknown)}')
            for shift_id, seats in values.items():
                require_int(seats, f'{day.isoformat()} {shift_id} seats')
        self._segments: dict[dt.date, tuple[CapacityInterval, ...]] = {}
        self._windows: dict[dt.date, tuple[CapacityInterval, ...]] = {}

    def seats_for(self, day: dt.date, shift: Shift) -> int:
        return self.date_seats.get(day, {}).get(shift.shift_id, shift.seats)

    def is_open_day(self, day: dt.date) -> bool:
        return day not in self.holidays and day not in self.closures

    def windows(self, day: dt.date) -> tuple[CapacityInterval, ...]:
        cached = self._windows.get(day)
        if cached is not None:
            return cached
        result: list[CapacityInterval] = []
        if self.is_open_day(day):
            for shift in self.shifts:
                seats = self.seats_for(day, shift)
                if day.weekday() not in shift.weekdays or seats <= 0:
                    continue
                start, end = shift.window(day, self.timezone)
                result.append(CapacityInterval(start, end, seats, shift.shift_id))
        ordered = tuple(sorted(result, key=lambda item: (item.start, item.shift_id)))
        self._windows[day] = ordered
        return ordered

    def segments(self, day: dt.date) -> tuple[CapacityInterval, ...]:
        cached = self._segments.get(day)
        if cached is not None:
            return cached
        result: list[CapacityInterval] = []
        if self.is_open_day(day):
            for shift in self.shifts:
                seats = self.seats_for(day, shift)
                if day.weekday() not in shift.weekdays or seats <= 0:
                    continue
                start, end = shift.window(day, self.timezone)
                cursor = start
                for pause in shift.breaks:
                    pause_start = dt.datetime.combine(day, pause.start, self.timezone)
                    pause_end = dt.datetime.combine(day, pause.end, self.timezone)
                    if pause_start > cursor:
                        result.append(CapacityInterval(cursor, pause_start, seats, shift.shift_id))
                    cursor = max(cursor, pause_end)
                if cursor < end:
                    result.append(CapacityInterval(cursor, end, seats, shift.shift_id))
        ordered = tuple(sorted(result, key=lambda item: (item.start, item.end, item.shift_id)))
        self._segments[day] = ordered
        return ordered

    def capacity_at(self, moment: dt.datetime) -> int:
        return sum(item.seats for item in self.segments(moment.date()) if item.start <= moment < item.end)

    def day_metrics(self, day: dt.date) -> dict[str, Any]:
        segments = self.segments(day)
        windows = self.windows(day)
        seat_seconds = sum(item.seats * int((item.end - item.start).total_seconds()) for item in segments)
        window_seat_seconds = sum(item.seats * int((item.end - item.start).total_seconds()) for item in windows)
        open_minutes = _union_minutes([(item.start, item.end) for item in windows])
        return {'date': day.isoformat(), 'open_minutes': open_minutes,
                'seat_minutes': round(seat_seconds / 60, 3),
                'window_seat_minutes': round(window_seat_seconds / 60, 3),
                'break_seat_minutes': round((window_seat_seconds - seat_seconds) / 60, 3),
                'seats': sum({item.shift_id: item.seats for item in windows}.values()) if windows else 0,
                'shifts': [item.shift_id for item in windows]}

    def structural_dict(self, include_seats: bool = False) -> dict[str, Any]:
        result = {'shifts': [item.as_dict(include_seats=include_seats) for item in self.shifts],
                  'holidays': sorted(day.isoformat() for day in self.holidays),
                  'closures': sorted(day.isoformat() for day in self.closures),
                  'timezone': str(self.timezone)}
        if include_seats:
            result['date_seats'] = {day.isoformat(): dict(sorted(values.items()))
                                    for day, values in sorted(self.date_seats.items())}
        return result

    def structural_fingerprint(self, include_seats: bool = False) -> str:
        return fingerprint(self.structural_dict(include_seats))

    def seat_configuration(self) -> dict[str, int]:
        return {item.shift_id: item.seats for item in self.shifts}

    def with_seats(self, seats_by_shift: Mapping[str, int]) -> 'ShiftCalendar':
        shifts = [item.with_seats(require_int(seats_by_shift[item.shift_id], f'{item.shift_id} seats'))
                  for item in self.shifts]
        return ShiftCalendar(shifts, self.timezone, self.holidays, self.closures, self.source,
                             {day: dict(values) for day, values in self.date_seats.items()})


def _union_minutes(intervals: Sequence[tuple[dt.datetime, dt.datetime]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += int((right - left).total_seconds())
            left, right = start, end
    total += int((right - left).total_seconds())
    return round(total / 60, 3)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seats_delta: int = 0
    seat_overrides: Mapping[str, int] = field(default_factory=dict)
    label: str = ''

    def __post_init__(self) -> None:
        require_text(self.scenario_id, 'scenario_id', 80)
        if isinstance(self.seats_delta, bool) or not isinstance(self.seats_delta, int):
            raise WaitRiskError('seats_delta must be an integer')
        for shift_id, seats in self.seat_overrides.items():
            require_text(shift_id, 'seat override shift_id', 80)
            require_int(seats, f'{shift_id} seats')

    def as_dict(self) -> dict[str, Any]:
        return {'scenario_id': self.scenario_id, 'label': self.label,
                'seats_delta': self.seats_delta, 'seat_overrides': dict(self.seat_overrides)}
