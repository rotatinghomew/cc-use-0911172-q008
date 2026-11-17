from __future__ import annotations

import datetime as dt
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..contracts import fingerprint
from .model import (
    Break,
    Scenario,
    ServiceProfile,
    Shift,
    ShiftCalendar,
    WaitRiskError,
    normalize_channel,
    parse_slot,
    parse_time,
    parse_timezone,
    require_int,
    require_text,
)
from .queue import Arrival, simulate_arrivals

FILL_POLICIES = {'exclude', 'include'}
DEFAULT_PROMISE_SECONDS = 1800
DEFAULT_TARGET_SECONDS = 900


def _slot_label(minutes: int) -> str:
    return f'{minutes // 60:02d}:{minutes % 60:02d}'


def _days(horizon: tuple[dt.date, dt.date]) -> Iterable[dt.date]:
    cursor = horizon[0]
    while cursor <= horizon[1]:
        yield cursor
        cursor += dt.timedelta(days=1)


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


def _distribution(waits: Sequence[float], target_seconds: int, promise_seconds: int) -> dict[str, Any]:
    if not waits:
        return {'count': 0, 'mean_seconds': 0.0, 'p50_seconds': 0.0, 'p90_seconds': 0.0,
                'p99_seconds': 0.0, 'max_seconds': 0.0, 'breach_target_rate': 0.0,
                'breach_promise_rate': 0.0, 'buckets': {}}
    buckets = {'0-5m': 0, '5-15m': 0, '15-30m': 0, '30-60m': 0, '60m+': 0}
    for value in waits:
        minutes = value / 60
        if minutes <= 5:
            buckets['0-5m'] += 1
        elif minutes <= 15:
            buckets['5-15m'] += 1
        elif minutes <= 30:
            buckets['15-30m'] += 1
        elif minutes <= 60:
            buckets['30-60m'] += 1
        else:
            buckets['60m+'] += 1
    return {'count': len(waits), 'mean_seconds': round(sum(waits) / len(waits), 1),
            'p50_seconds': round(_quantile(waits, 0.5), 1), 'p90_seconds': round(_quantile(waits, 0.9), 1),
            'p99_seconds': round(_quantile(waits, 0.99), 1), 'max_seconds': round(max(waits), 1),
            'breach_target_rate': round(sum(1 for value in waits if value > target_seconds) / len(waits), 4),
            'breach_promise_rate': round(sum(1 for value in waits if value > promise_seconds) / len(waits), 4),
            'buckets': buckets}


@dataclass(frozen=True)
class Observation:
    observation_id: str
    date: dt.date
    slot_minutes: int
    channel: str
    count: int
    source: str
    site_id: str

    @property
    def weekday(self) -> int:
        return self.date.weekday()


@dataclass(frozen=True)
class DemandBasisRow:
    date: dt.date
    slot_minutes: int
    channel: str
    count: int
    evidence: str
    coverage_days: int
    source_ids: tuple[str, ...] = ()
    source: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {'date': self.date.isoformat(), 'slot': _slot_label(self.slot_minutes),
                'channel': self.channel, 'count': self.count, 'evidence': self.evidence,
                'coverage_days': self.coverage_days,
                'source_ids': list(self.source_ids), 'source': self.source}


class Profile:
    """Empirical weekday × slot counts kept as raw observations (never pre-averaged away)."""

    def __init__(self, observations: Sequence[Observation], slot_minutes: int, minimum_days: int,
                 site_id: str | None = None):
        self.slot_minutes = slot_minutes
        self.minimum_days = minimum_days
        self.site_id = site_id
        self.observations = tuple(sorted(observations, key=lambda item: (item.date, item.slot_minutes, item.channel)))
        by_key: dict[tuple[int, int, str], list[Observation]] = defaultdict(list)
        for item in self.observations:
            by_key[(item.weekday, item.slot_minutes, item.channel)].append(item)
        self.by_key = {key: tuple(values) for key, values in by_key.items()}
        pooled: dict[tuple[int, str], list[Observation]] = defaultdict(list)
        for item in self.observations:
            pooled[(item.slot_minutes, item.channel)].append(item)
        self._pooled = {key: tuple(values) for key, values in pooled.items()}

    @property
    def date_range(self) -> tuple[str | None, str | None]:
        if not self.observations:
            return (None, None)
        return (self.observations[0].date.isoformat(), self.observations[-1].date.isoformat())

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({item.source for item in self.observations}))

    def coverage(self, weekday: int, slot_minutes: int) -> dict[str, Any]:
        entries = [item for key, values in self.by_key.items() if key[0] == weekday and key[1] == slot_minutes
                   for item in values]
        channels = sorted({item.channel for item in entries})
        days = len({item.date for item in entries})
        if not entries:
            return {'weekday': weekday, 'slot': _slot_label(slot_minutes), 'days_observed': 0,
                    'status': 'missing', 'channels': [], 'total': 0, 'source_ids': []}
        total = sum(item.count for item in entries)
        status = 'empirical' if days >= self.minimum_days else 'low_sample'
        return {'weekday': weekday, 'slot': _slot_label(slot_minutes), 'days_observed': days,
                'status': status, 'channels': channels, 'total': total,
                'source_ids': sorted(item.observation_id for item in entries)}

    def estimate(self, weekday: int, slot_minutes: int, channel: str, fill_policy: str,
                 rng: random.Random | None) -> tuple[int, str, int, tuple[str, ...]]:
        """Return (count, evidence, days_observed, source_ids).

        ``rng`` present → bootstrap draw from the raw per-day observations;
        ``rng`` absent → observed mean (still flagged low_sample when under threshold).
        """
        entries = self.by_key.get((weekday, slot_minutes, channel), ())
        days = len({item.date for item in entries})
        if entries:
            if rng is not None:
                chosen = rng.choice(entries)
                count = chosen.count
                ids = (chosen.observation_id,)
            else:
                count = int(math.floor(sum(item.count for item in entries) / len(entries) + 0.5))
                ids = tuple(sorted(item.observation_id for item in entries))
            return count, ('empirical' if days >= self.minimum_days else 'low_sample'), days, ids
        if fill_policy == 'exclude':
            return 0, 'missing', 0, ()
        pooled = self._pooled.get((slot_minutes, channel), ())
        if not pooled:
            return 0, 'missing', 0, ()
        if rng is not None:
            chosen = rng.choice(pooled)
            return chosen.count, 'imputed_pooled', 0, (chosen.observation_id,)
        values = [item.count for item in pooled]
        return int(math.floor(sum(values) / len(values) + 0.5)), 'imputed_pooled', 0, tuple(
            sorted(item.observation_id for item in pooled))

    def all_channels(self) -> tuple[str, ...]:
        return tuple(sorted({item.channel for item in self.observations}))

    def observed_peak(self, cells: set[tuple[int, int]] | None = None) -> dict[str, Any] | None:
        """Peak built only from weekday×slot×channel cells meeting the minimum observation days."""
        candidates = []
        for (weekday, slot, channel), entries in self.by_key.items():
            if cells is not None and (weekday, slot) not in cells:
                continue
            days = len({item.date for item in entries})
            if days < self.minimum_days:
                continue
            candidates.append({'weekday': weekday, 'slot': _slot_label(slot), 'channel': channel,
                               'mean': round(sum(item.count for item in entries) / days, 3),
                               'days_observed': days,
                               'source_ids': sorted(item.observation_id for item in entries)})
        if not candidates:
            return None
        peak = max(candidates, key=lambda item: (item['mean'], item['slot']))
        ranked = sorted(candidates, key=lambda item: (-item['mean'], item['weekday'], item['slot'], item['channel']))
        return {'peak': peak,
                'qualified_cells': len(ranked),
                'top_cells': ranked[:20],
                'note': '仅统计达到最小观测天数的 工作日×时段×渠道 单元；回填值不参与'}


class WaitRiskAnalyzer:
    """Slot-level waiting-risk analysis layered on top of the daily capacity forecast.

    The analyzer never rewrites the daily forecast; it adds a second view and reports
    where and why the two views differ.
    """

    def analyze(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        site_id = require_text(dataset.get('site_id', 'site'), 'site_id', 96)
        timezone = parse_timezone(dataset.get('timezone'))
        slot_minutes = require_int(int(dataset.get('slot_minutes', 60)), 'slot_minutes', 5, 240)
        if 24 * 60 % slot_minutes:
            raise WaitRiskError('slot_minutes must divide a day evenly')
        horizon = self._horizon(dataset)
        profile, observations = self._profile(dataset, slot_minutes, site_id)
        service = self._service_profile(dataset)
        calendar = self._calendar(dataset.get('calendar', {}), timezone, site_id)
        fill_policy = str(dataset.get('fill_policy', 'exclude'))
        if fill_policy not in FILL_POLICIES:
            raise WaitRiskError('fill_policy must be exclude or include')
        minimum_samples = require_int(int(dataset.get('minimum_samples', 5)), 'minimum_samples', 1, 1000)
        promise_seconds = require_int(int(dataset.get('promise_seconds', DEFAULT_PROMISE_SECONDS)),
                                      'promise_seconds', 1, 24 * 3600)
        target_seconds = require_int(int(dataset.get('target_seconds', DEFAULT_TARGET_SECONDS)),
                                     'target_seconds', 1, 24 * 3600)
        if target_seconds > promise_seconds:
            raise WaitRiskError('target_seconds cannot exceed promise_seconds')
        replications = require_int(int(dataset.get('replications', 200)), 'replications', 1, 5000)
        seed = int(dataset.get('seed', 20261117))
        discipline = str(dataset.get('discipline', 'fifo'))
        if discipline not in {'fifo', 'appointment_first'}:
            raise WaitRiskError('discipline must be fifo or appointment_first')
        scenarios = self._scenarios(dataset.get('scenarios'))
        baseline = self._baseline(dataset.get('baseline'))

        deterministic = self._basis(profile, horizon, calendar, slot_minutes, fill_policy, None)
        serialized = [row.as_dict() for row in deterministic]
        demand_fingerprint = fingerprint(serialized)
        calendar_fingerprint = calendar.structural_fingerprint()

        runs = self._scenario_runs(
            calendar, profile, horizon, slot_minutes, fill_policy, minimum_samples,
            service, scenarios, replications, seed, discipline, deterministic,
            target_seconds, promise_seconds, demand_fingerprint)
        excluded = self._closed_slot_demand(profile, horizon, calendar, slot_minutes, fill_policy)
        coverage = self._coverage(profile, horizon, slot_minutes, calendar)
        observed_peak = profile.observed_peak(_open_cells(calendar, horizon, slot_minutes))
        uncertainty = self._uncertainty(profile, coverage, observations, service, minimum_samples,
                                        fill_policy, slot_minutes, horizon, calendar)
        comparison = self._comparison(runs, demand_fingerprint, calendar_fingerprint)
        reconciliation = self._reconcile(runs[0], baseline, deterministic, excluded)

        calendar_days = [calendar.day_metrics(day) for day in _days(horizon)]
        return {
            'product': '人才驿站与服务窗口逐时段等待风险分析',
            'site_id': site_id,
            'timezone': str(timezone),
            'slot_minutes': slot_minutes,
            'horizon': {'start': horizon[0].isoformat(), 'end': horizon[1].isoformat(),
                        'days': (horizon[1] - horizon[0]).days + 1},
            'policy': {'target_seconds': target_seconds, 'promise_seconds': promise_seconds,
                       'fill_policy': fill_policy, 'discipline': discipline,
                       'minimum_samples': minimum_samples, 'minimum_observation_days': profile.minimum_days,
                       'replications': replications, 'seed': seed},
            'lineage': {
                'site_id': site_id,
                'demand_sources': [{'source': source, 'site_id': profile.site_id,
                                    'observations': sum(1 for item in observations if item.source == source),
                                    'date_range': [min(item.date for item in observations if item.source == source).isoformat(),
                                                   max(item.date for item in observations if item.source == source).isoformat()]}
                                   for source in profile.sources],
                'demand_fingerprint': demand_fingerprint,
                'demand_rows': len(deterministic),
                'calendar_source': calendar.source,
                'calendar_fingerprint': calendar_fingerprint,
                'seat_configuration': calendar.seat_configuration(),
                'calendar_days': calendar_days,
                'service_source': service.source,
            },
            'coverage': coverage,
            'observed_peak': observed_peak,
            'calendar_metrics': {
                'days': calendar_days,
                'break_seat_minutes': round(sum(day['break_seat_minutes'] for day in calendar_days), 3),
                'seat_minutes': round(sum(day['seat_minutes'] for day in calendar_days), 3),
                'window_seat_minutes': round(sum(day['window_seat_minutes'] for day in calendar_days), 3),
                'note': 'seat_minutes 已扣除休息；window_seat_minutes 为按班次名义时长、未扣休息的上限口径',
            },
            'scenarios': runs,
            'comparison': comparison,
            'baseline_reconciliation': reconciliation,
            'excluded_demand': excluded,
            'uncertainty': uncertainty,
            'assumptions': self._assumptions(fill_policy, replications, discipline),
            'demand_basis': serialized,
        }

    # ------------------------------------------------------------------ parsing
    def _horizon(self, dataset: Mapping[str, Any]) -> tuple[dt.date, dt.date]:
        raw = dataset.get('horizon') or {}
        observations = (dataset.get('arrivals') or {}).get('observations') or []
        if raw.get('start') and raw.get('end'):
            start = dt.date.fromisoformat(str(raw['start']))
            end = dt.date.fromisoformat(str(raw['end']))
        elif observations:
            last = max(dt.date.fromisoformat(str(row['date'])) for row in observations)
            start = last + dt.timedelta(days=1)
            end = start + dt.timedelta(days=6)
        else:
            raise WaitRiskError('horizon or arrival observations are required')
        if end < start:
            raise WaitRiskError('horizon end precedes start')
        if (end - start).days > 366:
            raise WaitRiskError('horizon longer than a year')
        return start, end

    def _profile(self, dataset: Mapping[str, Any], slot_minutes: int,
                 site_id: str) -> tuple[Profile, tuple[Observation, ...]]:
        arrivals = dataset.get('arrivals') or {}
        rows = arrivals.get('observations') or []
        minimum_days = require_int(int(arrivals.get('minimum_days', 2)), 'minimum_days', 1, 365)
        seen_ids: set[str] = set()
        seen_keys: set[tuple[dt.date, int, str]] = set()
        observations: list[Observation] = []
        for row in rows:
            identifier = require_text(row.get('observation_id') or row.get('id'), 'observation_id', 96)
            if identifier in seen_ids:
                raise WaitRiskError(f'duplicate observation_id {identifier}')
            seen_ids.add(identifier)
            day = dt.date.fromisoformat(str(row['date']))
            if 'slot' not in row and 'slot_start' not in row:
                raise WaitRiskError(f'observation {identifier} needs a slot')
            slot = parse_slot(row.get('slot', row.get('slot_start')), slot_minutes)
            channel = normalize_channel(row.get('channel', 'walk_in'))
            count = require_int(int(row.get('count', row.get('value', 0))), f'{identifier} count')
            source = require_text(row.get('source', arrivals.get('source', 'arrivals')), 'source', 160)
            row_site = require_text(row.get('site_id', site_id), 'site_id', 96)
            if row_site != site_id:
                raise WaitRiskError(f'observation {identifier} belongs to site {row_site}, not {site_id}')
            key = (day, slot, channel)
            if key in seen_keys:
                raise WaitRiskError(f'duplicate observation for {day.isoformat()} {_slot_label(slot)} {channel}')
            seen_keys.add(key)
            observations.append(Observation(identifier, day, slot, channel, count, source, row_site))
        if not observations:
            raise WaitRiskError('at least one arrival observation is required')
        return Profile(observations, slot_minutes, minimum_days, site_id), tuple(observations)

    def _service_profile(self, dataset: Mapping[str, Any]) -> ServiceProfile:
        raw = dataset.get('service') or {}
        samples = tuple(require_int(int(value), 'service sample', 1, 3600) for value in raw.get('samples_seconds', ()))
        by_channel = {}
        for channel, values in (raw.get('by_channel') or {}).items():
            name = normalize_channel(channel)
            by_channel[name] = tuple(require_int(int(value), f'{name} service sample', 1, 3600) for value in values)
        assumed = raw.get('assumed_mean_seconds')
        if assumed is not None:
            assumed = float(assumed)
            if assumed <= 0:
                raise WaitRiskError('assumed_mean_seconds must be positive')
        return ServiceProfile(samples, by_channel, assumed,
                              require_text(raw.get('source', 'service-log'), 'service source', 160))

    def _calendar(self, raw: Mapping[str, Any], timezone: dt.tzinfo, site_id: str) -> ShiftCalendar:
        shifts = []
        for row in raw.get('shifts') or []:
            breaks = tuple(Break(parse_time(item.get('start'), 'break start'), parse_time(item.get('end'), 'break end'),
                                 str(item.get('reason', 'break'))) for item in row.get('breaks', []))
            weekdays = frozenset(int(day) for day in row.get('weekdays', range(5)))
            shifts.append(Shift(require_text(row.get('shift_id'), 'shift_id', 80), weekdays,
                                parse_time(row.get('start'), 'shift start'), parse_time(row.get('end'), 'shift end'),
                                require_int(int(row.get('seats', 1)), 'seats'), breaks))
        holidays = {dt.date.fromisoformat(str(value)) for value in raw.get('holidays', [])}
        closures = {dt.date.fromisoformat(str(value)) for value in raw.get('closures', [])}
        date_seats = {dt.date.fromisoformat(str(day)): {str(key): int(value) for key, value in values.items()}
                      for day, values in (raw.get('date_seats') or {}).items()}
        return ShiftCalendar(shifts, timezone, holidays, closures,
                             require_text(raw.get('source', f'{site_id}-roster'), 'calendar source', 160), date_seats)

    def _scenarios(self, raw: Any) -> tuple[Scenario, ...]:
        if not raw:
            return (Scenario('as-is', 0, {}, 'current roster'),)
        scenarios = []
        for row in raw:
            scenarios.append(Scenario(require_text(row.get('scenario_id'), 'scenario_id', 80),
                                      int(row.get('seats_delta', 0)),
                                      {str(key): int(value) for key, value in (row.get('seat_overrides') or {}).items()},
                                      str(row.get('label', ''))))
        return tuple(scenarios)

    def _baseline(self, raw: Any) -> dict[str, Any] | None:
        if not raw:
            return None
        rows = []
        for row in raw.get('daily', []):
            rows.append({'date': dt.date.fromisoformat(str(row['date'])),
                         'demand': float(row.get('demand', 0)),
                         'agents': None if row.get('agents') is None else int(row['agents'])})
        return {'source': str(raw.get('source', 'daily-capacity-forecast')),
                'daily': rows, 'notes': list(raw.get('notes', []))}

    # ------------------------------------------------------------------ demand basis
    def _basis(self, profile: Profile, horizon: tuple[dt.date, dt.date], calendar: ShiftCalendar,
               slot_minutes: int, fill_policy: str, rng: random.Random | None) -> tuple[DemandBasisRow, ...]:
        rows, _ = self._basis_and_closed(profile, horizon, calendar, slot_minutes, fill_policy, rng)
        return rows

    def _basis_and_closed(self, profile: Profile, horizon: tuple[dt.date, dt.date], calendar: ShiftCalendar,
                          slot_minutes: int, fill_policy: str,
                          rng: random.Random | None) -> tuple[tuple[DemandBasisRow, ...], dict[str, Any]]:
        """Build the per-day/per-slot demand basis.

        Demand falling entirely outside every shift window (closed day, lunch gap, after close)
        is *not* silently served and is *not* counted as capacity; it is returned separately.
        """
        rows: list[DemandBasisRow] = []
        closed_rows: list[dict[str, Any]] = []
        channels = profile.all_channels()
        source_by_id = {item.observation_id: item.source for item in profile.observations}
        for day in _days(horizon):
            open_windows = calendar.windows(day)
            for slot in range(0, 24 * 60, slot_minutes):
                slot_start = dt.datetime.combine(day, dt.time.min, calendar.timezone) + dt.timedelta(minutes=slot)
                slot_end = slot_start + dt.timedelta(minutes=slot_minutes)
                is_open = any(item.start < slot_end and item.end > slot_start for item in open_windows)
                for channel in channels:
                    count, evidence, days, ids = profile.estimate(day.weekday(), slot, channel, fill_policy, rng)
                    if count <= 0 and evidence == 'missing':
                        continue
                    source = ','.join(sorted({source_by_id[identifier] for identifier in ids}))
                    if is_open:
                        rows.append(DemandBasisRow(day, slot, channel, count, evidence, days, ids, source))
                    else:
                        closed_rows.append({'date': day.isoformat(), 'slot': _slot_label(slot),
                                            'channel': channel, 'count': count, 'evidence': evidence,
                                            'source': source,
                                            'reason': 'closed_day' if not open_windows else 'outside_shift_window'})
        return tuple(rows), {'rows': closed_rows, 'total': sum(item['count'] for item in closed_rows),
                             'by_reason': _reason_totals(closed_rows)}

    def _closed_slot_demand(self, profile: Profile, horizon: tuple[dt.date, dt.date], calendar: ShiftCalendar,
                            slot_minutes: int, fill_policy: str) -> dict[str, Any]:
        _, closed = self._basis_and_closed(profile, horizon, calendar, slot_minutes, fill_policy, None)
        return closed

    def _arrivals(self, basis: Iterable[DemandBasisRow], calendar: ShiftCalendar,
                  slot_minutes: int, service: ServiceProfile, minimum_samples: int,
                  rng: random.Random) -> tuple[Arrival, ...]:
        arrivals: list[Arrival] = []
        counter = 0
        slot_seconds = slot_minutes * 60
        for row in sorted(basis, key=lambda item: (item.date, item.slot_minutes, item.channel)):
            if row.count <= 0:
                continue
            base = dt.datetime.combine(row.date, dt.time.min, calendar.timezone) + dt.timedelta(minutes=row.slot_minutes)
            for index in range(row.count):
                counter += 1
                # Appointments are spread evenly across the slot (booked start minutes are not
                # observed, so the slot is the resolution limit); walk-ins arrive at random
                # minutes inside the slot with a fixed seed for reproducibility.
                if row.channel == 'appointment':
                    offset = int(index * slot_seconds / row.count)
                else:
                    offset = rng.randrange(0, max(slot_seconds, 1))
                seconds, pooled = service.draw(row.channel, minimum_samples, rng)
                arrivals.append(Arrival(f'{row.date.isoformat()}-{_slot_label(row.slot_minutes)}-{row.channel}-{counter:05d}',
                                        row.date, base + dt.timedelta(seconds=offset), row.channel,
                                        row.slot_minutes, seconds, row.evidence,
                                        'empirical_pooled' if pooled else 'empirical', row.date))
        return tuple(arrivals)

    # ------------------------------------------------------------------ simulations
    def _scenario_runs(self, calendar: ShiftCalendar, profile: Profile, horizon: tuple[dt.date, dt.date],
                       slot_minutes: int, fill_policy: str, minimum_samples: int, service: ServiceProfile,
                       scenarios: Sequence[Scenario], replications: int, seed: int, discipline: str,
                       deterministic: Sequence[DemandBasisRow], target_seconds: int, promise_seconds: int,
                       demand_fingerprint: str) -> list[dict[str, Any]]:
        master = random.Random(seed)
        draws = [self._basis(profile, horizon, calendar, slot_minutes, fill_policy, master)
                 for _ in range(replications)]
        deterministic_arrivals = self._arrivals(deterministic, calendar, slot_minutes, service,
                                                minimum_samples, random.Random(seed))
        reports: list[dict[str, Any]] = []
        for scenario in scenarios:
            configured = _apply_scenario(calendar, scenario)
            fixed = simulate_arrivals(configured, deterministic_arrivals, horizon[0], horizon[1], discipline,
                                      target_seconds=target_seconds, promise_seconds=promise_seconds)
            slot_stats: dict[tuple[int, int], dict[str, Any]] = defaultdict(_slot_bucket)
            channel_stats: dict[str, dict[str, Any]] = defaultdict(_channel_bucket)
            day_stats: dict[dt.date, dict[str, Any]] = defaultdict(_day_bucket)
            waits: list[float] = []
            same_shift_waits: list[float] = []
            deferred_delays: list[float] = []
            empirical_waits: list[float] = []
            breach_slots: dict[tuple[int, int], int] = defaultdict(int)
            deferral_flows: dict[tuple[str, str], int] = defaultdict(int)
            totals = defaultdict(int)
            for replication in range(replications):
                rep_rng = random.Random(f'{seed}-service-{replication}')
                arrivals = self._arrivals(draws[replication], configured, slot_minutes, service,
                                          minimum_samples, rep_rng)
                result = simulate_arrivals(configured, arrivals, horizon[0], horizon[1], discipline,
                                           target_seconds=target_seconds, promise_seconds=promise_seconds)
                totals['arrivals'] += result.arrivals
                totals['served'] += len(result.records)
                totals['unserved'] += len(result.unserved)
                breached: set[tuple[int, int]] = set()
                for record in result.records:
                    key = (record.arrival_day.weekday(), record.arrival_slot_minutes)
                    bucket = slot_stats[key]
                    bucket['arrivals'] += 1
                    bucket['served'] += 1
                    bucket['same_shift'] += int(record.same_shift)
                    bucket['deferred'] += int(not record.same_shift)
                    bucket['waiting'] += int(record.wait_seconds > 0)
                    bucket['waits'].append(record.wait_seconds)
                    bucket['breach_target'] += int(record.wait_seconds > target_seconds)
                    bucket['breach_promise'] += int(record.wait_seconds > promise_seconds)
                    if record.wait_seconds > promise_seconds:
                        breached.add(key)
                    bucket['evidence'][record.demand_evidence] += 1
                    if record.service_evidence != 'empirical':
                        bucket['evidence'][f'service_{record.service_evidence}'] += 1
                    if record.same_shift:
                        target = 'same_shift'
                    else:
                        target = f'{record.served_shift_key[0].isoformat()}:{record.served_shift_key[1]}'
                        origin = (f'{record.arrival_shift_key[0].isoformat()}:{record.arrival_shift_key[1]}'
                                  if record.arrival_shift_key else 'outside_schedule')
                        deferral_flows[(origin, target)] += 1
                    bucket['deferred_to'][target] += 1
                    waits.append(record.wait_seconds)
                    (same_shift_waits if record.same_shift else deferred_delays).append(record.wait_seconds)
                    if record.demand_evidence != 'imputed_pooled':
                        empirical_waits.append(record.wait_seconds)
                    channel_bucket = channel_stats[record.channel]
                    channel_bucket['arrivals'] += 1
                    channel_bucket['served'] += 1
                    channel_bucket['deferred'] += int(not record.same_shift)
                    channel_bucket['waits'].append(record.wait_seconds)
                    channel_bucket['breach_promise'] += int(record.wait_seconds > promise_seconds)
                    day = day_stats[record.arrival_day]
                    day['arrivals'] += 1
                    day['served'] += 1
                    day['same_shift'] += int(record.same_shift)
                    day['deferred'] += int(not record.same_shift)
                    day['waits'].append(record.wait_seconds)
                    if record.started.date() != record.arrival_day:
                        day['backlog_end'] += 1
                for key in breached:
                    breach_slots[key] += 1
                for item in result.unserved:
                    key = (item.day.weekday(), item.slot_minutes)
                    slot_stats[key]['arrivals'] += 1
                    channel_stats[item.channel]['arrivals'] += 1
                    channel_stats[item.channel]['unserved'] += 1
                    day_stats[item.day]['arrivals'] += 1
                    day_stats[item.day]['unserved'] += 1
                    day_stats[item.day]['backlog_end'] += 1
            reports.append({
                'scenario': scenario.as_dict(),
                'seat_configuration': configured.seat_configuration(),
                'date_seat_overrides': {day.isoformat(): dict(values) for day, values in sorted(configured.date_seats.items())},
                'throughput': {
                    'arrivals': totals['arrivals'], 'served': totals['served'],
                    'unserved': totals['unserved'],
                    'deferred': sum(bucket['deferred'] for bucket in slot_stats.values()),
                    'deferral_rate': _rate(sum(bucket['deferred'] for bucket in slot_stats.values()), totals['served']),
                    'unserved_rate': _rate(totals['unserved'], totals['arrivals']),
                },
                'wait_distribution': _distribution(waits, target_seconds, promise_seconds),
                'same_shift_wait_distribution': _distribution(same_shift_waits, target_seconds, promise_seconds),
                'deferred_delay_distribution': _distribution(deferred_delays, target_seconds, promise_seconds),
                'wait_distribution_observed_demand_only': _distribution(empirical_waits, target_seconds, promise_seconds),
                'deferral_flows': [{'from': origin, 'to': target, 'count': count}
                                   for (origin, target), count in sorted(deferral_flows.items())],
                'slot_table': self._slot_table(slot_stats, breach_slots, replications),
                'channel_table': self._channel_table(channel_stats, target_seconds, promise_seconds),
                'risk_windows': self._risk_windows(slot_stats, breach_slots, replications,
                                                   target_seconds, promise_seconds),
                'daily': self._daily_table(day_stats, horizon),
                'deterministic_run': fixed.as_dict(),
                'demand_fingerprint': demand_fingerprint,
                'calendar_fingerprint': configured.structural_fingerprint(),
                'scenario_calendar_fingerprint': configured.structural_fingerprint(include_seats=True),
            })
        return reports

    def _slot_table(self, slot_stats: Mapping[tuple[int, int], Mapping[str, Any]],
                    breach_slots: Mapping[tuple[int, int], int], replications: int) -> list[dict[str, Any]]:
        rows = []
        for (weekday, slot), bucket in sorted(slot_stats.items()):
            served = bucket['served']
            rows.append({
                'weekday': weekday, 'slot': _slot_label(slot),
                'arrivals': bucket['arrivals'], 'served': served,
                'same_shift': bucket['same_shift'], 'deferred': bucket['deferred'],
                'deferral_rate': _rate(bucket['deferred'], served),
                'waiting_share': _rate(bucket['waiting'], served),
                'mean_wait_seconds': _mean(bucket['waits']),
                'p50_wait_seconds': round(_quantile(bucket['waits'], 0.5), 1),
                'p90_wait_seconds': round(_quantile(bucket['waits'], 0.9), 1),
                'p99_wait_seconds': round(_quantile(bucket['waits'], 0.99), 1),
                'max_wait_seconds': round(max(bucket['waits']), 1) if bucket['waits'] else 0.0,
                'breach_target': bucket['breach_target'], 'breach_promise': bucket['breach_promise'],
                'breach_promise_probability': _rate(breach_slots.get((weekday, slot), 0), replications),
                'deferred_to': {key: value for key, value in sorted(bucket['deferred_to'].items())
                                if key != 'same_shift'},
                'evidence': dict(bucket['evidence']),
            })
        return rows

    def _channel_table(self, channel_stats: Mapping[str, Mapping[str, Any]],
                       target_seconds: int, promise_seconds: int) -> list[dict[str, Any]]:
        rows = []
        for channel, bucket in sorted(channel_stats.items()):
            rows.append({
                'channel': channel, 'arrivals': bucket['arrivals'], 'served': bucket['served'],
                'unserved': bucket['unserved'], 'deferred': bucket['deferred'],
                'deferral_rate': _rate(bucket['deferred'], bucket['served']),
                'p50_wait_seconds': round(_quantile(bucket['waits'], 0.5), 1),
                'p90_wait_seconds': round(_quantile(bucket['waits'], 0.9), 1),
                'p99_wait_seconds': round(_quantile(bucket['waits'], 0.99), 1),
                'max_wait_seconds': round(max(bucket['waits']), 1) if bucket['waits'] else 0.0,
                'breach_promise_rate': _rate(bucket['breach_promise'], bucket['served']),
            })
        return rows

    def _risk_windows(self, slot_stats: Mapping[tuple[int, int], Mapping[str, Any]],
                      breach_slots: Mapping[tuple[int, int], int], replications: int,
                      target_seconds: int, promise_seconds: int) -> dict[str, Any]:
        """Slots where the promise is crossed with non-trivial probability, ranked by exposure."""
        breaches = []
        for (weekday, slot), bucket in slot_stats.items():
            probability = _rate(breach_slots.get((weekday, slot), 0), replications)
            if probability <= 0:
                continue
            breaches.append({'weekday': weekday, 'slot': _slot_label(slot),
                             'promise_breach_probability': probability,
                             'breach_promise': bucket['breach_promise'],
                             'deferred': bucket['deferred'],
                             'p90_wait_seconds': round(_quantile(bucket['waits'], 0.9), 1)})
        breaches.sort(key=lambda item: (-item['promise_breach_probability'], item['weekday'], item['slot']))
        return {'slots': breaches,
                'count': len(breaches),
                'promise_breach_events': sum(bucket['breach_promise'] for bucket in slot_stats.values()),
                'policy': {'target_seconds': target_seconds, 'promise_seconds': promise_seconds}}

    def _daily_table(self, day_stats: Mapping[dt.date, Mapping[str, Any]],
                     horizon: tuple[dt.date, dt.date]) -> list[dict[str, Any]]:
        rows = []
        for day in _days(horizon):
            bucket = day_stats.get(day)
            if bucket is None:
                rows.append({'date': day.isoformat(), 'weekday': day.weekday(), 'arrivals': 0, 'served': 0,
                             'same_shift': 0, 'deferred': 0, 'unserved': 0, 'backlog_end': 0,
                             'p90_wait_seconds': 0.0, 'max_wait_seconds': 0.0, 'open': False})
            else:
                rows.append({'date': day.isoformat(), 'weekday': day.weekday(),
                             'arrivals': bucket['arrivals'], 'served': bucket['served'],
                             'same_shift': bucket['same_shift'], 'deferred': bucket['deferred'],
                             'unserved': bucket['unserved'], 'backlog_end': bucket['backlog_end'],
                             'p90_wait_seconds': round(_quantile(bucket['waits'], 0.9), 1),
                             'max_wait_seconds': round(max(bucket['waits']), 1) if bucket['waits'] else 0.0,
                             'open': True})
        return rows

    # ------------------------------------------------------------------ diagnostics
    def _coverage(self, profile: Profile, horizon: tuple[dt.date, dt.date], slot_minutes: int,
                  calendar: ShiftCalendar) -> dict[str, Any]:
        open_cells = _open_cells(calendar, horizon, slot_minutes)
        slots = [profile.coverage(weekday, slot) for weekday, slot in sorted(open_cells)]
        relevant = slots
        counts = defaultdict(int)
        for entry in relevant:
            counts[entry['status']] += 1
        return {'slots': relevant,
                'summary': {'empirical': counts.get('empirical', 0), 'low_sample': counts.get('low_sample', 0),
                            'missing': counts.get('missing', 0), 'open_cells': sum(counts.values())},
                'minimum_observation_days': profile.minimum_days,
                'note': '只有达到最小观测天数的时段才支持“实测高峰”结论'}

    def _uncertainty(self, profile: Profile, coverage: Mapping[str, Any], observations: Sequence[Observation],
                     service: ServiceProfile, minimum_samples: int, fill_policy: str,
                     slot_minutes: int, horizon: tuple[dt.date, dt.date],
                     calendar: ShiftCalendar) -> dict[str, Any]:
        missing = [entry for entry in coverage['slots'] if entry['status'] == 'missing']
        low = [entry for entry in coverage['slots'] if entry['status'] == 'low_sample']
        empirical = [entry for entry in coverage['slots'] if entry['status'] == 'empirical']
        channels = profile.all_channels()
        service_evidence = [service.describe(channel, minimum_samples) for channel in channels]
        open_cells = _open_cells(calendar, horizon, slot_minutes)
        imputed = sum(row[0] for row in
                      (profile.estimate(day.weekday(), slot, channel, fill_policy, None)
                       for day in _days(horizon)
                       for slot in range(0, 24 * 60, slot_minutes)
                       if (day.weekday(), slot) in open_cells
                       for channel in channels)
                      if row[1] == 'imputed_pooled')
        observed_days = len({item.date for item in observations})
        return {
            'coverage': {'empirical_slots': len(empirical), 'low_sample_slots': len(low),
                         'missing_slots': len(missing), 'minimum_observation_days': profile.minimum_days},
            'missing_slots': missing,
            'low_sample_slots': low,
            'service_evidence': service_evidence,
            'observation_window': {'distinct_observed_days': observed_days,
                                   'horizon_days': (horizon[1] - horizon[0]).days + 1},
            'imputed_demand_count': imputed,
            'notes': self._uncertainty_notes(fill_policy, missing, low, service_evidence, imputed),
        }

    def _uncertainty_notes(self, fill_policy: str, missing: Sequence[Mapping[str, Any]],
                           low: Sequence[Mapping[str, Any]], service_evidence: Sequence[Mapping[str, Any]],
                           imputed: int) -> list[str]:
        notes = []
        if missing:
            notes.append(f'{len(missing)} 个工作日×时段没有观测，未计入实测高峰；'
                         + ('模拟中也不补值（fill_policy=exclude）。' if fill_policy == 'exclude'
                            else '模拟中使用同槽位跨工作日回填，回填需求单列 imputed_pooled，不参与实测高峰。'))
        if low:
            notes.append(f'{len(low)} 个工作日×时段观测天数低于门槛，等待分布按小样本自举，单独列入低样本不确定性。')
        unsupported = [item for item in service_evidence
                       if item['status'] in {'assumption', 'insufficient_samples', 'missing'}]
        if unsupported:
            notes.append(f'{len(unsupported)} 个渠道的服务时长证据不足，相应等待结论依赖显式假设而非实测分布。')
        if imputed:
            notes.append(f'回填需求合计 {imputed} 人次（分析窗口内），仅用于敏感性，不进入实测高峰。')
        if not notes:
            notes.append('所有使用时段均满足最小观测天数，服务时长样本充足。')
        return notes

    def _comparison(self, reports: Sequence[Mapping[str, Any]], demand_fingerprint: str,
                    calendar_fingerprint: str) -> dict[str, Any]:
        baseline = reports[0]
        rows = []
        for index, report in enumerate(reports):
            rows.append({
                'scenario_id': report['scenario']['scenario_id'],
                'label': report['scenario']['label'],
                'seats': report['seat_configuration'],
                'arrivals': report['throughput']['arrivals'],
                'served': report['throughput']['served'],
                'deferred': report['throughput']['deferred'],
                'unserved': report['throughput']['unserved'],
                'p90_wait_seconds': report['wait_distribution']['p90_seconds'],
                'p99_wait_seconds': report['wait_distribution']['p99_seconds'],
                'max_wait_seconds': report['wait_distribution']['max_seconds'],
                'breach_promise_rate': report['wait_distribution']['breach_promise_rate'],
                'delta_vs_first': {
                    'deferred': report['throughput']['deferred'] - baseline['throughput']['deferred'],
                    'unserved': report['throughput']['unserved'] - baseline['throughput']['unserved'],
                    'p90_wait_seconds': round(report['wait_distribution']['p90_seconds']
                                              - baseline['wait_distribution']['p90_seconds'], 1),
                    'p99_wait_seconds': round(report['wait_distribution']['p99_seconds']
                                              - baseline['wait_distribution']['p99_seconds'], 1),
                    'breach_promise_rate': round(report['wait_distribution']['breach_promise_rate']
                                                 - baseline['wait_distribution']['breach_promise_rate'], 4),
                } if index else None,
            })
        return {'demand_held_constant': all(item['demand_fingerprint'] == demand_fingerprint for item in reports),
                'calendar_held_constant': all(item['calendar_fingerprint'] == calendar_fingerprint for item in reports),
                'demand_fingerprint': demand_fingerprint,
                'calendar_fingerprint': calendar_fingerprint,
                'rows': rows,
                'note': '所有方案共用同批到达观测、同一缺档规则与同一班表结构；只有席位配置变化。'}

    def _reconcile(self, report: Mapping[str, Any], baseline: Mapping[str, Any] | None,
                   basis: Sequence[DemandBasisRow], excluded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        by_day = defaultdict(float)
        for row in basis:
            by_day[row.date] += row.count
        closed_by_day = defaultdict(float)
        for row in (excluded or {}).get('rows', []):
            closed_by_day[dt.date.fromisoformat(row['date'])] += row['count']
        if baseline is None:
            return {'source': None, 'rows': [], 'notes': [], 'explanations': [
                '未提供原按日容量预测；逐时段结论独立呈现，逐日需求合计见 demand_basis。']}
        baseline_days = {row['date']: row for row in baseline['daily']}
        rows = []
        for day, arrivals in sorted(by_day.items()):
            reference = baseline_days.get(day)
            if reference is None:
                rows.append({'date': day.isoformat(), 'simulated_arrivals': arrivals,
                             'forecast_demand': None, 'difference': None, 'difference_pct': None, 'agents': None})
                continue
            difference = arrivals - reference['demand']
            excluded_count = closed_by_day.get(day, 0)
            rows.append({'date': day.isoformat(), 'simulated_arrivals': arrivals,
                         'excluded_arrivals': excluded_count,
                         'total_observed_arrivals': arrivals + excluded_count,
                         'forecast_demand': reference['demand'], 'difference': round(difference, 3),
                         'difference_pct': round(difference / reference['demand'] * 100, 2) if reference['demand'] else None,
                         'agents': reference['agents']})
        return {'source': baseline['source'], 'rows': rows, 'notes': baseline.get('notes', []),
                'explanations': [
                    '原容量判断按“日需求×服务时长÷(班次时长×目标占用率)”得到建议人数，是全天平均口径。',
                    '本分析沿用同批到达观测与同一份班表，按小时槽排队；日合计接近也可能在高峰时段越过服务承诺。',
                    '休息时段已从可接待分钟中剔除，仅按真实可受理时间计算能力，因此与按班次名义总时长估算的建议人数不可直接等同。',
                    '延至下一班的需求计入 deferred，不算当班消化；给定时段内未消化的需求计入 unserved。两者都会放大尾部等待。',
                    'simulated_arrivals 只含班表开放时段的到达；closed/班次窗口外的需求单列在 excluded_arrivals，两者相加才是全部观测需求。',
                ]}

    def _assumptions(self, fill_policy: str, replications: int, discipline: str) -> list[str]:
        return [
            '到达在同一槽位内按渠道铺开：预约集中在槽起点、间隔 30 分钟，现场到达均匀分布；该铺陈固定并随结果披露。',
            f'队列规则为 {discipline}；服务一旦开始可跨休息或闭馆完成，但不会在休息/闭馆分钟开始。',
            f'需求不确定性用 {replications} 次有种子自举复现；所有席位方案共用同一批到达抽样，只改变席位配置。',
            f'缺档处理为 fill_policy={fill_policy}；低样本时段按小样本自举，不用填充均值替代实测高峰。',
            '席位=可同时接待的窗口数，默认在班次内恒定；按日覆盖（date_seats）单独记录在溯源信息中。',
            'wait_distribution 含跨班等待；同班排队看 same_shift_wait_distribution，延至下一班的等待单独看 deferred_delay_distribution。',
            'wait_distribution_observed_demand_only 只保留实测需求产生的等待；回填需求（imputed_pooled）的等待不计入该口径。',
            '观测为小时槽时，部分开放的槽位（如跨午休边界）整槽需求按观测值计入，不按开放比例折算；这是小时级观测的分辨率限制。',
        ]


def _open_cells(calendar: ShiftCalendar, horizon: tuple[dt.date, dt.date],
                slot_minutes: int) -> set[tuple[int, int]]:
    """Weekday×slot cells where the roster actually provides contact time."""
    cells: set[tuple[int, int]] = set()
    for day in _days(horizon):
        windows = calendar.windows(day)
        if not windows:
            continue
        for slot in range(0, 24 * 60, slot_minutes):
            start = dt.datetime.combine(day, dt.time.min, calendar.timezone) + dt.timedelta(minutes=slot)
            end = start + dt.timedelta(minutes=slot_minutes)
            if any(item.start < end and item.end > start for item in windows):
                cells.add((day.weekday(), slot))
    return cells


def _slot_bucket() -> dict[str, Any]:
    return {'arrivals': 0, 'served': 0, 'same_shift': 0, 'deferred': 0, 'waiting': 0,
            'waits': [], 'breach_target': 0, 'breach_promise': 0,
            'deferred_to': defaultdict(int), 'evidence': defaultdict(int)}


def _reason_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        totals[row['reason']] += row['count']
    return dict(sorted(totals.items()))


def _channel_bucket() -> dict[str, Any]:
    return {'arrivals': 0, 'served': 0, 'unserved': 0, 'deferred': 0, 'waits': [], 'breach_promise': 0}


def _day_bucket() -> dict[str, Any]:
    return {'arrivals': 0, 'served': 0, 'same_shift': 0, 'deferred': 0, 'unserved': 0,
            'backlog_end': 0, 'waits': []}


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def _apply_scenario(calendar: ShiftCalendar, scenario: Scenario) -> ShiftCalendar:
    """Seat plan applied without touching shifts, breaks, holidays, or demand."""
    base = {shift.shift_id: shift.seats for shift in calendar.shifts}
    overrides = dict(scenario.seat_overrides)
    unknown = sorted(set(overrides) - set(base))
    if unknown:
        raise WaitRiskError(f'seat override references unknown shift(s): {", ".join(unknown)}')
    resolved = {}
    for shift_id, seats in base.items():
        value = overrides.get(shift_id, seats + scenario.seats_delta)
        if value < 0:
            raise WaitRiskError(f'scenario {scenario.scenario_id} removes more seats than shift {shift_id} has')
        resolved[shift_id] = value
    date_seats: dict[dt.date, dict[str, int]] = {}
    for day, values in calendar.date_seats.items():
        adjusted = {}
        for shift_id, seats in values.items():
            adjusted[shift_id] = overrides.get(shift_id, seats + scenario.seats_delta)
            if adjusted[shift_id] < 0:
                raise WaitRiskError(f'scenario {scenario.scenario_id} removes more seats than {day} {shift_id} has')
        date_seats[day] = adjusted
    shifts = [shift.with_seats(resolved[shift.shift_id]) for shift in calendar.shifts]
    return ShiftCalendar(shifts, calendar.timezone, calendar.holidays, calendar.closures,
                         calendar.source, date_seats)
