import datetime as dt
import json
import unittest

from service_capacity.wait_risk import (
    Break,
    Shift,
    ShiftCalendar,
    WaitRiskAnalyzer,
    WaitRiskError,
    simulate_day,
)
from service_capacity.wait_risk.queue import Arrival, simulate_arrivals


def day(hour=0, minute=0):
    return dt.datetime(2026, 11, 16, hour, minute, tzinfo=dt.timezone.utc)


class CalendarTests(unittest.TestCase):
    def calendar(self):
        return ShiftCalendar([
            Shift('morning', frozenset(range(7)), dt.time(8), dt.time(12), 2,
                  (Break(dt.time(10), dt.time(11), 'lunch'),)),
        ], dt.timezone.utc, source='roster-1')

    def test_break_removes_capacity(self):
        metrics = self.calendar().day_metrics(dt.date(2026, 11, 16))
        self.assertEqual(metrics['seat_minutes'], 2 * 3 * 60)      # 3 open hours × 2 seats
        self.assertEqual(metrics['window_seat_minutes'], 2 * 4 * 60)
        self.assertEqual(metrics['break_seat_minutes'], 2 * 60)

    def test_capacity_at_break_is_zero(self):
        calendar = self.calendar()
        self.assertEqual(calendar.capacity_at(day(10, 30)), 0)
        self.assertEqual(calendar.capacity_at(day(9, 30)), 2)

    def test_seat_configuration_does_not_change_calendar_structure(self):
        calendar = self.calendar()
        bigger = calendar.with_seats({'morning': 4})
        self.assertEqual(calendar.structural_fingerprint(), bigger.structural_fingerprint())
        self.assertNotEqual(calendar.structural_fingerprint(include_seats=True),
                            bigger.structural_fingerprint(include_seats=True))

    def test_break_outside_shift_rejected(self):
        with self.assertRaises(WaitRiskError):
            Shift('bad', frozenset({0}), dt.time(9), dt.time(10), 1,
                  (Break(dt.time(8), dt.time(9)),))


class QueueTests(unittest.TestCase):
    def calendar(self):
        return ShiftCalendar([Shift('s', frozenset(range(7)), dt.time(9), dt.time(12), 1)], dt.timezone.utc)

    def test_single_seat_serialises(self):
        calendar = self.calendar()
        arrivals = [Arrival('a', dt.date(2026, 11, 16), day(9, 0), 'walk_in', 540, 600),
                    Arrival('b', dt.date(2026, 11, 16), day(9, 0), 'walk_in', 540, 600)]
        result = simulate_arrivals(calendar, arrivals, dt.date(2026, 11, 16), dt.date(2026, 11, 16))
        self.assertEqual([r.customer_id for r in result.records], ['a', 'b'])
        self.assertEqual(result.records[1].wait_seconds, 600)

    def test_deferred_to_next_shift(self):
        calendar = ShiftCalendar([
            Shift('early', frozenset(range(7)), dt.time(9), dt.time(10), 1),
            Shift('late', frozenset(range(7)), dt.time(11), dt.time(12), 1),
        ], dt.timezone.utc)
        arrivals = [Arrival(f'c{i}', dt.date(2026, 11, 16), day(9, 50), 'walk_in', 570, 1800) for i in range(3)]
        result = simulate_arrivals(calendar, arrivals, dt.date(2026, 11, 16), dt.date(2026, 11, 16))
        keys = {(r.arrival_shift_key, r.served_shift_key) for r in result.records}
        self.assertIn(((dt.date(2026, 11, 16), 'early'), (dt.date(2026, 11, 16), 'early')), keys)
        self.assertTrue(any(not r.same_shift for r in result.records))

    def test_no_start_during_break(self):
        calendar = ShiftCalendar([Shift('s', frozenset(range(7)), dt.time(9), dt.time(12), 1,
                                        (Break(dt.time(10), dt.time(11)),))], dt.timezone.utc)
        arrivals = [Arrival('a', dt.date(2026, 11, 16), day(9, 55), 'walk_in', 595, 900),
                    Arrival('b', dt.date(2026, 11, 16), day(9, 55), 'walk_in', 595, 600)]
        result = simulate_arrivals(calendar, arrivals, dt.date(2026, 11, 16), dt.date(2026, 11, 16))
        for record in result.records:
            self.assertFalse(dt.time(10) <= record.started.time() < dt.time(11))

    def test_appointment_first_orders_queue(self):
        calendar = self.calendar()
        arrivals = [Arrival('walk', dt.date(2026, 11, 16), day(9, 0), 'walk_in', 540, 1800),
                    Arrival('booked', dt.date(2026, 11, 16), day(9, 0), 'appointment', 540, 600)]
        result = simulate_arrivals(calendar, arrivals, dt.date(2026, 11, 16), dt.date(2026, 11, 16),
                                   discipline='appointment_first')
        self.assertEqual(result.records[0].customer_id, 'booked')


class AnalyzerTests(unittest.TestCase):
    def dataset(self, **changes):
        base = {
            'site_id': 'station-1',
            'slot_minutes': 60,
            'timezone': 'UTC',
            'horizon': {'start': '2026-11-16', 'end': '2026-11-20'},
            'arrivals': {
                'source': 'station-1-arrivals',
                'minimum_days': 2,
                'observations': [
                    {'observation_id': 'o1', 'date': '2026-11-02', 'slot': '09:00', 'channel': 'appointment', 'count': 2, 'source': 'booking-log'},
                    {'observation_id': 'o2', 'date': '2026-11-02', 'slot': '09:00', 'channel': 'walk_in', 'count': 1, 'source': 'queue-log'},
                    {'observation_id': 'o3', 'date': '2026-11-09', 'slot': '09:00', 'channel': 'appointment', 'count': 4, 'source': 'booking-log'},
                    {'observation_id': 'o4', 'date': '2026-11-09', 'slot': '09:00', 'channel': 'walk_in', 'count': 3, 'source': 'queue-log'},
                    {'observation_id': 'o5', 'date': '2026-11-02', 'slot': '14:00', 'channel': 'walk_in', 'count': 2, 'source': 'queue-log'},
                    {'observation_id': 'o6', 'date': '2026-11-09', 'slot': '14:00', 'channel': 'walk_in', 'count': 2, 'source': 'queue-log'},
                ],
            },
            'service': {'samples_seconds': [600, 720, 540, 900, 660, 600],
                        'source': 'station-1-service-log'},
            'calendar': {
                'source': 'station-1-roster',
                'shifts': [{'shift_id': 'am', 'weekdays': [0, 1, 2, 3, 4], 'start': '09:00', 'end': '12:00',
                            'seats': 1, 'breaks': [{'start': '10:00', 'end': '10:30'}]},
                           {'shift_id': 'pm', 'weekdays': [0, 1, 2, 3, 4], 'start': '13:00', 'end': '17:00',
                            'seats': 1}],
                'holidays': ['2026-11-18'],
            },
            'scenarios': [
                {'scenario_id': 'as-is', 'seats_delta': 0, 'label': 'current'},
                {'scenario_id': 'plus-one', 'seats_delta': 1, 'label': 'add one seat'},
            ],
            'baseline': {'source': 'daily-capacity-forecast',
                         'daily': [{'date': '2026-11-16', 'demand': 8, 'agents': 1},
                                   {'date': '2026-11-17', 'demand': 8, 'agents': 1}]},
            'replications': 20,
            'seed': 7,
        }
        base.update(changes)
        return base

    def test_report_structure(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        self.assertEqual(report['site_id'], 'station-1')
        self.assertEqual(report['lineage']['demand_fingerprint'],
                         report['comparison']['demand_fingerprint'])
        self.assertTrue(report['comparison']['demand_held_constant'])
        self.assertTrue(report['comparison']['calendar_held_constant'])
        self.assertEqual(len(report['scenarios']), 2)
        self.assertIn('wait_distribution', report['scenarios'][0])
        self.assertIn('slot_table', report['scenarios'][0])
        self.assertIn('deferred_to', report['scenarios'][0]['slot_table'][0])

    def test_holiday_has_no_capacity(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        holiday = next(row for row in report['calendar_metrics']['days'] if row['date'] == '2026-11-18')
        self.assertEqual(holiday['seat_minutes'], 0)
        self.assertEqual(holiday['open_minutes'], 0)
        day_row = next(row for row in report['scenarios'][0]['daily'] if row['date'] == '2026-11-18')
        self.assertFalse(day_row['open'])

    def test_missing_slot_not_imputed_by_default(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        rows = [row for row in report['demand_basis'] if row['slot'] == '13:00']
        self.assertEqual(rows, [])
        self.assertGreater(report['uncertainty']['coverage']['missing_slots'], 0)
        self.assertEqual(report['uncertainty']['imputed_demand_count'], 0)

    def test_fill_policy_is_labelled(self):
        report = WaitRiskAnalyzer().analyze(self.dataset(fill_policy='include'))
        self.assertGreater(report['uncertainty']['imputed_demand_count'], 0)
        imputed = [row for row in report['demand_basis'] if row['evidence'] == 'imputed_pooled']
        self.assertTrue(imputed)
        self.assertTrue(report['observed_peak'])
        peak_sources = {tuple(row['source_ids']) for row in report['demand_basis']
                        if row['slot'] == report['observed_peak']['peak']['slot']}
        self.assertTrue(peak_sources)

    def test_observed_peak_excludes_low_sample(self):
        data = self.dataset()
        data['arrivals']['observations'] = data['arrivals']['observations'][:2]
        report = WaitRiskAnalyzer().analyze(data)
        self.assertIsNone(report['observed_peak'])
        self.assertGreater(report['uncertainty']['coverage']['low_sample_slots'], 0)

    def test_more_seats_reduce_deferral(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        baseline, larger = report['scenarios']
        self.assertGreaterEqual(baseline['throughput']['deferred'], larger['throughput']['deferred'])
        self.assertGreater(larger['seat_configuration']['am'], baseline['seat_configuration']['am'])
        self.assertGreaterEqual(baseline['wait_distribution']['p90_seconds'],
                                larger['wait_distribution']['p90_seconds'])

    def test_seat_comparison_uses_same_demand(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        self.assertTrue(report['comparison']['demand_held_constant'])
        self.assertTrue(report['comparison']['calendar_held_constant'])
        self.assertEqual(report['scenarios'][0]['demand_fingerprint'],
                         report['scenarios'][1]['demand_fingerprint'])
        self.assertEqual(report['comparison']['rows'][0]['arrivals'],
                         report['comparison']['rows'][1]['arrivals'])

    def test_scenarios_share_slot_level_demand(self):
        data = self.dataset()
        data['scenarios'] = [
            {'scenario_id': 'as-is', 'seats_delta': 0},
            {'scenario_id': 'plus-one', 'seats_delta': 1},
            {'scenario_id': 'repeat-as-is', 'seats_delta': 0},
        ]
        report = WaitRiskAnalyzer().analyze(data)
        as_is, plus, repeat = report['scenarios']
        self.assertEqual(as_is['slot_table'], repeat['slot_table'])
        self.assertEqual(as_is['wait_distribution'], repeat['wait_distribution'])
        key = lambda rows: {(row['weekday'], row['slot']): row['arrivals'] for row in rows}
        self.assertEqual(key(as_is['slot_table']), key(plus['slot_table']))
        self.assertEqual(as_is['throughput']['served'], plus['throughput']['served'])

    def test_unknown_shift_override_rejected(self):
        data = self.dataset()
        data['scenarios'] = [{'scenario_id': 'bad', 'seat_overrides': {'ghost': 1}}]
        with self.assertRaises(WaitRiskError):
            WaitRiskAnalyzer().analyze(data)

    def test_negative_seat_scenario_rejected(self):
        data = self.dataset()
        data['calendar']['shifts'][0]['seats'] = 1
        data['scenarios'] = [{'scenario_id': 'minus', 'seats_delta': -2}]
        with self.assertRaises(WaitRiskError):
            WaitRiskAnalyzer().analyze(data)

    def test_baseline_reconciliation_rows(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        rows = report['baseline_reconciliation']['rows']
        self.assertTrue(any(row['forecast_demand'] == 8 for row in rows))
        self.assertEqual(report['baseline_reconciliation']['source'], 'daily-capacity-forecast')

    def test_observation_duplicate_rejected(self):
        data = self.dataset()
        data['arrivals']['observations'].append(dict(data['arrivals']['observations'][0]))
        with self.assertRaises(WaitRiskError):
            WaitRiskAnalyzer().analyze(data)

    def test_wrong_site_observation_rejected(self):
        data = self.dataset()
        data['arrivals']['observations'][0]['site_id'] = 'station-2'
        with self.assertRaises(WaitRiskError):
            WaitRiskAnalyzer().analyze(data)

    def test_source_attribution_in_demand_basis(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        sources = {row['source'] for row in report['demand_basis'] if row['source']}
        self.assertIn('booking-log', sources)
        self.assertIn('queue-log', sources)

    def test_seed_makes_bootstrap_repeatable(self):
        first = WaitRiskAnalyzer().analyze(self.dataset())
        second = WaitRiskAnalyzer().analyze(self.dataset())
        self.assertEqual(first['scenarios'][0]['wait_distribution'], second['scenarios'][0]['wait_distribution'])

    def test_coverage_only_counts_open_cells(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        # Horizon Mon 16..Fri 20 with Wed 18 a holiday: 4 open days × 7 open hours.
        self.assertEqual(report['coverage']['summary']['open_cells'], 7 * 4)
        holiday = next(entry for entry in report['calendar_metrics']['days'] if entry['date'] == '2026-11-18')
        self.assertEqual(holiday['shifts'], [])

    def test_demand_outside_window_is_not_capacity(self):
        data = self.dataset()
        data['arrivals']['observations'].append(
            {'observation_id': 'o7', 'date': '2026-11-02', 'slot': '18:00', 'channel': 'walk_in',
             'count': 3, 'source': 'queue-log'})
        data['arrivals']['observations'].append(
            {'observation_id': 'o8', 'date': '2026-11-09', 'slot': '18:00', 'channel': 'walk_in',
             'count': 3, 'source': 'queue-log'})
        report = WaitRiskAnalyzer().analyze(data)
        self.assertGreater(report['excluded_demand']['total'], 0)
        self.assertTrue(all(row['slot'] == '18:00' for row in report['excluded_demand']['rows']))
        self.assertFalse(any(row['slot'] == '18:00' for row in report['demand_basis']))
        by_date = {row['date']: row for row in report['baseline_reconciliation']['rows']}
        self.assertEqual(by_date['2026-11-16']['excluded_arrivals'], 3)

    def test_break_minutes_are_not_capacity(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        metrics = report['calendar_metrics']['days']
        open_day = next(entry for entry in metrics if entry['date'] == '2026-11-16')
        self.assertEqual(open_day['open_minutes'], 420)      # 3h morning + 4h afternoon
        self.assertEqual(open_day['seat_minutes'], (180 - 30) + 240)   # 1 seat, minus 30-min break
        self.assertGreater(open_day['break_seat_minutes'], 0)

    def test_risk_windows_and_channel_split(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        scenario = report['scenarios'][0]
        channels = {row['channel'] for row in scenario['channel_table']}
        self.assertEqual(channels, {'appointment', 'walk_in'})
        total = sum(row['arrivals'] for row in scenario['channel_table'])
        self.assertEqual(total, scenario['throughput']['arrivals'])
        for row in scenario['risk_windows']['slots']:
            self.assertGreater(row['promise_breach_probability'], 0)
        probabilities = [row['promise_breach_probability'] for row in scenario['risk_windows']['slots']]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_queue_vs_deferral_split(self):
        report = WaitRiskAnalyzer().analyze(self.dataset())
        scenario = report['scenarios'][0]
        total = scenario['wait_distribution']['count']
        split = (scenario['same_shift_wait_distribution']['count']
                 + scenario['deferred_delay_distribution']['count'])
        self.assertEqual(total, split)
        if scenario['deferral_flows']:
            self.assertTrue(all(flow['from'] != flow['to'] for flow in scenario['deferral_flows']))

    def test_cli_summary(self):
        import io
        from contextlib import redirect_stdout
        from service_capacity.cli import main
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(['wait-risk', '--input', 'fixtures/wait_risk_hongqiao.json', '--summary'])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload['site_id'], 'station-hongqiao-1')
        self.assertEqual(len(payload['scenarios']), 3)


if __name__ == '__main__':
    unittest.main()
