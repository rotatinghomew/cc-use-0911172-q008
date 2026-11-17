人才驿站与服务窗口需求预测容量计划台
测试：python3 -m unittest discover -s tests -v
构建：python3 -m compileall -q service_capacity tests
验收：python3 -m service_capacity --database /tmp/service_capacity.sqlite3 run --fixture fixtures/acceptance.json --output /tmp/service_capacity-output.json
单测：python3 -m unittest tests.test_domain -v

逐时段等待风险（叠加在原按日容量预测之上，不改写原口径）：
摘要：python3 -m service_capacity wait-risk --input fixtures/wait_risk_hongqiao.json --summary
全量：python3 -m service_capacity wait-risk --input fixtures/wait_risk_hongqiao.json --output /tmp/wait-risk.json
单测：python3 -m unittest tests.test_wait_risk -v

输入 JSON 需要：site_id、timezone、slot_minutes、horizon（或从观测末日起 7 天）、
arrivals.observations（逐日逐槽，channel=appointment/walk_in，可带 source/site_id）、
service（samples_seconds / by_channel / 可选 assumed_mean_seconds）、
calendar.shifts（weekdays/start/end/seats/breaks，另有 holidays/closures/date_seats）、
scenarios（seats_delta 或 seat_overrides，第一个为对照基线）、可选 baseline.daily 用于与原按日容量预测对账。
