from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .reporting import Reporter
from .service import ApplicationService
from .storage import Database
from .wait_risk import WaitRiskAnalyzer, WaitRiskError

def load_json(path:str):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main(argv=None):
    parser=argparse.ArgumentParser(prog='service_capacity')
    parser.add_argument('--database',default=':memory:')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run'); run.add_argument('--fixture',required=True); run.add_argument('--output',required=True)
    report=sub.add_parser('report'); report.add_argument('--kind')
    wait=sub.add_parser('wait-risk',help='逐时段等待风险分析（不改写原按日容量预测）')
    wait.add_argument('--input',required=True,help='到达观测+班表+服务时长+席位方案 JSON')
    wait.add_argument('--output',help='输出 JSON 路径；缺省打印到标准输出')
    wait.add_argument('--summary',action='store_true',help='只打印摘要')
    args=parser.parse_args(argv)
    if args.command=='wait-risk':
        try:
            payload=WaitRiskAnalyzer().analyze(load_json(args.input))
        except WaitRiskError as exc:
            print(json.dumps({'error':str(exc)},ensure_ascii=False),file=sys.stderr); return 2
        if args.summary:
            payload={'site_id':payload['site_id'],'horizon':payload['horizon'],
                     'coverage':payload['coverage']['summary'],'observed_peak':payload['observed_peak'],
                     'scenarios':[{'scenario_id':item['scenario']['scenario_id'],'seats':item['seat_configuration'],
                                   'throughput':item['throughput'],'wait_distribution':item['wait_distribution']}
                                  for item in payload['scenarios']],
                     'comparison':payload['comparison'],'uncertainty':payload['uncertainty'],
                     'baseline_reconciliation':payload['baseline_reconciliation']}
        text=json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)
        if args.output:
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            Path(args.output).write_text(text,encoding='utf-8')
            print(json.dumps({'written':args.output,'scenarios':[item['scenario']['scenario_id'] for item in payload['scenarios']]},ensure_ascii=False))
        else:
            print(text)
        return 0
    database=Database(args.database)
    if args.command=='run':
        fixture=load_json(args.fixture); service=ApplicationService(database); outputs=[]
        for index,item in enumerate(fixture.get('items',[])):
            outputs.append(service.process(item,fixture.get('context',{}),item.get('request_key',f'item-{index}')))
        payload={'product':'人才驿站与服务窗口需求预测容量计划台','processed':len(outputs),'outputs':outputs,'report':Reporter(database).overview()}
        Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(payload,ensure_ascii=False,sort_keys=True)); return 0
    payload=Reporter(database).owner_breakdown(args.kind) if args.kind else Reporter(database).overview(); print(json.dumps(payload,ensure_ascii=False,sort_keys=True)); return 0
