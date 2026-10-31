"""Opt-in paid-provider smoke evaluation; never runs in unittest discovery."""
import json
import sys
import time
from pathlib import Path
import httpx
from datatalker.app import catalog
from datatalker.sql import execute

cases=[
('What is the total revenue for 2025?',"SELECT SUM(amount) FROM orders WHERE order_date>=DATE '2025-01-01' AND order_date<DATE '2026-01-01'"),
('What is the total revenue across all years?', 'SELECT SUM(amount) FROM orders'),
('How many orders are there?', 'SELECT COUNT(*) FROM orders'),
('Show total revenue by region across all years, highest first.', 'SELECT r.region, SUM(o.amount) FROM orders o JOIN reps r ON o.rep_id=r.rep_id GROUP BY r.region ORDER BY 2 DESC'),
('Who are the top 3 salespeople by total revenue across all years? Return only name and revenue.', 'SELECT r.name,SUM(o.amount) FROM orders o JOIN reps r ON o.rep_id=r.rep_id GROUP BY r.name ORDER BY 2 DESC LIMIT 3'),
('How much revenue was eligible across all years?', 'SELECT SUM(o.amount) FROM orders o WHERE EXISTS(SELECT 1 FROM eligibility_periods e WHERE e.rep_id=o.rep_id AND o.order_date BETWEEN e.start_date AND e.end_date)'),
('How much revenue was ineligible across all years?', 'SELECT SUM(o.amount) FROM orders o WHERE NOT EXISTS(SELECT 1 FROM eligibility_periods e WHERE e.rep_id=o.rep_id AND o.order_date BETWEEN e.start_date AND e.end_date)'),
('What were Alex Morgan\'s commissions in 2025 under Taylor Gray at Northstar?', "SELECT SUM(commission) FROM commissions WHERE agent_name='Alex Morgan' AND manager='Taylor Gray' AND agency='Northstar' AND month>=DATE '2025-01-01' AND month<DATE '2026-01-01'"),
]

def normalize(rows):
    return [[round(v,2) if isinstance(v,(float,int)) else v for v in row] for row in rows]

if __name__=='__main__':
    base=sys.argv[1] if len(sys.argv)>1 else 'http://localhost:8070'
    report=[]
    with httpx.Client(base_url=base,timeout=100) as client:
        for question,reference in cases:
            r=client.post('/api/query',json={'question':question});d=r.json()
            expected=execute(reference,catalog.path,catalog.tables)['rows']
            passed=r.status_code==200 and normalize(d.get('rows',[]))==normalize(expected)
            report.append({'question':question,'passed':passed,'status':r.status_code,'sql':d.get('sql'),'expected':expected,'actual':d.get('rows'),'detail':d.get('detail')})
            print(('PASS' if passed else 'FAIL'),question,flush=True)
            time.sleep(15)
        r=client.post('/api/query',json={'question':'Show commissions for Alex Morgan'})
        passed=r.status_code==200 and len(r.json().get('choices',[]))==2
        report.append({'question':'Ambiguous entity asks for context','passed':passed});print('PASS' if passed else 'FAIL','ambiguity',flush=True)
        r=client.post('/api/query',json={'question':'Show revenue in 2025 only','history':[{'question':'What is revenue by region?','sql':'SELECT r.region,SUM(o.amount) revenue FROM orders o JOIN reps r ON o.rep_id=r.rep_id GROUP BY r.region ORDER BY revenue DESC'}]})
        expected=execute("SELECT r.region,SUM(o.amount) FROM orders o JOIN reps r ON o.rep_id=r.rep_id WHERE year(o.order_date)=2025 GROUP BY r.region ORDER BY 2 DESC",catalog.path,catalog.tables)['rows']
        passed=r.status_code==200 and normalize(r.json().get('rows',[]))==normalize(expected)
        report.append({'question':'Follow-up preserves region grouping and changes year','passed':passed,'sql':r.json().get('sql')});print('PASS' if passed else 'FAIL','follow-up',flush=True)
    path=Path('/tmp/data-talker-live-eval.json');path.write_text(json.dumps(report,indent=2));print(f'{sum(r["passed"] for r in report)}/{len(report)} checks passed. Report: {path}')
    sys.exit(0 if all(r['passed'] for r in report) else 1)
