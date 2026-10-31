import asyncio
import datetime
import json
import os
import time
from collections import defaultdict,deque
from pathlib import Path
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,ConfigDict
from dotenv import load_dotenv
from .demo import generate,DEFAULT_DB,DEFAULT_CONTRACT
from .catalog import Catalog
from .sql import execute
from . import prompts

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')
from .model import complete,MODEL
# This public demo always uses generated data, never a local case-study database.
generate()
catalog=Catalog(DEFAULT_DB,DEFAULT_CONTRACT)
app=FastAPI(title='Data Talker',docs_url=None,redoc_url=None)
app.mount('/assets',StaticFiles(directory=ROOT/'web'),name='assets')
semaphore=asyncio.Semaphore(2)
requests=defaultdict(deque)
daily={'day':'','calls':0}
MAX_CALLS=max(0,int(os.getenv('DAILY_MODEL_CALLS','100')))

class Turn(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question:str=Field(max_length=1200)
    sql:str=Field(default='',max_length=12000)

class Query(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question:str=Field(min_length=2,max_length=1200)
    history:list[Turn]=Field(default_factory=list,max_length=6)

@app.middleware('http')
async def headers(request,call_next):
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
    return response

@app.get('/')
async def index():return FileResponse(ROOT/'web/index.html')

@app.get('/api/health')
async def health():return {'status':'ok','dataset':catalog.contract['id'],'live_configured':bool(os.getenv('GROQ_API_KEY'))}

@app.get('/api/catalog')
async def get_catalog():return catalog.public()


def chart_spec(chart,result):
    columns={c['name']:i for i,c in enumerate(result['columns'])}
    if not isinstance(chart,dict) or chart.get('type') not in ('bar','line','scatter'):return None
    if chart.get('x') not in columns or chart.get('y') not in columns:return None
    y=columns[chart['y']]
    if not result['rows'] or len(result['rows'])>60 or any(r[y] is not None and (not isinstance(r[y],(int,float)) or isinstance(r[y],bool)) for r in result['rows']):return None
    return {k:chart[k] for k in ('type','x','y')}

@app.post('/api/examples/{example_id}')
async def example(example_id:str,request:Request):
    throttle(request,limit=60,window=60,kind='example')
    if semaphore.locked():raise HTTPException(503,'The demo is handling two requests. Please retry in a moment.')
    item=next((e for e in catalog.contract['examples'] if e['id']==example_id),None)
    if item is None:raise HTTPException(404,'Unknown example')
    async with semaphore:
        start=time.monotonic()
        result=await asyncio.to_thread(execute,item['sql'],catalog.path,catalog.tables)
    return {**result,'question':item['question'],'title':item['title'],'answer':'This verified example runs a reviewed query against the synthetic dataset. Inspect its SQL and results below.','assumptions':[],'explanation':'Reviewed example query, executed now. No model call.','chart':chart_spec(item.get('chart'),result),'mode':'Verified example','latency_ms':round((time.monotonic()-start)*1000),'model_calls':0}


def reserve_call():
    day=datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    if daily['day']!=day:daily.update(day=day,calls=0)
    if daily['calls']>=MAX_CALLS:raise RuntimeError('The public demo’s daily model-call allowance is reached. Verified examples remain available.')
    daily['calls']+=1

def throttle(request,limit=15,window=3600,kind='live'):
    # The hosting proxy supplies the client IP. The global model allowance
    # remains authoritative even if IPs rotate or forwarded headers are forged.
    client=request.headers.get('x-forwarded-for',request.client.host if request.client else 'unknown').split(',')[0].strip()[:100]
    now=time.monotonic()
    if len(requests)>10000:
        for key in list(requests):
            if not requests[key] or requests[key][-1]<now-3600:del requests[key]
        if len(requests)>10000:raise HTTPException(429,'The public demo is busy. Please retry later.')
    queue=requests[(kind,client)]
    while queue and queue[0]<now-window:queue.popleft()
    if len(queue)>=limit:raise HTTPException(429,'Public demo request limit reached. Please retry later.')
    queue.append(now)


@app.post('/api/query')
async def query(body:Query,request:Request):
    throttle(request)
    if semaphore.locked():raise HTTPException(503,'The demo is handling two requests. Please retry in a moment.')
    ambiguity=catalog.ambiguity(body.question)
    if ambiguity:
        options=[' · '.join(f'{k}: {v}' for k,v in c.items()) for c in ambiguity['choices']]
        return {'clarification':'Which context do you mean for '+ambiguity['value']+'? Include the manager and agency in your next question.','choices':options}
    async with semaphore:
        start=time.monotonic();calls=0;tokens=0
        context={'catalog':catalog.prompt_data(),'question':body.question,'history':[t.model_dump() for t in body.history]}
        try:
            reserve_call();calls+=1
            content,usage=await complete(prompts.PLANNER,json.dumps(context),True)
            tokens+=usage.get('total_tokens',0)
            plan=json.loads(content)
            if plan.get('clarification') and not plan.get('sql'):return {'clarification':str(plan['clarification'])[:500]}
            result=None
            for attempt in range(2):
                try:
                    result=await asyncio.to_thread(execute,plan.get('sql',''),catalog.path,catalog.tables)
                    break
                except Exception as error:
                    if attempt:raise RuntimeError('A safe query could not be produced. Try rephrasing or use a verified example.') from error
                    reserve_call();calls+=1
                    repair={**context,'previous_query':plan.get('sql',''),'validation_error':str(error)[:1000],'instruction':'Repair the query using only the supplied catalog and original question.'}
                    content,usage=await complete(prompts.PLANNER,json.dumps(repair),True)
                    tokens+=usage.get('total_tokens',0);plan=json.loads(content)
                    if plan.get('clarification') and not plan.get('sql'):return {'clarification':str(plan['clarification'])[:500]}
            assumptions=plan.get('assumptions',[])
            assumptions=[str(v)[:500] for v in assumptions[:8]] if isinstance(assumptions,list) else []
            # Describe executed values directly; a second model cannot invent a
            # narrative statistic or change the meaning of a missing result.
            if len(result['rows'])==1 and len(result['columns'])<=4:
                answer=' · '.join(c['name']+': '+('NULL' if v is None else str(v)) for c,v in zip(result['columns'],result['rows'][0]))
            elif not result['rows']:
                answer='No rows matched this query.'
            else:
                answer=str(len(result['rows']))+' rows returned. Inspect the calculation, chart, and executed SQL below.'
            if result['truncated']:answer+=' Display limit reached; this is a subset of the result.'
            return {**result,'question':body.question,'title':str(plan.get('title','Query result'))[:120],'answer':answer,'explanation':str(plan.get('explanation',''))[:800],'assumptions':assumptions,'chart':chart_spec(plan.get('chart'),result),'mode':'Live query','model':MODEL,'model_calls':calls,'tokens':tokens,'latency_ms':round((time.monotonic()-start)*1000)}
        except RuntimeError as error:raise HTTPException(503,str(error))
        except Exception:raise HTTPException(503,'The model returned an unusable response. Please rephrase or try a verified example.')
