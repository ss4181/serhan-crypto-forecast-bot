"""Fill only the approved missing event windows; Trade1 is read-only input.

Data acquisition is independent of outcomes. Original entry rules are untouched.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd

BAR=300000
DAY=86400000
COLS=('open_time_ms','close_time_ms','open','high','low','close')
BASE='https://data.binance.vision/data/futures/um/daily/klines'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(frame,event):
    if not set(COLS).issubset(frame): return None,'MISSING_OHLC_COLUMNS'
    start=int(event['bar_close_time_ms'])-BAR+1
    end=start+144*BAR
    w=frame.loc[frame.open_time_ms.between(start,end),list(COLS)].copy()
    if w.open_time_ms.duplicated().any(): return None,'DUPLICATE_TIME'
    w=w.sort_values('open_time_ms').reset_index(drop=True)
    if len(w)!=145: return None,'INCOMPLETE_WINDOW'
    if not np.array_equal(w.open_time_ms.to_numpy(),np.arange(start,end+1,BAR)): return None,'GAP_OR_WRONG_GRID'
    if not w.close_time_ms.eq(w.open_time_ms+BAR-1).all(): return None,'INVALID_CLOSE_TIME'
    if not np.isfinite(w.to_numpy(dtype=float)).all(): return None,'NONFINITE'
    if not (w[['open','high','low','close']]>0).all().all(): return None,'NONPOSITIVE_PRICE'
    if not (w.high>=w[['open','low','close']].max(axis=1)).all() or not (w.low<=w[['open','close']].min(axis=1)).all(): return None,'INVALID_RANGE'
    if not math.isclose(float(w.iloc[0]['close']),float(event['source_price']),rel_tol=1e-9,abs_tol=1e-12): return None,'SIGNAL_PRICE_CONFLICT'
    if not math.isclose(float(w.iloc[1]['open']),float(event['entry_price']),rel_tol=1e-9,abs_tol=1e-12): return None,'ENTRY_PRICE_CONFLICT'
    return w,'VALID'


def parse_zip(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members=[i for i in archive.infolist() if not i.is_dir() and i.filename.endswith('.csv')]
        if len(members)!=1 or members[0].file_size>20_000_000: raise ValueError('Unexpected market archive members')
        text=archive.read(members[0]).decode('utf-8-sig')
    rows=[]
    for fields in csv.reader(io.StringIO(text)):
        if not fields: continue
        if fields[0].lower() in ('open_time','open time'): continue
        if len(fields)<7: raise ValueError('Short kline row')
        rows.append({'open_time_ms':int(fields[0]),'open':float(fields[1]),'high':float(fields[2]),
                     'low':float(fields[3]),'close':float(fields[4]),'close_time_ms':int(fields[6])})
    return pd.DataFrame(rows,columns=COLS)


def archive_read(path):
    checksum=Path(str(path)+'.CHECKSUM')
    if not checksum.exists(): raise ValueError('CHECKSUM missing')
    expected=checksum.read_text().split()[0].lower()
    actual=digest(path)
    if not re.fullmatch('[a-f0-9]{64}',expected) or actual!=expected: raise ValueError('CHECKSUM mismatch')
    return parse_zip(path.read_bytes()),{'path':str(path),'sha256':actual,'checksum_verified':True}


def get(url):
    if not (url.startswith(BASE+'/') or url.startswith('https://fapi.binance.com/fapi/v1/klines?')): raise ValueError('Unapproved endpoint')
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'Trade3-readonly-research/1.0'}),timeout=20) as response:
                raw=response.read(20_000_001)
                if len(raw)>20_000_000: raise ValueError('Oversized public market response')
                return raw
        except HTTPError as error:
            if error.code in (418,429): raise RuntimeError('RATE_LIMIT_STOP') from error
            if error.code<500 or attempt==2: raise
        except (URLError,TimeoutError):
            if attempt==2: raise
        time.sleep(0.7*(attempt+1))


def dates(event):
    start=int(event['bar_close_time_ms'])-BAR+1
    end=int(event['bar_close_time_ms'])+144*BAR
    return [datetime.fromtimestamp(x/1000,timezone.utc).strftime('%Y-%m-%d') for x in range(start//DAY*DAY,end//DAY*DAY+1,DAY)]


def run(trade1,events_path,coverage_path,output,download=False):
    payload=json.loads(events_path.read_text(encoding='utf-8'))
    previous=json.loads(coverage_path.read_text(encoding='utf-8'))
    missing={r['setup_id'] for r in previous['coverage'] if r['status']!='COVERED'}
    events=[e for e in payload['events'] if e['setup_id'] in missing]
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['events_sha256']!=digest(events_path) or manifest['old_study_sha256']!=digest(coverage_path): raise ValueError('Inputs changed')
    else:
        if output.exists(): raise ValueError('Nonempty unowned output; refusing reuse')
        output.mkdir(parents=True)
        manifest={'authorization':'User approved read-only Trade1 price reference; download only missing event windows on 2026-10-28.',
                  'events_sha256':digest(events_path),'old_study_sha256':digest(coverage_path),'requested_events':len(events),
                  'trade1_root':str(trade1),'records':[],'trade1_modified':False,'live_activation_allowed':False}
    records={r['setup_id']:r for r in manifest['records']}
    sources={}
    for e in events:
        sid=e['setup_id']; symbol=e['perpetual_symbol']
        if not re.fullmatch('[A-Z0-9]{2,25}USDT',symbol) or not re.fullmatch('[a-f0-9]{64}',sid): raise ValueError('Invalid event identifiers')
        existing=records.get(sid)
        if existing and existing.get('status')=='COPIED':
            if digest(Path(existing['output']))!=existing['output_sha256']: raise ValueError('Existing output changed')
            continue
        record=existing or {'setup_id':sid,'symbol':symbol,'strategy':e['strategy_code'],'signal_ms':e['bar_close_time_ms'],'attempts':[]}
        records[sid]=record; window=None; provenance=None; source_kind=None
        if not record.get('trade1_checked'):
            p=trade1/'research/data/extended-history-2026-10-03/um/5m'/f'{symbol}.parquet'
            if p.exists():
                before=digest(p)
                start=int(e['bar_close_time_ms'])-BAR+1; end=start+144*BAR
                f=pd.read_parquet(p,columns=['open_time','open','high','low','close'],filters=[('open_time','>=',start),('open_time','<=',end)])
                f=f.rename(columns={'open_time':'open_time_ms'}); f['close_time_ms']=f.open_time_ms+BAR-1
                window,status=validate(f,e)
                if digest(p)!=before: raise ValueError('Trade1 source changed during read')
                record['attempts'].append({'source':'TRADE1_UM_5M_PARQUET','path':str(p),'sha256':before,'status':status})
                if window is not None: provenance=[{'path':str(p),'sha256':before}]; source_kind='TRADE1_REFERENCE'
            if window is None:
                cache=trade1/'research/data/all-execution-2026-10-02/cache/um'/symbol
                paths=[cache/f'{symbol}-5m-{date}.zip' for date in dates(e)]
                if all(p.exists() for p in paths):
                    tables=[]; provenance=[]
                    for p in paths:
                        f,info=archive_read(p); tables.append(f); provenance.append(info)
                    window,status=validate(pd.concat(tables,ignore_index=True),e)
                    record['attempts'].append({'source':'TRADE1_UM_5M_ZIP','status':status,'files':provenance})
                    if window is not None: source_kind='TRADE1_REFERENCE'
            record['trade1_checked']=True
        if window is None and download:
            try:
                tables=[]; provenance=[]
                for date in dates(e):
                    filename=f'{symbol}-5m-{date}.zip'
                    local=output/'raw-binance-daily'/symbol/filename
                    url=f'{BASE}/{symbol}/5m/{filename}'
                    if not local.exists():
                        checksum=get(url+'.CHECKSUM'); raw=get(url)
                        expected=checksum.decode().split()[0].lower()
                        if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('Downloaded checksum mismatch')
                        local.parent.mkdir(parents=True,exist_ok=True)
                        local.write_bytes(raw); Path(str(local)+'.CHECKSUM').write_bytes(checksum)
                    f,info=archive_read(local); info['url']=url; tables.append(f); provenance.append(info)
                    time.sleep(.15)
                window,status=validate(pd.concat(tables,ignore_index=True),e)
                record['attempts'].append({'source':'BINANCE_DAILY_UM_5M','status':status,'files':provenance})
                if window is not None: source_kind='BINANCE_DAILY_UM_5M'
            except RuntimeError as error:
                if str(error)=='RATE_LIMIT_STOP':
                    record['status']='RATE_LIMIT_STOP'; manifest['records']=list(records.values()); manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8'); raise
                raise
            except (HTTPError,URLError,TimeoutError,ValueError,zipfile.BadZipFile) as error:
                record['attempts'].append({'source':'BINANCE_DAILY_UM_5M','status':'DOWNLOAD_OR_VALIDATION_ERROR','error_type':type(error).__name__,'http_code':getattr(error,'code',None)})
                # Archive absence is not evidence of no trade; use the same
                # official USD-M public kline endpoint, not spot or mark price.
                if isinstance(error,HTTPError) and error.code==404:
                    from urllib.parse import urlencode
                    start=int(e['bar_close_time_ms'])-BAR+1
                    query=urlencode({'symbol':symbol,'interval':'5m','startTime':start,'endTime':int(e['bar_close_time_ms'])+144*BAR,'limit':499})
                    url='https://fapi.binance.com/fapi/v1/klines?'+query
                    try:
                        raw=get(url); parsed=json.loads(raw)
                        if not isinstance(parsed,list): raise ValueError('Public kline error')
                        f=pd.DataFrame([{'open_time_ms':int(r[0]),'open':float(r[1]),'high':float(r[2]),'low':float(r[3]),'close':float(r[4]),'close_time_ms':int(r[6])} for r in parsed])
                        local=output/'raw-binance-api'/f'{sid}.json'; local.parent.mkdir(parents=True,exist_ok=True); local.write_bytes(raw)
                        window,status=validate(f,e); provenance=[{'path':str(local),'sha256':digest(local),'url':url}]
                        record['attempts'].append({'source':'BINANCE_REST_UM_5M','status':status,'files':provenance})
                        if window is not None: source_kind='BINANCE_REST_UM_5M'
                    except (HTTPError,URLError,TimeoutError,ValueError,KeyError,IndexError,TypeError) as fallback:
                        record['attempts'].append({'source':'BINANCE_REST_UM_5M','status':'DOWNLOAD_OR_VALIDATION_ERROR','error_type':type(fallback).__name__,'http_code':getattr(fallback,'code',None)})
        if window is not None:
            target=output/'windows'/sid/f'{symbol}_5m_futures.csv'
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): raise ValueError('Refusing overwrite of price window')
            window.to_csv(target,index=False)
            record.update(status='COPIED',source_kind=source_kind,output=str(target),output_sha256=digest(target),bars=len(window),provenance=provenance)
        else:
            record['status']='UNKNOWN_AFTER_DOWNLOAD' if download else 'NEEDS_DOWNLOAD'
        manifest['records']=list(records.values()); manifest['script_sha256']=digest(Path(__file__))
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(symbol,sid[:8],record['status'],record.get('source_kind',''),flush=True)
    from collections import Counter
    print('SUMMARY',dict(Counter(r['status'] for r in records.values())),dict(Counter(r.get('source_kind','NONE') for r in records.values())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trade1','events','coverage','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--download',action='store_true')
    a=p.parse_args(); run(a.trade1,a.events,a.coverage,a.output,a.download)
