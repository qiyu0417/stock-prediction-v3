"""Download latest data with timeout — saves to data/new_week.csv"""
import json, pandas as pd, baostock as bs, sys, os
from multiprocessing import Process, Queue
import time

with open('data/stock_mapping.json') as f: mapping = json.load(f)
reverse = {v: k for k, v in mapping.items()}
codes = list(reverse.keys())[:300]
print(f'Downloading {len(codes)} stocks...')

def download_batch(rc_list, q):
    lg = bs.login()
    results = []
    for rc in rc_list:
        bsc = f'sh.{rc}' if rc.startswith('6') else f'sz.{rc}'
        try:
            rs = bs.query_history_k_data_plus(bsc,
                'date,code,open,high,low,close,preclose,volume,amount,turn,pctChg',
                start_date='2026-06-12', end_date='2026-06-30',
                frequency='d', adjustflag='1')
            rows = []
            while (rs.error_code == '0') & rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df_new = pd.DataFrame(rows, columns=rs.fields)
                for c in ['open','high','low','close','preclose','volume','amount','turn','pctChg']:
                    df_new[c] = pd.to_numeric(df_new[c], errors='coerce')
                df_new['振幅'] = ((df_new['high']-df_new['low'])/df_new['preclose']*100).round(2)
                df_new['涨跌额'] = (df_new['close']-df_new['preclose']).round(2)
                df_new['code'] = rc
                df_new['date'] = pd.to_datetime(df_new['date'])
                df_new = df_new.rename(columns={'code':'股票代码','date':'日期','open':'开盘','close':'收盘',
                    'high':'最高','low':'最低','volume':'成交量','amount':'成交额','turn':'换手率','pctChg':'涨跌幅'})
                results.append(df_new[['股票代码','日期','开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌额','换手率','涨跌幅']])
        except:
            pass
    bs.logout()
    q.put(results)

# Split into batches of 30
BATCH = 30
all_results = []
for start in range(0, len(codes), BATCH):
    batch = codes[start:start+BATCH]
    q = Queue()
    p = Process(target=download_batch, args=(batch, q))
    p.start()
    p.join(timeout=15)  # 15 second timeout per batch
    if p.is_alive():
        print(f'  Batch {start//BATCH+1}: TIMEOUT, killing...')
        p.terminate()
        p.join()
    else:
        try:
            res = q.get_nowait()
            all_results.extend(res)
            print(f'  Batch {start//BATCH+1}: {len(res)} stocks downloaded')
        except:
            print(f'  Batch {start//BATCH+1}: no results')

new_data = pd.concat(all_results, ignore_index=True)
new_data['股票代码'] = new_data['股票代码'].map(reverse)
new_data = new_data.dropna(subset=['股票代码'])
new_data.to_csv('data/new_week.csv', index=False)
print(f'\nSaved {len(new_data)} rows to data/new_week.csv')
print(f'Dates: {sorted(new_data["日期"].unique())[0].date()} ~ {sorted(new_data["日期"].unique())[-1].date()}')
print(f'Stocks: {new_data["股票代码"].nunique()}')
