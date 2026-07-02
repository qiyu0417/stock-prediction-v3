"""Fetch latest data (6/19-6/26) from baostock and append to new_week.csv"""
import baostock as bs
import pandas as pd
import numpy as np
import time, os

print('Step 1: Build stock code mapping...')
raw_train = pd.read_csv('data/train.csv', dtype={'股票代码': str})
match_day = raw_train[raw_train['日期'].astype(str).str[:10] == '2024-01-02']

hs300 = pd.read_csv('data/hs300_stock_list.csv')
hs300['real_code'] = hs300['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)

# Download match day data
bs.login()
match_data = []
for _, row in hs300.iterrows():
    rc = row['real_code']
    try:
        rs = bs.query_history_k_data_plus(row['code'],
            'date,open', start_date='2024-01-02', end_date='2024-01-02',
            frequency='d', adjustflag='1')
        while (rs.error_code == '0') & rs.next():
            d = rs.get_row_data()
            match_data.append({'real_code': rc, 'open': float(d[1])})
    except:
        pass
bs.logout()

match_df = pd.DataFrame(match_data)
mapping = {}
for _, tr in match_day.iterrows():
    tc = tr['股票代码']
    to = float(tr['开盘'])
    for _, mr in match_df.iterrows():
        if abs(mr['open'] - to) < 0.5:
            mapping[tc] = mr['real_code']
            break
reverse = {v: k for k, v in mapping.items()}
print(f'Mapped {len(mapping)} stocks')

# Step 2: Download 6/19-6/27
print('\nStep 2: Downloading 2026-06-19 ~ 2026-06-27...')
bs.login()
real_codes = list(reverse.keys())
all_new = []

for i, rc in enumerate(real_codes):
    bs_code = f'sh.{rc}' if rc.startswith('6') else f'sz.{rc}'
    try:
        rs = bs.query_history_k_data_plus(bs_code,
            'date,code,open,high,low,close,preclose,volume,amount,turn,pctChg',
            start_date='2026-06-19', end_date='2026-06-27',
            frequency='d', adjustflag='1')
        data = []
        while (rs.error_code == '0') & rs.next():
            data.append(rs.get_row_data())
        if data:
            df = pd.DataFrame(data, columns=rs.fields)
            for c in ['open','high','low','close','preclose','volume','amount','turn','pctChg']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df['振幅'] = ((df['high']-df['low'])/df['preclose']*100).round(2)
            df['涨跌额'] = (df['close']-df['preclose']).round(2)
            df['code'] = rc
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df.rename(columns={
                'code':'股票代码','date':'日期','open':'开盘','close':'收盘',
                'high':'最高','low':'最低','volume':'成交量','amount':'成交额',
                'turn':'换手率','pctChg':'涨跌幅'
            })
            all_new.append(df[['股票代码','日期','开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌额','换手率','涨跌幅']])
    except:
        pass
    if (i+1) % 50 == 0:
        print(f'  {i+1}/{len(real_codes)}')

bs.logout()

if all_new:
    new_data = pd.concat(all_new, ignore_index=True)
    new_data['股票代码'] = new_data['股票代码'].map(reverse)
    new_data = new_data.dropna(subset=['股票代码'])
    new_data = new_data.sort_values(['日期','股票代码']).reset_index(drop=True)
    print(f'\nDownloaded: {len(new_data)} rows')
    print(f'Dates: {sorted(new_data["日期"].unique())}')
    print(f'Stocks: {new_data["股票代码"].nunique()}')

    # Append to new_week.csv
    old = pd.read_csv('data/new_week.csv', dtype={'股票代码': str})
    combined = pd.concat([old, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=['股票代码','日期'], keep='last')
    combined = combined.sort_values(['日期','股票代码']).reset_index(drop=True)
    combined.to_csv('data/new_week.csv', index=False)
    print(f'Updated new_week.csv: {len(old)} -> {len(combined)} rows')
    print(f'New dates: {sorted(new_data["日期"].unique())}')
else:
    print('No new data found! (market may not have 6/19+ data yet)')
