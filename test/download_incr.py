"""Incremental download — use REAL stock codes for baostock queries"""
import json, pandas as pd, baostock as bs, os, time

with open('data/stock_mapping.json') as f: mapping = json.load(f)
real_codes = sorted(mapping.keys())
output_file = 'data/new_week.csv'

done_codes = set()
if os.path.exists(output_file):
    existing = pd.read_csv(output_file, dtype={'股票代码': str})
    done_codes = set(existing['股票代码'].unique())
    print(f'Resuming: {len(done_codes)} stocks already done')

bs.login()
count = 0; new_count = 0
for real_code in real_codes:
    if real_code in done_codes:
        count += 1
        continue

    bsc = f'sh.{real_code}' if real_code.startswith('6') else f'sz.{real_code}'
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
            df_new['股票代码'] = real_code
            df_new['日期'] = pd.to_datetime(df_new['date'])
            df_new = df_new.rename(columns={'open':'开盘','close':'收盘','high':'最高','low':'最低',
                'volume':'成交量','amount':'成交额','turn':'换手率','pctChg':'涨跌幅'})
            out = df_new[['股票代码','日期','开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌额','换手率','涨跌幅']]
            out.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            new_count += 1
    except Exception as e:
        pass

    count += 1
    if count % 50 == 0:
        print(f'  {count}/{len(real_codes)}... new={new_count}')

bs.logout()
print(f'Done! {count} processed, {new_count} new stocks')
