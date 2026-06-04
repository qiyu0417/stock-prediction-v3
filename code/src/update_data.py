"""下载最新数据并更新本地train"""
import baostock as bs
import pandas as pd
import json, time, os

# 1. 找映射：用hs300真实代码下载，按日期价格匹配到编码
print('Step 1: 建立映射...')
raw_train = pd.read_csv('data/train.csv', dtype={'股票代码': str})
raw_train['日期_str'] = raw_train['日期'].astype(str).str[:10]  # normalize

# 取一天的数据做匹配
match_day = raw_train[raw_train['日期_str'] == '2024-01-02']
print(f'匹配日股票: {len(match_day)}')

# 读hs300列表获取真实代码
hs300 = pd.read_csv('data/hs300_stock_list.csv')
hs300['real_code'] = hs300['code'].str.replace('sh.','').str.replace('sz.','').str.zfill(6)

# 下载匹配日的真实数据
bs.login()
print('下载匹配日数据...')

match_data = []
for _, row in hs300.iterrows():
    rc = row['real_code']
    bs_code = row['code']
    try:
        rs = bs.query_history_k_data_plus(bs_code,
            'date,open', start_date='2024-01-02', end_date='2024-01-02',
            frequency='d', adjustflag='1')
        while (rs.error_code == '0') & rs.next():
            d = rs.get_row_data()
            match_data.append({'real_code': rc, 'open': float(d[1])})
    except:
        pass
bs.logout()

match_df = pd.DataFrame(match_data)
print(f'下载到 {len(match_df)} 只股票的匹配日数据')

# 价格映射
mapping = {}
for _, tr in match_day.iterrows():
    tc = tr['股票代码']
    to = float(tr['开盘'])
    for _, mr in match_df.iterrows():
        if abs(mr['open'] - to) < 0.5:
            mapping[tc] = mr['real_code']
            break
print(f'映射: {len(mapping)} stocks')
if len(mapping) < 200:
    print('映射不足，退出')
    exit(1)

reverse = {v: k for k, v in mapping.items()}

# 2. 下载最新数据
print('\nStep 2: 下载 2026-03-14 ~ 2026-05-31...')
bs.login()
real_codes = list(reverse.keys())
all_new = []

for i, rc in enumerate(real_codes):
    bs_code = f'sh.{rc}' if rc.startswith('6') else f'sz.{rc}'
    try:
        rs = bs.query_history_k_data_plus(bs_code,
            'date,code,open,high,low,close,preclose,volume,amount,turn,pctChg',
            start_date='2026-03-14', end_date='2026-05-31',
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
    print(f'下载: {len(new_data)}行, {new_data["日期"].min()} ~ {new_data["日期"].max()}')

    # 3. 合并
    print('\nStep 3: 合并...')
    train_df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
    combined = pd.concat([train_df, new_data], ignore_index=True)
    combined.to_csv('data/train_updated.csv', index=False)
    print(f'更新后: {len(combined)}行, {combined["日期"].min()} ~ {combined["日期"].max()}')
else:
    print('无新数据!')
