"""映射股票编码到真实代码和名称"""
import json, pandas as pd

with open('data/code_mapping.json') as f:
    mp = json.load(f)
to_real = mp['to_real']

hs300 = pd.read_csv('data/hs300_stock_list.csv')
hs300['code'] = hs300['code'].str.replace('sh.','').str.replace('sz.','').str.zfill(6)
code_name = dict(zip(hs300['code'], hs300['code_name']))

pred_june  = ['603986', '300442', '300502', '688256', '688126']
pred_test  = ['300274', '300394', '300033', '300502', '603986']

print('=== 6月1-5日 预测持仓 ===')
for pe in pred_june:
    real = to_real.get(pe, pe)
    name = code_name.get(real, '未知')
    print(f'  {pe} -> {real}  {name}')

print()
print('=== 测试集选股 (得分4.73%) ===')
for pe in pred_test:
    real = to_real.get(pe, pe)
    name = code_name.get(real, '未知')
    print(f'  {pe} -> {real}  {name}')
