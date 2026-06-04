"""最终评估"""
import pandas as pd

td = pd.read_csv('data/test.csv', dtype={'股票代码': str})
od = pd.read_csv('output/result.csv', dtype={'stock_id': str})
od = od.rename(columns={'stock_id': '股票代码', 'weight': '权重'})

# Check all test codes
test_codes = set(td['股票代码'].unique())
pred_codes = set(od['股票代码'].tolist())

print(f'测试集股票数: {len(test_codes)}')
print(f'预测: {od["股票代码"].tolist()}')
print(f'预测码在测试集中: {[c in test_codes for c in od["股票代码"]]}')

# Search for specific codes
for pc in od['股票代码']:
    if pc in test_codes:
        print(f'  {pc}: 在测试集中 ✓')
    else:
        # Check partial match
        matches = [c for c in test_codes if str(pc) in str(c) or str(c) in str(pc)]
        print(f'  {pc}: 不在测试集中 ✗ 相似码: {matches[:3]}')

# If all match, calc score
if pred_codes.issubset(test_codes):
    print('\n计算得分...')
    tf = td[td['股票代码'].isin(od['股票代码'])].groupby('股票代码').tail(5)
    def cr(g):
        return (g.iloc[-1]['开盘'] - g.iloc[0]['开盘']) / g.iloc[0]['开盘']
    rets = tf.groupby('股票代码').apply(cr).reset_index().rename(columns={0:'收益率'})
    res = rets.merge(od, on='股票代码')
    fs = (res['收益率'] * res['权重']).sum()
    for _,r in res.iterrows():
        print(f'  {r["股票代码"]}: {r["收益率"]:+.4%}')
    print(f'\n===== 综合得分: {fs:.6f} = {fs:.4%} =====')
else:
    # Print sample of test codes that are 6-digit
    six_digit = [c for c in test_codes if len(c) == 6]
    print(f'\n测试集6位码: {six_digit[:20]}')
    three_digit = [c for c in test_codes if len(c) <= 4]
    print(f'测试集短码: {three_digit[:20]}')
