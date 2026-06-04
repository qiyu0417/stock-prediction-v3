"""检查训练数据和测试数据的股票代码格式"""
import pandas as pd
for name in ['data/train.csv', 'data/test.csv']:
    df = pd.read_csv(name, dtype={'股票代码': str})
    print(f'{name}: {df["股票代码"].nunique()} stocks, codes: {sorted(df["股票代码"].unique())[:5]}...')

df2 = pd.read_csv('output/result.csv')
print(f'output: {df2["stock_id"].tolist()}')
print(f'test has these stocks: {set(df2["stock_id"].tolist()).issubset(set(pd.read_csv("data/test.csv", dtype={"股票代码":str})["股票代码"].unique()))}')
