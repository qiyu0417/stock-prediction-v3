"""评估集成模型在官方测试集上的得分"""
import pandas as pd
import sys

# 读取数据
test_data = pd.read_csv('data/test.csv', dtype={'股票代码': str})
test_data['股票代码'] = test_data['股票代码'].astype(str).str.zfill(6)

output_data = pd.read_csv('output/result.csv')
output_data = output_data.rename(columns={'stock_id': '股票代码', 'weight': '权重'})

# 验证预测格式
print(f"预测股票数: {len(output_data)}")
print(f"权重和: {output_data['权重'].sum():.2f}")
if len(output_data) > 5:
    print("警告: 超过5只股票!")
    sys.exit(1)

# 计算5日收益率
def calc_return(group):
    start = group.iloc[0]
    end = group.iloc[-1]
    return (end['开盘'] - start['开盘']) / start['开盘']

test_filtered = test_data[test_data['股票代码'].isin(output_data['股票代码'])]
if len(test_filtered) == 0:
    print("错误: 预测股票不在测试集中!")
    print("预测:", output_data['股票代码'].tolist())
    sys.exit(1)

test_filtered = test_filtered.groupby('股票代码').tail(5)
groups = test_filtered.groupby('股票代码')

print("\n各股票5日收益率:")
returns_dict = {}
for name, group in groups:
    r = calc_return(group)
    returns_dict[name] = r
    print(f"  {name}: {r:+.4%} ({r:+.6f})")

returns_df = pd.DataFrame(
    [(k, v) for k, v in returns_dict.items()],
    columns=['股票代码', '收益率']
)
result = returns_df.merge(output_data, on='股票代码')

# 综合得分 = 加权收益率之和
final_score = (result['收益率'] * result['权重']).sum()

print(f"\n{'='*50}")
print(f"综合得分 (加权5日收益率): {final_score:.6f}")
print(f"综合得分 (百分比):        {final_score:.4%}")
print(f"{'='*50}")

# 对比baseline
print(f"\n参考: 原始baseline得分约 0.025 (2.5%)")
print(f"提升: {(final_score/0.025 - 1)*100:+.1f}%" if final_score > 0 else "N/A")
