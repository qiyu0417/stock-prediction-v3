import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="按日期区间将股票数据切分为 train.csv 和 test.csv"
	)
	parser.add_argument(
		"--input",
		type=str,
		default="data/stock_data.csv",
		help="原始数据文件路径，默认 data/stock_data.csv",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="data",
		help="输出目录，默认 data",
	)
	parser.add_argument(
		"--train-start",
		type=str,
		default="2024-01-02",
		help="训练集开始日期，默认 2024-01-02",
	)
	parser.add_argument(
		"--train-end",
		type=str,
		default="2026-03-06",
		help="训练集结束日期，默认 2026-03-06",
	)
	parser.add_argument(
		"--test-start",
		type=str,
		default="2026-03-09",
		help="测试集开始日期，默认 2026-03-09",
	)
	parser.add_argument(
		"--test-end",
		type=str,
		default="2026-03-13",
		help="测试集结束日期，默认 2026-03-13",
	)
	return parser.parse_args()


def _to_timestamp(date_str: str, name: str) -> pd.Timestamp:
	ts = pd.to_datetime(date_str, errors="coerce")
	if pd.isna(ts):
		raise ValueError(f"参数 {name} 的日期格式无效: {date_str}")
	return ts.normalize()


def _validate_columns(df: pd.DataFrame) -> None:
	required = {"股票代码", "日期"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"输入文件缺少必要列: {sorted(missing)}")


def _filter_by_date(
	df: pd.DataFrame,
	start_date: pd.Timestamp,
	end_date: pd.Timestamp,
) -> pd.DataFrame:
	if start_date > end_date:
		raise ValueError(f"开始日期晚于结束日期: {start_date.date()} > {end_date.date()}")

	mask = (df["日期"] >= start_date) & (df["日期"] <= end_date)
	out = df.loc[mask].copy()
	out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)
	out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
	return out


def main() -> None:
	args = parse_args()

	input_path = Path(args.input)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	train_start = _to_timestamp(args.train_start, "--train-start")
	train_end = _to_timestamp(args.train_end, "--train-end")
	test_start = _to_timestamp(args.test_start, "--test-start")
	test_end = _to_timestamp(args.test_end, "--test-end")

	df = pd.read_csv(input_path)
	_validate_columns(df)

	df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
	if df["日期"].isna().any():
		bad_rows = int(df["日期"].isna().sum())
		raise ValueError(f"原始数据中存在无法解析的日期，共 {bad_rows} 行")
	source_min_date = df["日期"].min().date()
	source_max_date = df["日期"].max().date()

	train_df = _filter_by_date(df, train_start, train_end)
	test_df = _filter_by_date(df, test_start, test_end)

	train_path = output_dir / "train.csv"
	test_path = output_dir / "test.csv"

	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)

	print(f"训练集: {train_path}，共 {len(train_df)} 行，股票数 {train_df['股票代码'].nunique()}")
	print(f"测试集: {test_path}，共 {len(test_df)} 行，股票数 {test_df['股票代码'].nunique()}")
	print(
		f"训练集日期范围: {train_start.date()} ~ {train_end.date()} | "
		f"测试集日期范围: {test_start.date()} ~ {test_end.date()}"
	)
	if train_df.empty or test_df.empty:
		print(
			"警告: 训练集或测试集为空，请检查日期范围是否与原始数据重叠。"
		)
		print(f"原始数据日期范围: {source_min_date} ~ {source_max_date}")


if __name__ == "__main__":
	main()
