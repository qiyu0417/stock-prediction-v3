#!/usr/bin/env python3
"""
获取沪深300指数成分股历史数据
- 获取2026年2月20日沪深300的300个成分股
- 抓取每只股票从2015年至今的历史量价数据
- 使用baostock平台
- 保存格式: 股票代码,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌额,换手率,涨跌幅
"""

import baostock as bs
import pandas as pd
from datetime import datetime
import os
import time


def login():
    """登录baostock"""
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"登录失败: {lg.error_msg}")
    print("baostock登录成功")
    return lg


def logout():
    """登出baostock"""
    bs.logout()
    print("baostock已登出")


def get_hs300_stocks():
    """获取沪深300成分股列表"""
    print("正在获取沪深300成分股列表...")
    
    rs = bs.query_hs300_stocks()
    
    if rs.error_code != '0':
        raise Exception(f"获取成分股失败: {rs.error_msg}")
    
    stocks = []
    while (rs.error_code == '0') & rs.next():
        stocks.append(rs.get_row_data())
    
    df = pd.DataFrame(stocks, columns=rs.fields)
    print(f"获取到 {len(df)} 只沪深300成分股")
    return df


def get_stock_history(bs_code, start_date, end_date):
    """获取单只股票历史数据"""
    rs = bs.query_history_k_data_plus(bs_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="1")  # adjustflag="1"表示后复权
    
    if rs.error_code != '0':
        raise Exception(f"查询失败: {rs.error_msg}")
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    if not data_list:
        return None
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 转换数据类型
    numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 计算振幅和涨跌额
    df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(2)
    df['涨跌额'] = (df['close'] - df['preclose']).round(2)
    
    # 转换日期格式 YYYY/M/D
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y/%-m/%-d')
    
    # 提取纯数字股票代码（统一为6位格式，不足前面补0）
    df['code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
    df['code'] = df['code'].str.zfill(6)
    
    # 重命名列
    df = df.rename(columns={
        'code': '股票代码',
        'date': '日期',
        'open': '开盘',
        'close': '收盘',
        'high': '最高',
        'low': '最低',
        'volume': '成交量',
        'amount': '成交额',
        'turn': '换手率',
        'pctChg': '涨跌幅'
    })
    
    columns = ['股票代码', '日期', '开盘', '收盘', '最高', '最低', 
               '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅']
    df = df[columns]
    
    return df


def get_existing_stocks(output_path):
    """获取已经保存的股票代码列表"""
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_csv(output_path)
        if '股票代码' in df.columns and len(df) > 0:
            return set(df['股票代码'].unique())
    except:
        pass
    return set()


def get_stock_date_range(output_path, stock_code, start_date=None, end_date=None):
    """获取某只股票在现有数据中的日期范围（可限定目标时间窗）"""
    if not os.path.exists(output_path):
        return None, None
    try:
        df = pd.read_csv(output_path)
        if '股票代码' not in df.columns or '日期' not in df.columns:
            return None, None
        stock_df = df[df['股票代码'].astype(str).str.zfill(6) == stock_code].copy()
        if len(stock_df) == 0:
            return None, None

        # 解析日期
        stock_df.loc[:, '日期_dt'] = pd.to_datetime(stock_df['日期'], format='%Y/%m/%d', errors='coerce')
        stock_df = stock_df.dropna(subset=['日期_dt'])
        if len(stock_df) == 0:
            return None, None

        # 若设置了目标时间窗，仅统计目标区间内的数据覆盖情况
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            stock_df = stock_df[stock_df['日期_dt'] >= start_dt]
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            stock_df = stock_df[stock_df['日期_dt'] <= end_dt]
        if len(stock_df) == 0:
            return None, None

        return stock_df['日期_dt'].min().strftime('%Y-%m-%d'), stock_df['日期_dt'].max().strftime('%Y-%m-%d')
    except Exception as e:
        print(f"  警告: 读取股票 {stock_code} 现有日期范围失败: {e}")
        return None, None


def parse_api_date(date_str):
    """将API返回的日期 YYYY-MM-DD 转为 datetime"""
    return datetime.strptime(date_str, '%Y-%m-%d')


def format_api_date(dt):
    """将datetime转为API日期格式 YYYY-MM-DD"""
    return dt.strftime('%Y-%m-%d')


def filter_data_by_date_range(df, start_date, end_date):
    """过滤DataFrame，仅保留目标时间窗内的数据"""
    if df is None or df.empty:
        return df

    if '日期' not in df.columns:
        return df

    filtered = df.copy()
    filtered.loc[:, '日期_dt'] = pd.to_datetime(filtered['日期'], errors='coerce')
    filtered = filtered.dropna(subset=['日期_dt'])

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    filtered = filtered[(filtered['日期_dt'] >= start_dt) & (filtered['日期_dt'] <= end_dt)].copy()
    filtered = filtered.drop(columns=['日期_dt'])
    return filtered


def merge_stock_data(existing_df, new_df, stock_code):
    """合并现有数据和新数据，保持同一股票数据相邻"""
    if new_df is None or new_df.empty:
        return existing_df
    
    # 统一股票代码为6位字符串格式用于比较
    existing_df['股票代码_str'] = existing_df['股票代码'].astype(str).str.zfill(6)
    
    # 从现有数据中移除该股票的旧数据
    other_df = existing_df[existing_df['股票代码_str'] != stock_code].drop(columns=['股票代码_str'])
    
    # 获取该股票的现有数据
    stock_existing = existing_df[existing_df['股票代码_str'] == stock_code].drop(columns=['股票代码_str']) if stock_code in existing_df['股票代码_str'].values else pd.DataFrame()
    
    # 合并该股票的新旧数据
    if not stock_existing.empty:
        # 将日期转为datetime用于比较和去重
        stock_existing_copy = stock_existing.copy()
        new_df_copy = new_df.copy()
        stock_existing_copy['日期_dt'] = pd.to_datetime(stock_existing_copy['日期'], format='%Y/%m/%d')
        new_df_copy['日期_dt'] = pd.to_datetime(new_df_copy['日期'], format='%Y/%m/%d')
        
        # 合并并去重
        combined = pd.concat([stock_existing_copy, new_df_copy], ignore_index=True)
        combined = combined.drop_duplicates(subset=['日期_dt'], keep='last')
        combined = combined.sort_values('日期_dt')
        
        # 删除临时列
        combined = combined.drop(columns=['日期_dt'])
    else:
        combined = new_df
    
    # 重新组装：其他股票数据 + 该股票合并后的数据
    result = pd.concat([other_df, combined], ignore_index=True)
    return result


def main():
    save_dir = "./data"
    os.makedirs(save_dir, exist_ok=True)
    
    start_date = "2024-01-01"
    end_date = "2026-06-04"
    
    output_path = os.path.join(save_dir, "stock_data.csv")
    
    print(f"目标数据时间范围: {start_date} 至 {end_date}")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    
    # 检查已有的数据
    existing_stocks = get_existing_stocks(output_path)
    if existing_stocks:
        print(f"发现已有数据，包含 {len(existing_stocks)} 只股票，将检查每只股票是否需要增量更新")
    
    # 登录baostock
    login()
    
    try:
        # 获取沪深300成分股
        hs300_df = get_hs300_stocks()
        
        # 保存成分股列表
        hs300_list_path = os.path.join(save_dir, "hs300_stock_list.csv")
        hs300_df.to_csv(hs300_list_path, index=False, encoding='utf-8-sig')
        
        # 读取现有数据（用于增量合并）
        existing_df = None
        if os.path.exists(output_path) and len(existing_stocks) > 0:
            try:
                existing_df = pd.read_csv(output_path)
                raw_len = len(existing_df)
                existing_df = filter_data_by_date_range(existing_df, start_date, end_date)
                filtered_len = len(existing_df)
                print(f"  已加载现有数据: {len(existing_df)} 条记录")
                if filtered_len != raw_len:
                    print(f"  已按目标区间过滤旧数据: {raw_len} -> {filtered_len}")
            except Exception as e:
                print(f"  警告: 读取现有数据失败: {e}")
        
        # 准备处理所有股票（统一为6位字符串格式）
        hs300_df['纯代码'] = hs300_df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
        
        # 统计信息
        failed_stocks = []
        total = len(hs300_df)
        success_count = 0
        new_stock_count = 0
        incremental_count = 0
        total_new_records = 0
        
        for idx, row in hs300_df.iterrows():
            bs_code = row.get('code', '')
            stock_name = row.get('code_name', '')
            pure_code = row.get('纯代码', '')
            
            # 检查该股票是否已存在数据
            existing_min_date, existing_max_date = get_stock_date_range(output_path, pure_code, start_date, end_date)
            
            if existing_min_date and existing_max_date:
                # 已有数据，检查是否需要增量
                need_early = existing_min_date > start_date
                need_late = existing_max_date < end_date
                
                if not need_early and not need_late:
                    print(f"\n[{idx+1}/{total}] {bs_code} {stock_name} - 数据已完整 ({existing_min_date} 至 {existing_max_date})，跳过")
                    continue
                
                print(f"\n[{idx+1}/{total}] {bs_code} {stock_name} - 增量更新")
                print(f"  现有数据范围: {existing_min_date} 至 {existing_max_date}")
                
                # 计算需要获取的日期范围
                fetch_ranges = []
                if need_early:
                    fetch_start = start_date
                    fetch_end = (datetime.strptime(existing_min_date, '%Y-%m-%d') - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    fetch_ranges.append((fetch_start, fetch_end, "早期"))
                if need_late:
                    late_start = datetime.strptime(existing_max_date, '%Y-%m-%d') + pd.Timedelta(days=1)
                    fetch_start = max(pd.to_datetime(start_date), pd.to_datetime(late_start)).strftime('%Y-%m-%d')
                    fetch_end = end_date
                    fetch_ranges.append((fetch_start, fetch_end, "近期"))
            else:
                # 全新股票
                print(f"\n[{idx+1}/{total}] {bs_code} {stock_name} - 全新获取")
                fetch_ranges = [(start_date, end_date, "全量")]
            
            try:
                all_new_data = []
                for fetch_start, fetch_end, period_name in fetch_ranges:
                    print(f"  获取{period_name}数据: {fetch_start} 至 {fetch_end}")
                    stock_data = get_stock_history(bs_code, fetch_start, fetch_end)
                    if stock_data is not None and not stock_data.empty:
                        all_new_data.append(stock_data)
                
                if all_new_data:
                    new_data = pd.concat(all_new_data, ignore_index=True)
                    
                    if existing_df is not None and len(existing_df) > 0:
                        # 增量更新：合并数据并保持同一股票相邻
                        existing_df = merge_stock_data(existing_df, new_data, pure_code)
                        # 立即写回文件
                        existing_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                        incremental_count += 1
                    else:
                        # 首次写入
                        new_data.to_csv(output_path, index=False, encoding='utf-8-sig')
                        existing_df = new_data
                        new_stock_count += 1
                    
                    total_new_records += len(new_data)
                    success_count += 1
                    print(f"  ✓ 获取成功，新增 {len(new_data)} 条记录")
                else:
                    print(f"  ✗ 无新数据")
                    
            except Exception as e:
                print(f"  ✗ 失败: {e}")
                failed_stocks.append((bs_code, stock_name))
            
            # 每10只成功获取的股票暂停一下
            if success_count > 0 and success_count % 10 == 0:
                print(f"\n  --- 已处理 {success_count} 只，暂停2秒 ---")
                time.sleep(2)
        
        # 显示结果
        print("\n" + "=" * 60)
        print("本次运行完成!")
        print(f"  - 全新获取: {new_stock_count} 只股票")
        print(f"  - 增量更新: {incremental_count} 只股票")
        print(f"  - 失败: {len(failed_stocks)} 只股票")
        print(f"  - 新增记录: {total_new_records}")
        
        # 验证总数据
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            print(f"\n文件总览:")
            print(f"  - 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            print(f"  - 总行数: {len(df)}")
            print(f"  - 股票数量: {df['股票代码'].nunique()}")
            if len(df) > 0:
                print(f"  - 时间范围: {df['日期'].min()} 至 {df['日期'].max()}")
                
                # 验证同一股票数据是否相邻
                stock_blocks = df.groupby('股票代码').apply(lambda x: x.index.max() - x.index.min() + 1).sum()
                if stock_blocks == len(df):
                    print("  - 数据组织: ✓ 同一股票数据相邻")
                else:
                    print(f"  - 数据组织: 警告，股票数据块总长度({stock_blocks})与总行数({len(df)})不一致")
                
                print("\n前3行数据预览:")
                print(df.head(3).to_string(index=False))
                print("\n最后3行数据预览:")
                print(df.tail(3).to_string(index=False))
        
        # 保存失败列表
        if failed_stocks:
            failed_df = pd.DataFrame(failed_stocks, columns=['股票代码', '股票名称'])
            failed_path = os.path.join(save_dir, "failed_stocks.csv")
            failed_df.to_csv(failed_path, index=False, encoding='utf-8-sig')
            print(f"\n失败股票列表已保存至: {failed_path}")
    
    finally:
        logout()


if __name__ == "__main__":
    main()
