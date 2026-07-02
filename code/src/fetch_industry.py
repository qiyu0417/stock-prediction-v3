"""
获取沪深300成分股的申万行业分类
输出: data/industry.csv (股票代码, industry_code, sector, industry_name)
"""
import baostock as bs
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')


def main():
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"登录失败: {lg.error_msg}")
    print("baostock登录成功")

    # 获取全市场行业分类
    print("获取全市场行业分类...")
    rs = bs.query_stock_industry()
    if rs.error_code != '0':
        raise Exception(f"行业查询失败: {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    df_ind = pd.DataFrame(rows, columns=rs.fields)
    df_ind['code'] = df_ind['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    print(f"全市场行业数据: {len(df_ind)} 条")

    # 加载沪深300成分股列表
    hs300_path = os.path.join(DATA_DIR, 'hs300_stock_list.csv')
    if os.path.exists(hs300_path):
        hs300 = pd.read_csv(hs300_path)
        hs300['code'] = hs300['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
        target_codes = set(hs300['code'].unique())
    else:
        # 从训练数据中获取股票列表
        train_path = os.path.join(DATA_DIR, 'train.csv')
        if os.path.exists(train_path):
            train = pd.read_csv(train_path)
            target_codes = set(train['股票代码'].astype(str).str.zfill(6).unique())
        else:
            target_codes = set(df_ind['code'].unique())

    print(f"目标股票数: {len(target_codes)}")

    # 过滤到目标股票
    df_target = df_ind[df_ind['code'].isin(target_codes)].copy()

    # 清理行业数据
    df_target['industry'] = df_target['industry'].fillna('').str.strip()
    df_target['sector'] = df_target['industry'].str[0].where(
        df_target['industry'].str.len() > 0, 'Z')

    # 统计
    n_unknown = (df_target['sector'] == 'Z').sum()
    n_sectors = df_target['sector'].nunique()
    print(f"沪深300行业数据: {len(df_target)} 条")
    print(f"  门类数: {n_sectors} (含未知Z)")
    print(f"  无行业分类: {n_unknown} 只")

    # 保存
    out_path = os.path.join(DATA_DIR, 'industry.csv')
    df_target[['code', 'industry', 'sector', 'industryClassification']].rename(
        columns={'code': '股票代码', 'industry': 'industry_code',
                 'industryClassification': 'csrc_class', 'sector': 'sector'}
    ).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已保存: {out_path}")

    bs.logout()


if __name__ == '__main__':
    main()
