"""
合并原始数据
"""

import pandas as pd

hs_news_df = pd.read_parquet("data/FinMultiTime-hs300/text.parquet")
sp_news_df = pd.read_parquet("data/FinMultiTime-sp500/text.parquet")

hs_ts_df = pd.read_parquet("data/FinMultiTime-hs300/time_series.parquet")
sp_ts_df = pd.read_parquet("data/FinMultiTime-sp500/time_series.parquet")

# 新闻表
hs_news_df = hs_news_df.rename(columns={'publication_date': 'date'})
hs_news_df = hs_news_df[['item', 'date', 'title', 'summary']]
hs_news_df['date'] = pd.to_datetime(hs_news_df['date'])
hs_news_df = hs_news_df.groupby('item', group_keys=False).apply(
    lambda x: x.sort_values('date')
).reset_index(drop=True)

sp_news_df = sp_news_df.rename(columns={'publication_date': 'date', 'Article_title': 'title', 'Article': 'article'})
sp_news_df = sp_news_df[['item', 'date', 'title', 'article']]
sp_news_df['date'] = pd.to_datetime(sp_news_df['date'])
sp_news_df = sp_news_df.groupby('item', group_keys=False).apply(
    lambda x: x.sort_values('date')
).reset_index(drop=True)

news_df = pd.concat((hs_news_df, sp_news_df))

# 去除 title 为空的
news_df = news_df[~news_df['title'].isna()].reset_index(drop=True)    
news_df = news_df[~(news_df['title'] == '')].reset_index(drop=True)
print(f"新闻条目数目: {news_df.shape[0]}, 平均每个item新闻数目: {int(news_df.shape[0] / len(set(news_df['item'])))}, HS300: {hs_news_df.shape[0]}, SP500: {sp_news_df.shape[0]}")

# 时序表
hs_ts_df['date'] = pd.to_datetime(hs_ts_df['date'])
hs_ts_df = hs_ts_df.groupby('item', group_keys=False).apply(
    lambda x: x.sort_values('date')
).reset_index(drop=True)

sp_ts_df['date'] = pd.to_datetime(sp_ts_df['date'])
sp_ts_df = sp_ts_df.groupby('item', group_keys=False).apply(
    lambda x: x.sort_values('date')
).reset_index(drop=True)

hs_ts_df = hs_ts_df[['item', 'date', 'Close']]
sp_ts_df = sp_ts_df[['item', 'date', 'Close']]

ts_df = pd.concat((hs_ts_df, sp_ts_df))
print(f"平均每个item时间点: {int(ts_df.shape[0] / len(set(ts_df['item'])))}, HS300: {int(hs_ts_df.shape[0] / len(set(hs_ts_df['item'])))}, SP500: {int(sp_ts_df.shape[0] / len(set(sp_ts_df['item'])))}")

news_df.to_csv("data/v1/news.csv", index=False)
ts_df.to_csv("data/v1/ts.csv", index=False)
