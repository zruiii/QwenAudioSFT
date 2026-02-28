import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class MultimodalTimeSeriesIndexer:
    """
    多模态时间序列数据索引生成器
    用于生成结合时序数据和新闻数据的训练样本索引
    """
    
    def __init__(self, 
                 ts_path: str = 'data/v1/ts.csv',
                 news_path: str = 'data/v1/news.csv',
                 lookback_days: int = 30,
                 prediction_days: int = 5,
                 min_news_count: int = 0,
                 duplicate_news_dropout: float = 0.8,
                 max_no_news_ratio: float = 0.3,
                 min_cv_threshold: float = 0.1,
                 random_seed: int = 42):
        """
        参数:
            ts_path: 时序数据文件路径
            news_path: 新闻数据文件路径
            lookback_days: 回溯天数(用于训练的历史数据长度)
            prediction_days: 预测天数(需要预测的未来数据长度)
            min_news_count: 最小新闻数量(样本至少需要的新闻条数)
            duplicate_news_dropout: 新闻完全相同的样本的丢弃比例(0-1之间)
            max_no_news_ratio: 无新闻样本的最大比例(0-1之间)
            min_cv_threshold: 输入时序的标准差阈值
            random_seed: 随机种子,用于可复现性
        """
        self.ts_path = ts_path
        self.news_path = news_path
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.min_news_count = min_news_count
        self.duplicate_news_dropout = duplicate_news_dropout
        self.max_no_news_ratio = max_no_news_ratio
        self.random_seed = random_seed
        self.min_cv_threshold = min_cv_threshold
        
        self.ts_df = None
        self.news_df = None
        
        # 设置随机种子
        np.random.seed(random_seed)
        
    def load_data(self):
        """加载并预处理时序和新闻数据"""
        print("加载时序数据...")
        self.ts_df = pd.read_csv(self.ts_path)
        self.ts_df['date'] = pd.to_datetime(self.ts_df['date'])
        
        print("加载新闻数据...")
        self.news_df = pd.read_csv(self.news_path, low_memory=False)
        self.news_df['date'] = pd.to_datetime(self.news_df['date'])
        
        print(f"时序数据: {len(self.ts_df)} 行")
        print(f"新闻数据: {len(self.news_df)} 行")
        print(f"时序数据包含 {self.ts_df['item'].nunique()} 个股票")
        print(f"新闻数据包含 {self.news_df['item'].nunique()} 个股票")
        
    def generate_index(self) -> pd.DataFrame:
        """
        生成训练样本索引表
        
        返回:
            包含所有训练样本索引的DataFrame
        """
        if self.ts_df is None or self.news_df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        index_list = []
        no_news_samples = []
        
        # 按股票代号分组处理
        items = self.ts_df['item'].unique()
        print(f"\n开始生成索引,共 {len(items)} 个股票...")
        
        for idx, item in enumerate(items):
            if (idx + 1) % 10 == 0:
                print(f"处理进度: {idx + 1}/{len(items)}")

            # warning: 临时代码
            if idx > 500:
                break

            # 获取该股票的时序数据和新闻数据
            item_ts = self.ts_df[self.ts_df['item'] == item].reset_index(drop=True)
            item_news = self.news_df[self.news_df['item'] == item].reset_index(drop=True)
            
            # 为时序数据添加原始索引
            item_ts['original_idx'] = self.ts_df[self.ts_df['item'] == item].index
            item_news['original_idx'] = self.news_df[self.news_df['item'] == item].index
            
            # 生成该股票的所有样本索引
            item_indices, item_no_news = self._generate_item_indices(item_ts, item_news, item)
            index_list.extend(item_indices)
            no_news_samples.extend(item_no_news)
        
        print(f"\n初步生成:")
        print(f"  有新闻样本: {len(index_list)} 个")
        print(f"  无新闻样本: {len(no_news_samples)} 个")
        
        # 控制无新闻样本的比例
        final_indices = self._balance_no_news_samples(index_list, no_news_samples)
        
        print(f"\n总共生成 {len(final_indices)} 个训练样本")
        
        # 转换为DataFrame
        index_df = pd.DataFrame(final_indices, columns=[
            'item', 'ts_start_id', 'ts_end_id', 'ts_pred_id', 
            'news_start_id', 'news_end_id', 'news_count'
        ])
        
        return index_df
    
    def _generate_item_indices(self, 
                              item_ts: pd.DataFrame, 
                              item_news: pd.DataFrame,
                              item: str) -> Tuple[list, list]:
        """
        为单个股票生成所有可能的训练样本索引
        
        参数:
            item_ts: 该股票的时序数据
            item_news: 该股票的新闻数据
            item: 股票代号
            
        返回:
            (有新闻的样本列表, 无新闻的样本列表)
        """
        has_news_indices = []
        no_news_indices = []
        min_cv_threshold = getattr(self, 'min_cv_threshold', 0.01)
        
        # 需要至少 lookback_days + prediction_days 的数据
        min_required_rows = self.lookback_days + self.prediction_days
        
        if len(item_ts) < min_required_rows:
            return has_news_indices, no_news_indices
        
        # 用于追踪新闻组合相同的样本
        news_groups = {}  # key: (news_start_id, news_end_id), value: 样本列表
        
        # 滑动窗口生成样本
        for i in range(len(item_ts) - min_required_rows + 1):
            ts_start_idx = i
            ts_end_idx = i + self.lookback_days - 1
            ts_pred_idx = i + self.lookback_days + self.prediction_days - 1

            # 如果有缺失值 & 变异系数太小，则删除
            full_window_values = item_ts.iloc[ts_start_idx : ts_pred_idx + 1]['Close']
            if full_window_values.isna().any() or (full_window_values <= 0).any():
                continue
                
            input_values = full_window_values.iloc[:self.lookback_days]
            mu = input_values.mean()
            sigma = input_values.std()
            cv = sigma / mu if mu != 0 else 0
            if cv < min_cv_threshold:
                continue
            
            # 获取时间范围
            start_date = item_ts.iloc[ts_start_idx]['date']
            end_date = item_ts.iloc[ts_end_idx]['date']
            
            # 查找该时间段内的新闻
            news_mask = (item_news['date'] >= start_date) & (item_news['date'] <= end_date)
            period_news = item_news[news_mask]
            
            # 构建样本字典
            sample = {
                'item': item,
                'ts_start_id': int(item_ts.iloc[ts_start_idx]['original_idx']),
                'ts_end_id': int(item_ts.iloc[ts_end_idx]['original_idx']),
                'ts_pred_id': int(item_ts.iloc[ts_pred_idx]['original_idx']),
                'news_start_id': -1,
                'news_end_id': -1,
                'news_count': 0
            }
            
            # 检查新闻数量
            if len(period_news) == 0 or len(period_news) < self.min_news_count:
                # 无新闻样本
                no_news_indices.append(sample)
            else:
                # 有新闻样本
                news_start_id = int(period_news.iloc[0]['original_idx'])
                news_end_id = int(period_news.iloc[-1]['original_idx'])
                news_count = len(period_news)
                
                sample['news_start_id'] = news_start_id
                sample['news_end_id'] = news_end_id
                sample['news_count'] = news_count
                
                # 按新闻范围分组
                news_key = (news_start_id, news_end_id)
                if news_key not in news_groups:
                    news_groups[news_key] = []
                news_groups[news_key].append(sample)
        
        # 对每组新闻相同的样本进行dropout
        for news_key, samples in news_groups.items():
            if len(samples) > 1:
                # 有多个样本共享相同的新闻,进行随机丢弃
                keep_count = max(1, int(len(samples) * (1 - self.duplicate_news_dropout)))
                kept_samples = np.random.choice(len(samples), keep_count, replace=False)
                for idx in kept_samples:
                    has_news_indices.append(samples[idx])
            else:
                # 只有一个样本,保留
                has_news_indices.append(samples[0])
        
        return has_news_indices, no_news_indices
    
    def _balance_no_news_samples(self, has_news_samples: list, no_news_samples: list) -> list:
        """
        控制无新闻样本的比例
        
        参数:
            has_news_samples: 有新闻的样本列表
            no_news_samples: 无新闻的样本列表
            
        返回:
            平衡后的所有样本列表
        """
        total_has_news = len(has_news_samples)
        total_no_news = len(no_news_samples)
        
        if total_no_news == 0:
            print("  没有无新闻样本")
            return has_news_samples
        
        # 计算当前无新闻样本的比例
        current_ratio = total_no_news / (total_has_news + total_no_news)
        print(f"  当前无新闻样本比例: {current_ratio:.2%}")
        
        # 如果当前比例已经低于目标,保留所有样本
        if current_ratio <= self.max_no_news_ratio:
            print(f"  无新闻样本比例已低于目标 {self.max_no_news_ratio:.2%},保留所有样本")
            return has_news_samples + no_news_samples
        
        # 计算需要保留的无新闻样本数量
        if self.max_no_news_ratio >= 1.0:
            keep_no_news_count = total_no_news
        else:
            keep_no_news_count = int(total_has_news * self.max_no_news_ratio / (1 - self.max_no_news_ratio))
        
        # 随机选择要保留的无新闻样本
        if keep_no_news_count >= total_no_news:
            kept_no_news = no_news_samples
        else:
            kept_indices = np.random.choice(total_no_news, keep_no_news_count, replace=False)
            kept_no_news = [no_news_samples[i] for i in kept_indices]
        
        final_ratio = len(kept_no_news) / (total_has_news + len(kept_no_news))
        print(f"保留 {len(kept_no_news)} 个无新闻样本")
        print(f"调整后无新闻样本比例: {final_ratio:.2%}")
        
        return has_news_samples + kept_no_news
    
    def save_index(self, index_df: pd.DataFrame, output_path: str = 'data/v1/index.csv'):
        """保存索引表到CSV文件"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        index_df.to_csv(output_path, index=False)
        print(f"\n索引表已保存到: {output_path}")
        
    def print_statistics(self, index_df: pd.DataFrame):
        """打印索引表的统计信息"""
        print("\n" + "="*60)
        print("索引表统计信息")
        print("="*60)
        print(f"总样本数: {len(index_df)}")
        print(f"涵盖股票数: {index_df['item'].nunique()}")
        print(f"有新闻的样本数: {(index_df['news_count'] > 0).sum()}")
        print(f"无新闻的样本数: {(index_df['news_count'] == 0).sum()}")
        print(f"平均每个样本的新闻数: {index_df['news_count'].mean():.2f}")
        print(f"新闻数中位数: {index_df['news_count'].median():.0f}")
        print(f"最多新闻数: {index_df['news_count'].max()}")
        print("\n样本示例:")
        print(index_df.head(10))
        print("="*60)


def main():
    """主函数"""
    # 创建索引生成器
    indexer = MultimodalTimeSeriesIndexer(
        ts_path='data/v1/ts.csv',
        news_path='data/v1/news.csv',
        lookback_days=60,              
        prediction_days=20,            
        min_news_count=0,              # 允许没有新闻的样本(设为0),如需至少1条新闻则设为1
        duplicate_news_dropout=0.7,    # 新闻完全相同的样本丢弃比例
        max_no_news_ratio=0.2,         # 无新闻样本最多占比
        min_cv_threshold=0.01,
        random_seed=42                 # 随机种子,保证可复现
    )
    
    # 加载数据
    indexer.load_data()
    
    # 生成索引
    index_df = indexer.generate_index()
    
    # 打印统计信息
    indexer.print_statistics(index_df)
    
    # 保存索引表
    indexer.save_index(index_df, output_path='data/v1/index.csv')
    
    print("\n处理完成!")
    
    # 验证示例:展示如何使用索引
    print("\n" + "="*60)
    print("使用示例:")
    print("="*60)
    
    # 展示有新闻的样本
    has_news_sample = index_df[index_df['news_count'] > 0].iloc[0] if (index_df['news_count'] > 0).any() else None
    if has_news_sample is not None:
        print(f"\n有新闻样本示例:")
        print(f"  股票: {has_news_sample['item']}")
        print(f"  时序数据: 行 {has_news_sample['ts_start_id']} 到 {has_news_sample['ts_end_id']} (用于训练)")
        print(f"  预测目标: 行 {has_news_sample['ts_end_id']+1} 到 {has_news_sample['ts_pred_id']} (需要预测)")
        print(f"  新闻数据: 行 {has_news_sample['news_start_id']} 到 {has_news_sample['news_end_id']} ({has_news_sample['news_count']} 条新闻)")
    
    # 展示无新闻的样本
    no_news_sample = index_df[index_df['news_count'] == 0].iloc[0] if (index_df['news_count'] == 0).any() else None
    if no_news_sample is not None:
        print(f"\n无新闻样本示例:")
        print(f"  股票: {no_news_sample['item']}")
        print(f"  时序数据: 行 {no_news_sample['ts_start_id']} 到 {no_news_sample['ts_end_id']} (用于训练)")
        print(f"  预测目标: 行 {no_news_sample['ts_end_id']+1} 到 {no_news_sample['ts_pred_id']} (需要预测)")
        print(f"  无对应新闻数据")


if __name__ == "__main__":
    main()