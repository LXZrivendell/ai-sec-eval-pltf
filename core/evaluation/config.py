from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class EvaluationConfig:
    """统一的评估配置类"""
    sample_size: int = 1000
    batch_size: int = 32
    max_memory_usage: float = 0.8  # 最大内存使用率
    enable_memory_optimization: bool = True
    save_adversarial_samples: bool = True
    
    def validate(self) -> bool:
        """验证配置参数"""
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < self.max_memory_usage <= 1:
            raise ValueError("max_memory_usage must be between 0 and 1")
        return True
    
    def get_optimal_batch_size(self, total_samples: int, available_memory_gb: float) -> int:
        """根据内存情况计算最优批次大小"""
        # 简化的内存估算：假设每个样本需要约4MB内存
        max_batch_by_memory = int(available_memory_gb * 1024 * self.max_memory_usage / 4)
        optimal_batch = min(self.batch_size, max_batch_by_memory, total_samples)
        return max(1, optimal_batch)