import psutil
import gc
import numpy as np
from typing import Optional
import streamlit as st

class MemoryManager:
    """内存管理类"""
    
    def __init__(self, max_memory_usage: float = 0.8):
        self.max_memory_usage = max_memory_usage
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用率"""
        return psutil.virtual_memory().percent / 100.0
    
    def get_available_memory_gb(self) -> float:
        """获取可用内存（GB）"""
        return psutil.virtual_memory().available / (1024**3)
    
    def check_memory_limit(self) -> bool:
        """检查是否超过内存限制"""
        current_usage = self.get_memory_usage()
        return current_usage < self.max_memory_usage
    
    def optimize_batch_size(self, initial_batch_size: int, total_samples: int) -> int:
        """根据内存情况优化批次大小"""
        available_memory = self.get_available_memory_gb()
        
        # 估算每个样本的内存需求（简化估算）
        memory_per_sample_mb = 4  # 假设每个样本需要4MB
        max_batch_by_memory = int(available_memory * 1024 * self.max_memory_usage / memory_per_sample_mb)
        
        optimal_batch = min(initial_batch_size, max_batch_by_memory, total_samples)
        
        if optimal_batch != initial_batch_size:
            st.info(f"根据内存情况调整批次大小: {initial_batch_size} -> {optimal_batch}")
        
        return max(1, optimal_batch)
    
    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        
    def monitor_memory_during_processing(self, current_batch: int, total_batches: int):
        """处理过程中监控内存"""
        current_usage = self.get_memory_usage()
        
        if current_usage > self.max_memory_usage:
            st.warning(f"内存使用率过高: {current_usage:.1%}，正在清理内存...")
            self.cleanup_memory()
        
        # 每10个批次显示一次内存状态
        if current_batch % 10 == 0:
            st.info(f"批次 {current_batch}/{total_batches}, 内存使用率: {current_usage:.1%}")