import numpy as np
from typing import Dict, Any, Tuple, Optional
import streamlit as st
from .memory_manager import MemoryManager
from ..attack_manager import AttackManager

class AttackExecutor:
    """攻击执行专用类"""
    
    def __init__(self, attack_manager: AttackManager, memory_manager: MemoryManager):
        self.attack_manager = attack_manager
        self.memory_manager = memory_manager
    
    def execute_attack(self, estimator: Any, x_data: np.ndarray, y_data: np.ndarray,
                      attack_config: Dict, batch_size: int = 32) -> Tuple[np.ndarray, Dict]:
        """执行攻击"""
        try:
            # 创建攻击实例
            attack_instance = self.attack_manager.create_attack_instance(
                attack_config['algorithm'],
                attack_config['params'],
                estimator
            )
            
            if attack_instance is None:
                raise ValueError("攻击实例创建失败")
            
            # 优化批次大小
            optimal_batch_size = self.memory_manager.optimize_batch_size(
                batch_size, len(x_data)
            )
            
            # 分批生成对抗样本
            adversarial_samples = []
            attack_stats = {
                'total_batches': 0,
                'successful_batches': 0,
                'failed_batches': 0,
                'memory_cleanups': 0
            }
            
            total_batches = (len(x_data) + optimal_batch_size - 1) // optimal_batch_size
            
            for i in range(0, len(x_data), optimal_batch_size):
                batch_x = x_data[i:i+optimal_batch_size]
                batch_y = y_data[i:i+optimal_batch_size] if y_data is not None else None
                
                try:
                    # 生成对抗样本
                    if attack_config.get('targeted', False) and batch_y is not None:
                        adv_batch = attack_instance.generate(x=batch_x, y=batch_y)
                    else:
                        adv_batch = attack_instance.generate(x=batch_x)
                    
                    adversarial_samples.append(adv_batch)
                    attack_stats['successful_batches'] += 1
                    
                except Exception as e:
                    st.warning(f"批次 {i//optimal_batch_size + 1} 攻击失败: {str(e)}")
                    # 使用原始样本作为fallback
                    adversarial_samples.append(batch_x)
                    attack_stats['failed_batches'] += 1
                
                attack_stats['total_batches'] += 1
                
                # 内存监控和清理
                self.memory_manager.monitor_memory_during_processing(
                    attack_stats['total_batches'], total_batches
                )
                
                if not self.memory_manager.check_memory_limit():
                    self.memory_manager.cleanup_memory()
                    attack_stats['memory_cleanups'] += 1
                
                # 更新进度
                progress = min((i + len(batch_x)) / len(x_data), 1.0)
                st.progress(progress)
            
            # 合并结果
            adversarial_samples = np.concatenate(adversarial_samples, axis=0)
            
            st.info(f"攻击执行完成: {attack_stats['successful_batches']}/{attack_stats['total_batches']} 批次成功")
            
            return adversarial_samples, attack_stats
            
        except Exception as e:
            st.error(f"攻击执行失败: {str(e)}")
            return None, None