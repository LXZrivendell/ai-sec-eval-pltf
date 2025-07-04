class DefenseMetrics:
    def calculate_adversarial_accuracy_gap(self, original_acc: float, adversarial_acc: float) -> float:
        """计算对抗精度差距"""
        return float(original_acc - adversarial_acc)
    
    def calculate_purification_recovery_rate(self, original_acc: float, purified_acc: float, adversarial_acc: float) -> float:
        """计算净化恢复率"""
        if original_acc == adversarial_acc:
            return 1.0  # 没有攻击损失
        recovery_rate = (purified_acc - adversarial_acc) / (original_acc - adversarial_acc)
        return float(max(0, min(1, recovery_rate)))
    
    def calculate_defense_effectiveness(self, baseline_attack_success: float, defended_attack_success: float) -> float:
        """计算防御有效性"""
        if baseline_attack_success == 0:
            return 1.0
        return float(1 - defended_attack_success / baseline_attack_success)
    
    def calculate_clean_accuracy_preservation(self, original_acc: float, defended_clean_acc: float) -> float:
        """计算干净样本准确率保持度"""
        return float(defended_clean_acc / original_acc) if original_acc > 0 else 0