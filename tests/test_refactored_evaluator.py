import unittest
from core.evaluation.config import EvaluationConfig
from core.evaluation.data_processor import DataProcessor

class TestRefactoredEvaluator(unittest.TestCase):
    def test_evaluation_config(self):
        config = EvaluationConfig(sample_size=100, batch_size=16)
        self.assertTrue(config.validate())
    
    def test_data_processor(self):
        processor = DataProcessor()
        # 添加具体的测试用例
        pass