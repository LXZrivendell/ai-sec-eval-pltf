# Tests模块初始化文件
"""
AI模型安全评估平台 - 测试模块

本模块包含平台的测试用例：
- test_core.py: 核心模块测试
"""

# 测试配置
TEST_CONFIG = {
    'test_data_dir': 'test_data',
    'temp_dir': 'temp_test',
    'mock_models': True,
    'verbose': True
}

# 测试用例分类
TEST_CATEGORIES = {
    'unit': '单元测试',
    'integration': '集成测试',
    'performance': '性能测试',
    'security': '安全测试'
}