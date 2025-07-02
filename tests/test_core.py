import unittest
import os
import sys
import tempfile
import shutil
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心模块
try:
    from core.auth_manager import AuthManager
    from core.model_loader import ModelLoader
    from core.dataset_manager import DatasetManager
    from core.attack_manager import AttackManager
    from core.security_evaluator import SecurityEvaluator
    from core.report_generator import ReportGenerator
except ImportError as e:
    print(f"导入核心模块失败: {e}")
    sys.exit(1)

class TestCoreModules(unittest.TestCase):
    """核心模块测试类"""
    
    def setUp(self):
        """测试前准备"""
        print("\n" + "="*60)
        print("开始测试 AI模型安全评估平台 核心模块")
        print("="*60)
        
        # 创建临时测试目录
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # 创建必要的目录结构
        os.makedirs('data/models/upload', exist_ok=True)
        os.makedirs('data/models/pretrained', exist_ok=True)
        os.makedirs('data/datasets/builtin', exist_ok=True)
        os.makedirs('data/datasets/uploaded', exist_ok=True)
        os.makedirs('data/attack_configs', exist_ok=True)
        os.makedirs('data/evaluation_results', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # 初始化核心管理器
        self.auth_manager = AuthManager()
        self.model_loader = ModelLoader()
        self.dataset_manager = DatasetManager()
        self.attack_manager = AttackManager()
        self.security_evaluator = SecurityEvaluator()
        self.report_generator = ReportGenerator()
    
    def tearDown(self):
        """测试后清理"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_01_auth_manager(self):
        """测试认证管理器"""
        print("\n🔐 测试认证管理器...")
        
        # 测试用户注册
        success, message = self.auth_manager.register_user('testuser', 'password123', 'test@example.com')
        self.assertTrue(success, f"用户注册失败: {message}")
        
        # 测试用户认证
        auth_success, auth_message = self.auth_manager.authenticate_user('testuser', 'password123')
        self.assertTrue(auth_success, f"用户认证失败: {auth_message}")
        
        # 测试获取用户信息
        user_info = self.auth_manager.get_user_info('testuser')
        self.assertIsInstance(user_info, dict, "用户信息应该是字典类型")
        self.assertEqual(user_info.get('email'), 'test@example.com', "用户邮箱不匹配")
        self.assertIn('role', user_info, "用户信息应该包含角色字段")
        
        # 测试会话管理
        session_id = self.auth_manager.create_session('testuser')
        self.assertIsNotNone(session_id, "会话创建失败")
        
        # 测试会话验证
        is_valid = self.auth_manager.validate_session(session_id)
        self.assertTrue(is_valid, "会话验证失败")
        
        print("   ✅ 认证管理器测试通过")
    
    def test_02_model_loader(self):
        """测试模型加载器"""
        print("\n📦 测试模型加载器...")
        
        # 测试获取用户模型（返回字典类型）
        user_models = self.model_loader.get_user_models('testuser')
        self.assertIsInstance(user_models, dict, "用户模型列表应该是字典类型")
        
        # 测试获取所有模型
        all_models = self.model_loader.get_all_models()
        self.assertIsInstance(all_models, dict, "所有模型列表应该是字典类型")
        
        # 测试存储统计
        stats = self.model_loader.get_storage_stats()
        self.assertIsInstance(stats, dict, "存储统计应该是字典类型")
        self.assertIn('total_models', stats, "统计信息应该包含总模型数")
        
        print("   ✅ 模型加载器测试通过")
    
    def test_03_dataset_manager(self):
        """测试数据集管理器"""
        print("\n📊 测试数据集管理器...")
        
        # 测试加载数据集信息
        datasets_info = self.dataset_manager.load_datasets_info()
        self.assertIsInstance(datasets_info, dict, "数据集信息应该是字典类型")
        
        # 测试获取所有数据集（返回字典类型）
        all_datasets = self.dataset_manager.get_all_datasets()
        self.assertIsInstance(all_datasets, dict, "所有数据集列表应该是字典类型")
        
        # 验证内置数据集存在
        builtin_datasets = [k for k in all_datasets.keys() if k.startswith('builtin_')]
        self.assertGreater(len(builtin_datasets), 0, "应该有内置数据集")
        
        print("   ✅ 数据集管理器测试通过")
    
    def test_04_attack_manager(self):
        """测试攻击管理器"""
        print("\n⚔️ 测试攻击管理器...")
        
        # 测试获取用户配置
        user_configs = self.attack_manager.get_user_configs('testuser')
        self.assertIsInstance(user_configs, list, "用户配置列表应该是列表类型")
        
        # 测试获取攻击算法（返回字典类型）
        attack_algorithms = self.attack_manager.get_attack_algorithms()
        self.assertIsInstance(attack_algorithms, dict, "攻击算法列表应该是字典类型")
        self.assertGreater(len(attack_algorithms), 0, "应该有可用的攻击算法")
        
        # 测试按类型获取攻击
        evasion_attacks = self.attack_manager.get_attack_by_type('evasion')
        self.assertIsInstance(evasion_attacks, dict, "按类型获取的攻击应该是字典类型")
        
        # 测试获取攻击信息
        fgsm_info = self.attack_manager.get_attack_info('FGSM')
        self.assertIsNotNone(fgsm_info, "应该能获取FGSM攻击信息")
        self.assertIsInstance(fgsm_info, dict, "攻击信息应该是字典类型")
        
        print("   ✅ 攻击管理器测试通过")
    
    def test_05_security_evaluator(self):
        """测试安全评估器"""
        print("\n🛡️ 测试安全评估器...")
        
        # 测试安全评估器初始化
        self.assertIsNotNone(self.security_evaluator.model_loader, "模型加载器应该被初始化")
        self.assertIsNotNone(self.security_evaluator.dataset_manager, "数据集管理器应该被初始化")
        self.assertIsNotNone(self.security_evaluator.attack_manager, "攻击管理器应该被初始化")
        
        print("   ✅ 安全评估器测试通过")
    
    def test_06_report_generator(self):
        """测试报告生成器"""
        print("\n📊 测试报告生成器...")
        
        # 创建测试评估数据 - 添加缺少的必需字段
        test_results = {
            'name': 'test_evaluation',
            'type': 'security_evaluation',  # 添加缺少的 type 字段
            'evaluation_id': 'test_001',
            'timestamp': '2024-01-01T00:00:00',
            'model_info': {'name': 'test_model'},
            'dataset_info': {'name': 'test_dataset'},
            'attack_config': {'algorithm': 'FGSM'},
            'config': {  # 添加 config 字段以避免模板数据准备时的错误
                'model': {'name': 'test_model', 'framework': 'tensorflow'},
                'dataset': {'name': 'test_dataset'},
                'parameters': {'sample_size': 100},
                'attack_configs': [{'algorithm': 'FGSM'}]
            },
            'results': {
                'original_accuracy': 0.95,
                'adversarial_accuracy': 0.75,
                'attack_success_rate': 0.20,
                'average_perturbation': 0.05,
                'robustness_score': 0.8,
                'security_level': 'medium'
            }
        }
        
        # 测试报告生成 - 使用正确的参数名
        report_path = self.report_generator.generate_report(test_results, format_type='html')
        self.assertIsNotNone(report_path, "报告生成应该成功")
        
        # 测试获取报告列表
        report_list = self.report_generator.get_report_list()
        self.assertIsInstance(report_list, list, "报告列表应该是列表类型")
        
        print("   ✅ 报告生成器测试通过")
    
    def test_07_integration(self):
        """测试模块集成"""
        print("\n🔗 测试模块集成...")
        
        # 注册集成测试用户
        success, message = self.auth_manager.register_user('integrationuser', 'password123', 'integration@example.com')
        self.assertTrue(success, f"集成测试用户注册失败: {message}")
        
        # 验证用户信息
        user_info = self.auth_manager.get_user_info('integrationuser')
        self.assertIsInstance(user_info, dict, "用户信息应该是字典类型")
        self.assertEqual(user_info.get('email'), 'integration@example.com', "跨模块用户邮箱不一致")
        
        # 测试模块间的数据一致性（返回字典类型）
        user_models = self.model_loader.get_user_models('integrationuser')
        self.assertIsInstance(user_models, dict, "跨模块模型列表类型应该是字典")
        
        # 测试数据集管理器集成
        all_datasets = self.dataset_manager.get_all_datasets()
        self.assertIsInstance(all_datasets, dict, "跨模块数据集列表类型应该是字典")
        
        print("   ✅ 模块集成测试通过")

def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCoreModules)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 生成测试报告
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    print(f"总测试数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)} ✅")
    print(f"失败: {len(result.failures)} ❌")
    print(f"错误: {len(result.errors)} ⚠️")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            # 修复f-string中的反斜杠问题
            newline = '\n'
            error_msg = traceback.split('AssertionError: ')[-1].split(newline)[0]
            print(f"  - {test}: {error_msg}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            newline = '\n'
            error_msg = traceback.split(newline)[-2] if newline in traceback else traceback
            print(f"  - {test}: {error_msg}")
    
    if result.failures or result.errors:
        print(f"\n⚠️ 有 {len(result.failures) + len(result.errors)} 个测试未通过，请检查相关功能。")
    else:
        print("\n🎉 所有测试通过！")
    
    print("="*60)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)