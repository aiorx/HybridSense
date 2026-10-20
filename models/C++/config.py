#!/usr/bin/env python3
"""
系统配置文件 - 训练与特征提取 (config.py)
"""

# ========== 数据集路径 (请核对是否与你本地一致) ==========
DATA_PATHS = {
    'human_code_dir': r"/HUBU-AI022/ai-human_code_classfication/human_written_code/progf/C++",
    'ai_code_dir': r"/HUBU-AI022/ai-human_code_classfication/ai_generated_code/progf/C++"
}

# ========== 输出路径配置 ==========
OUTPUT_PATHS = {
    'features_dir': r"./data/processed/features_cpp",
    'models_dir': r"./data/processed/models_cpp",
    'results_dir': r"./results_cpp"
}

# ========== 特征提取模块开关 ==========
FEATURES = {
    'lexical': True,          # 1. 词法特征
    'syntactic': True,        # 2. 语法特征 (AST N-gram 与 结构标量)
    'cpp_specific': True,     # 3. C++特有特征
    'human_error': True,      # 4. 人类失误与习惯特征 (我们刚刚新增的杀手锏)
    'codeberta': False,
    'ngram_range': [2, 3,4]     # AST N-gram的范围 
}

# ========== C++特有特征词库配置 ==========
CPP_CONFIG = {
    'stl_containers': [
        'vector', 'array', 'deque', 'list', 'forward_list', 'stack', 'queue',
        'priority_queue', 'set', 'multiset', 'map', 'multimap', 'unordered_set',
        'unordered_multiset', 'unordered_map', 'unordered_multimap', 'bitset'
    ],
    'stl_algorithms': [
        'sort', 'find', 'binary_search', 'copy', 'move', 'transform', 'accumulate',
        'for_each', 'count', 'max_element', 'min_element', 'reverse', 'unique'
    ],
    'smart_pointers': ['unique_ptr', 'shared_ptr', 'weak_ptr', 'auto_ptr'],
}

# ========== 模型与训练配置 ==========
MODEL = {
    'save_model': True,
    'model_name': 'lgbm_model_cpp'
}

# 专为400维黄金特征调优的 LightGBM 参数
LIGHTGBM_PARAMS = {
    'n_estimators': 400,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.7,     # 配合 SelectKBest 调高采样率
    'reg_alpha': 1.0,            # L1 正则化抗噪
    'reg_lambda': 1.0,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': 6,
    'importance_type': 'gain',
    'verbose': -1
}

# ========== 可视化图表配置 ==========
VISUALIZATION = {
    'dpi': 300,
    'show_plots': False
}
