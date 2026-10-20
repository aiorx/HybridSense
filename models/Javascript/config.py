#!/usr/bin/env python3
"""
AI代码检测系统 - JavaScript/TypeScript 专属配置文件
"""

DATA_PATHS = {
    'human_code_dir': r"/HUBU-AI022/ai-human_code_classfication/human_written_code/progf/Javascript",
    'ai_code_dir': r"/HUBU-AI022/ai-human_code_classfication/ai_generated_code/progf/Javascript"
}

OUTPUT_PATHS = {
    'features_dir': "data/processed/features_js_m",
    'models_dir': "data/processed/models_js",
    'results_dir': "results_js",
}

FEATURES = {
    'lexical': True,
    'syntactic': True,        # 仅保留 AST 标量结构特征
    'js_specific': True,
    'human_error': True,
    'codeberta': True
}

MODEL = {
    'type': "ensemble",
    'save_model': True,
    'cv_folds': 5,
}

# 专为 JS 密集型强特征（60~80维）调优的抗噪参数
LIGHTGBM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 5,              # 密集特征限制深度防过拟合
    'num_leaves': 25,
    'subsample': 0.8,
    'colsample_bytree': 0.8,     # 提高列采样，让每一个强特征充分发挥
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': 6,
    'verbose': -1
}

VISUALIZATION = {
    'dpi': 300,
    'show_plots': False
}
