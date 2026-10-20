#!/usr/bin/env python3
"""
AI代码检测系统 - Java 专属配置文件 (领域专家特征版)
"""

DATA_PATHS = {
    'human_code_dir': r"/HUBU-AI022/ai-human_code_classfication/human_written_code/progf/Java",
    'ai_code_dir': r"/HUBU-AI022/ai-human_code_classfication/ai_generated_code/progf/Java"
}

OUTPUT_PATHS = {
    'features_dir': "data/processed/features_java",
    'models_dir': "data/processed/models_java",
    'results_dir': "results_java",
}

FEATURES = {
    'lexical': True,
    'syntactic': True,        # 仅保留 AST 标量，无 N-gram
    'java_specific': False,
    'human_error': False,
    'codeberta': False
}

MODEL = {
    'type': "ensemble",
    'save_model': True,
    'cv_folds': 5,
}

# 专为 70维左右“极高信息密度”特征调优的参数
LIGHTGBM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 5,              # 密集特征不需要太深的树
    'num_leaves': 25,
    'subsample': 0.8,
    'colsample_bytree': 0.8,     # 提高采样率，让强特征充分发挥
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
