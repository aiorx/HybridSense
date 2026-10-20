#!/usr/bin/env python3
"""
AI代码检测系统 - TypeScript 专属配置文件 (领域专家特征版)
"""

DATA_PATHS = {
    'human_code_dir': r"/HUBU-AI022/ai-human_code_classfication/human_written_code/progf/Typescript",
    'ai_code_dir': r"/HUBU-AI022/ai-human_code_classfication/ai_generated_code/progf/Typescript"
}

OUTPUT_PATHS = {
    'features_dir': "data/processed/features_ts",
    'models_dir': "data/processed/models_ts",
    'results_dir': "results_ts",
}

FEATURES = {
    'lexical': True,
    'syntactic': True,
    'ts_specific': True,
    'human_error': True,
    'codeberta': True
}

# ========== 【新增】CodeBERTa 模型配置 ==========
CODEBERTA_MODEL = "huggingface/CodeBERTa-small-v1"

MODEL = {
    'type': "ensemble",
    'save_model': True,
    'cv_folds': 5,
}

# 针对 TypeScript 160+ 维特征调优的 LightGBM 参数
LIGHTGBM_PARAMS = {
    'n_estimators': 450,
    'learning_rate': 0.05,
    'max_depth': 8,              # TS特征较多，稍微放宽深度
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.5,            # L1 正则，自动忽略无用特征
    'reg_lambda': 1.0,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

VISUALIZATION = {
    'dpi': 300,
    'show_plots': False
}
