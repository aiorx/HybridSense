#!/usr/bin/env python3
"""
系统配置文件 - 训练与特征提取 (config.py)
"""

# ========== 数据集路径 (请核对是否与你本地一致) ==========
DATA_PATHS = {
    'human_code_dir': r"/HUBU-AI022/ai-human_code_classfication/human_written_code/progf/C",
    'ai_code_dir': r"/HUBU-AI022/ai-human_code_classfication/ai_generated_code/progf/C"
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
    'cpp_specific': True,     # 3. C特有特征
    'human_error': True,      # 4. 人类失误与习惯特征
    'codeberta': False,
    'ngram_range': [2, 3,4]     # 【必须开启】AST N-gram的范围，提供深度的语法结构差异
}

# ========== C语言特有特征词库配置 ==========
C_CONFIG = {
    'standard_libs': [
        'stdio.h', 'stdlib.h', 'string.h', 'math.h', 'ctype.h',
        'stdbool.h', 'stdint.h', 'time.h', 'assert.h', 'stdarg.h'
    ],
    'memory_funcs': ['malloc', 'calloc', 'realloc', 'free', 'memset', 'memcpy', 'memmove'],
    'unsafe_funcs': ['gets', 'strcpy', 'strcat', 'sprintf', 'scanf'],
    'safe_funcs': ['fgets', 'strncpy', 'strncat', 'snprintf']
}

# ========== 模型与训练配置 ==========
MODEL = {
    'save_model': True,
    'model_name': 'lgbm_model_cpp'
}

# 专为 400 维黄金特征调优的 LightGBM 参数 (终极抗噪版)
LIGHTGBM_PARAMS = {
    'n_estimators': 400,         # 树的数量
    'learning_rate': 0.05,       # 较小的学习率防止过拟合
    'max_depth': 6,              # 限制单棵树深度，防止过拟合
    'num_leaves': 31,            # 配合深度
    'subsample': 0.8,            # 样本采样率
    'colsample_bytree': 0.7,     # 【关键】回调特征采样率到0.7，配合 SelectKBest
    'reg_alpha': 1.0,            # 降低L1惩罚，因为已经用SelectKBest做过强力过滤
    'reg_lambda': 1.0,
    'class_weight': 'balanced',  # 自动平衡人类和AI样本
    'random_state': 42,
    'n_jobs': 6,                 # 限制 CPU 核心数，绝对防止电脑死机
    'importance_type': 'gain',   # 以信息增益作为特征重要性指标
    'verbose': -1                # 屏蔽多余的底层日志
}

# ========== 可视化图表配置 ==========
VISUALIZATION = {
    'dpi': 300,               # 图表清晰度 (300适合论文/报告)
    'show_plots': False       # 是否在运行时弹窗显示图片 (设为False则直接保存到硬盘，防止卡住)
}
