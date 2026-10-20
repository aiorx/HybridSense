#!/usr/bin/env python3
"""
Python 特征提取模块 - 终极增强融合版
严格分为四大框架类：词法、AST语法、Python专有、人类失误。
包含原版 240+ 维度的动态特征与领域专家鉴谎特征。
"""

import logging
import numpy as np
from typing import List, Dict, Set, Tuple
import ast
import statistics
import token
import tokenize
import io
import re
import math
from collections import Counter


logger = logging.getLogger(__name__)

# ====== 【新增】CodeBERTa 依赖 ======
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("未安装 torch 或 transformers！CodeBERTa 将退回全零矩阵。建议: pip install torch transformers")


# ==============================================================================
# 1. 词法特征提取器 (Lexical)
# 包含关键字分布、标点符号密度、变量命名风格、Halstead估算与香农熵
# ==============================================================================
class LexicalFeatureExtractor:
    def __init__(self, code: str):
        self.code = code
        self.length_chars = max(len(self.code), 1)
        self.lines = code.split('\n')
        self.total_lines = max(len(self.lines), 1)

        self.PYTHON_KEYWORDS = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
            'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
            'try', 'while', 'with', 'yield'
        }
        self.CORE_KEYWORDS = {
            'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'try',
            'except', 'finally', 'with', 'import', 'from', 'return', 'yield',
            'async', 'await'
        }
        self.BUILTIN_OBJECTS = {
            'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
            'sum', 'min', 'max', 'abs', 'round', 'any', 'all', 'sorted',
            'open', 'int', 'float', 'str', 'bool', 'list', 'dict', 'set',
            'tuple', 'type', 'isinstance', 'issubclass', 'Exception', 'ValueError'
        }
        self.operators = {'+', '-', '*', '/', '%', '**', '//', '=', '+=', '-=', '*=', '/=', '%=',
                          '==', '!=', '>', '<', '>=', '<=', 'and', 'or', 'not', 'is', 'in', '&', '|', '^', '~', '<<',
                          '>>'}

        try:
            self.tokens = []
            for tok in tokenize.generate_tokens(io.StringIO(self.code).readline):
                self.tokens.append(tok)
        except (tokenize.TokenError, IndentationError, TabError, SyntaxError, Exception) as e:
            # 🚨 终极防弹层：捕获包括 Tab混用、语法严重畸形在内的所有底层解析错误
            logger.debug(f"跳过不规范代码的Token解析: {e}")
            self.tokens = None

    def get_feature_names(self) -> List[str]:
        names = [
            'log_keywords_per_char', 'log_words_per_char', 'log_comments_per_char',
            'avg_line_length', 'std_line_length', 'max_line_length',
            'empty_lines_density', 'comment_density', 'avg_indent_level',
            'py_keyword_ratio', 'operator_density', 'string_literal_density',
            'number_literals_density', 'builtin_usage_density',
            'shannon_entropy_char', 'avg_id_length', 'identifier_length_variance',
            'halstead_volume_norm', 'halstead_difficulty'
        ]
        # 动态添加17维关键字频率
        names.extend([f'{kw}_density' for kw in self.CORE_KEYWORDS])
        return names

    def extract(self) -> Dict[str, float]:
        features = {name: 0.0 for name in self.get_feature_names()}

        # 1. 基础样式与行特征
        line_lengths = [len(l) for l in self.lines]
        indents = [len(l) - len(l.lstrip()) for l in self.lines if l.strip()]
        features['avg_line_length'] = statistics.mean(line_lengths) if line_lengths else 0
        features['std_line_length'] = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0
        features['max_line_length'] = max(line_lengths) if line_lengths else 0
        features['empty_lines_density'] = sum(1 for l in self.lines if not l.strip()) / self.total_lines
        features['comment_density'] = sum(1 for l in self.lines if l.strip().startswith('#')) / self.total_lines
        features['avg_indent_level'] = statistics.mean(indents) if indents else 0

        # 2. Token 分析
        if self.tokens:
            keyword_counts = {kw: 0 for kw in self.CORE_KEYWORDS}
            total_keywords = 0
            unique_keywords = set()
            literals = {'string': 0, 'number': 0}
            identifiers = []

            for tok in self.tokens:
                if tok.type == token.NAME and tok.string in self.PYTHON_KEYWORDS:
                    total_keywords += 1
                    unique_keywords.add(tok.string)
                    if tok.string in keyword_counts:
                        keyword_counts[tok.string] += 1
                elif tok.type == token.STRING:
                    literals['string'] += 1
                elif tok.type == token.NUMBER:
                    literals['number'] += 1
                elif tok.type == token.NAME and tok.string not in self.PYTHON_KEYWORDS:
                    identifiers.append(tok.string)

            for kw in self.CORE_KEYWORDS:
                features[f'{kw}_density'] = keyword_counts[kw] / self.length_chars

            features['py_keyword_ratio'] = total_keywords / max(len(identifiers), 1)
            features['string_literal_density'] = literals['string'] / self.length_chars
            features['number_literals_density'] = literals['number'] / self.length_chars
            features['builtin_usage_density'] = sum(
                1 for i in identifiers if i in self.BUILTIN_OBJECTS) / self.length_chars

            id_lengths = [len(i) for i in identifiers]
            features['avg_id_length'] = statistics.mean(id_lengths) if id_lengths else 0
            features['identifier_length_variance'] = statistics.variance(id_lengths) if len(id_lengths) > 1 else 0

        # 3. 正则补充 (因为 token 解析可能失败)
        words = re.findall(r'\b[a-zA-Z_]\w*\b', self.code)
        comment_lines = len(re.findall(r'#.*', self.code))
        features['log_keywords_per_char'] = np.log(
            (features.get('py_keyword_ratio', 0) * len(words)) / self.length_chars + 1e-10)
        features['log_words_per_char'] = np.log(len(words) / self.length_chars + 1e-10)
        features['log_comments_per_char'] = np.log(comment_lines / self.length_chars + 1e-10)
        features['operator_density'] = sum(self.code.count(op) for op in self.operators) / self.length_chars

        # 4. 信息熵与 Halstead 复杂度
        char_counts = Counter(self.code)
        features['shannon_entropy_char'] = -sum(
            (c / self.length_chars) * math.log2(c / self.length_chars) for c in char_counts.values() if c > 0)

        tokens_str = re.findall(r'\b\w+\b|[^\w\s]', self.code)
        ops = [t for t in tokens_str if t in self.operators]
        operands = [t for t in tokens_str if re.match(r'\b\w+\b', t) and t not in self.operators]
        n1, n2 = len(set(ops)), len(set(operands))
        N1, N2 = len(ops), len(operands)
        n, N = n1 + n2 + 1e-10, N1 + N2 + 1e-10
        features['halstead_volume_norm'] = (N * math.log2(n) if n >= 1 else 0) / self.length_chars
        features['halstead_difficulty'] = (n1 / 2) * (N2 / (n2 + 1e-10))

        return features


# ==============================================================================
# 2. 语法特征提取器 (Syntactic)
# 完全基于 Python ast 库，包含结构深度与高达 150+ 维的节点频率展开
# ==============================================================================
class SyntacticFeatureExtractor:
    def __init__(self, code: str):
        self.code = code
        self.length_chars = max(len(self.code), 1)
        self.AST_NODE_TYPES = [
            'Module', 'Interactive', 'Expression', 'FunctionDef', 'AsyncFunctionDef',
            'ClassDef', 'Return', 'Delete', 'Assign', 'AugAssign', 'AnnAssign',
            'For', 'AsyncFor', 'While', 'If', 'With', 'AsyncWith', 'Match', 'Raise',
            'Try', 'Assert', 'Import', 'ImportFrom', 'Global', 'Nonlocal', 'Expr',
            'Pass', 'Break', 'Continue', 'BoolOp', 'NamedExpr', 'BinOp', 'UnaryOp',
            'Lambda', 'IfExp', 'Dict', 'Set', 'ListComp', 'SetComp', 'DictComp',
            'GeneratorExp', 'Await', 'Yield', 'YieldFrom', 'Compare', 'Call',
            'FormattedValue', 'JoinedStr', 'Constant', 'Attribute', 'Subscript',
            'Starred', 'Name', 'List', 'Tuple', 'Slice'
        ]
        try:
            self.ast_tree = ast.parse(self.code)
        except Exception:
            self.ast_tree = None

    def get_feature_names(self) -> List[str]:
        names = ['total_ast_nodes', 'max_ast_depth', 'ast_node_density',
                 'ast_branching_ratio', 'ast_loop_ratio', 'avg_attribute_depth']
        for node_type in self.AST_NODE_TYPES:
            names.extend([f'nttf_{node_type}', f'ntad_{node_type}', f'ntmd_{node_type}'])
        return names

    def extract(self) -> Dict[str, float]:
        features = {name: 0.0 for name in self.get_feature_names()}
        if not self.ast_tree: return features

        stats = {'total': 0, 'max_depth': 0, 'branch': 0, 'loop': 0, 'attr_depths': []}
        node_depths = {nt: [] for nt in self.AST_NODE_TYPES}

        def traverse(node, depth):
            stats['total'] += 1
            stats['max_depth'] = max(stats['max_depth'], depth)

            ntype = type(node).__name__
            if ntype in node_depths:
                node_depths[ntype].append(depth)

            if isinstance(node, (ast.If, ast.Try, ast.With, ast.Match)):
                stats['branch'] += 1
            elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                stats['loop'] += 1
            elif isinstance(node, ast.Attribute):
                d = 0
                curr = node
                while isinstance(curr, ast.Attribute):
                    d += 1
                    curr = curr.value
                stats['attr_depths'].append(d)

            for child in ast.iter_child_nodes(node):
                traverse(child, depth + 1)

        traverse(self.ast_tree, 0)

        total_nodes = stats['total'] + 1e-10
        features['total_ast_nodes'] = stats['total'] / self.length_chars
        features['max_ast_depth'] = stats['max_depth']
        features['ast_node_density'] = stats['total'] / self.length_chars
        features['ast_branching_ratio'] = stats['branch'] / total_nodes
        features['ast_loop_ratio'] = stats['loop'] / total_nodes
        features['avg_attribute_depth'] = statistics.mean(stats['attr_depths']) if stats['attr_depths'] else 0.0

        for node_type, depths in node_depths.items():
            if depths:
                features[f'nttf_{node_type}'] = len(depths) / self.length_chars
                features[f'ntad_{node_type}'] = statistics.mean(depths)
                features[f'ntmd_{node_type}'] = statistics.median(depths)

        return features


# ==============================================================================
# 3. Python 专属特性提取器 (PythonSpecific)
# 包含 Python 高级语法特征、类型提示、修饰器与 CP（竞技编程）特有库调用
# ==============================================================================
class PythonSpecificFeatureExtractor:
    def __init__(self, code: str):
        self.code = code
        self.length_chars = max(len(self.code), 1)
        self.total_lines = max(len(self.code.split('\n')), 1)

    def get_feature_names(self) -> List[str]:
        return [
            'type_hint_density', 'comprehension_density', 'decorator_density',
            'docstring_ratio', 'lambda_density', 'with_context_density',
            'main_guard_flag', 'dataclass_or_pydantic_density',
            'cp_fast_io_density', 'cp_sys_recursion_density',
            'cp_collections_density', 'cp_math_density', 'cp_itertools_density'
        ]

    def extract(self) -> Dict[str, float]:
        code = self.code
        lc = self.length_chars
        tl = self.total_lines

        type_hints = len(re.findall(r':\s*[A-Z][a-zA-Z0-9_]*\[|->\s*[A-Z]', code))
        comprehensions = len(re.findall(r'\[.+? for .+? in .+?\]|\{.+? for .+? in .+?\}', code))
        decorators = len(re.findall(r'^\s*@[\w\.]+', code, re.MULTILINE))
        docstrings = len(re.findall(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', code))
        main_guard = 1.0 if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", code) else 0.0

        return {
            'type_hint_density': type_hints / lc,
            'comprehension_density': comprehensions / lc,
            'decorator_density': decorators / lc,
            'docstring_ratio': docstrings / tl,
            'lambda_density': len(re.findall(r'\blambda\s+', code)) / lc,
            'with_context_density': len(re.findall(r'\bwith\s+.+?:', code)) / lc,
            'main_guard_flag': main_guard,
            'dataclass_or_pydantic_density': len(re.findall(r'@(?:dataclass|validator|root_validator)', code)) / lc,

            'cp_fast_io_density': len(re.findall(r'\bsys\.stdin\.read\b|\bsys\.stdout\.write\b', code)) / lc,
            'cp_sys_recursion_density': len(re.findall(r'\bsys\.setrecursionlimit\b', code)) / lc,
            'cp_collections_density': len(re.findall(r'\b(defaultdict|deque|Counter|heapq)\b', code)) / lc,
            'cp_math_density': len(re.findall(r'\bmath\.(gcd|ceil|floor|sqrt)\b', code)) / lc,
            'cp_itertools_density': len(re.findall(r'\b(permutations|combinations|accumulate)\b', code)) / lc
        }


# ==============================================================================
# 4. 人类失误与排版特征 (HumanError)
# 包含人类拼写错误、魔法数字、空 except、嵌套结构以及用于防伪装的排版制衡
# ==============================================================================
class HumanErrorFeatureExtractor:
    def __init__(self, code: str):
        self.code = code
        self.code_lower = code.lower()
        self.lines = code.split('\n')
        self.total_lines = max(len(self.lines), 1)
        self.length_chars = max(len(self.code), 1)

        self.common_typos = {'lenght', 'heigth', 'recieve', 'occured', 'sucess', 'retun', 'funtion', 'ture', 'flase',
                             'paramter'}
        self.informal_tags = {'todo', 'fixme', 'hack', 'wtf', 'shit', 'fuck', 'resources', 'lol', 'test'}
        self.human_atypical = {'aids', 'composed', 'development', 'materials', 'segments', 'utilities', 'content', 'samples' , 'blocks', 'via', 'third-party', 'borrowed'}
        self.human_vars = {'tmp', 'temp', 'foo', 'bar', 'res', 'ans', 'val', 'arr', 'idx', 'cnt'}

    def get_feature_names(self) -> List[str]:
        return [
            'typo_density', 'human_atypical_density', 'informal_tag_density', 'commented_code_ratio',
            'empty_except_ratio', 'print_debug_density',
            'mixed_quotes_ratio', 'consecutive_identical_call_ratio',
            'perfect_indent_with_crowded_ops', 'magic_number_density',
            'human_var_density', 'nested_functions_density', 'comprehension_nesting_density'
        ]

    def extract(self) -> Dict[str, float]:
        lc = self.length_chars
        tl = self.total_lines
        code = self.code

        total_except = len(re.findall(r'\bexcept\b', code)) + 1e-10
        # 防死循环：避免在正则中使用 (*?) 跨行搜索，改为限制字符
        empty_except = len(re.findall(r'except[^:\n]*:\s*\n\s*pass\b', code))

        single_q = len(re.findall(r"'[^']*'", code))
        double_q = len(re.findall(r'"[^"]*"', code))
        mixed_quotes = min(single_q, double_q) / (single_q + double_q + 1e-10)

        debugs = len(re.findall(r'\bprint\s*\(\s*["\']\s*(?:test|debug|here|111|aaa|ok)\s*["\']\s*\)', self.code_lower))
        magic_nums = len([n for n in re.findall(r'\b\d+\b', code) if n not in ('0', '1', '2')]) / lc

        stripped_lines = [l.strip() for l in self.lines if l.strip()]
        consecutive_calls = 0
        for i in range(len(stripped_lines) - 2):
            m1 = re.match(r'^([a-zA-Z_][\w\.]*)\s*\(', stripped_lines[i])
            m2 = re.match(r'^([a-zA-Z_][\w\.]*)\s*\(', stripped_lines[i + 1])
            m3 = re.match(r'^([a-zA-Z_][\w\.]*)\s*\(', stripped_lines[i + 2])
            if m1 and m2 and m3 and (m1.group(1) == m2.group(1) == m3.group(1)):
                consecutive_calls += 1

        crowded_ops = len(re.findall(r'[a-zA-Z0-9_](==|!=|<=|>=|\+=|-=|\*=|/=|=(?!=)|<|>)[a-zA-Z0-9_]', code)) / lc
        irregular_indent = sum(
            1 for l in self.lines if l.startswith(' ') and (len(l) - len(l.lstrip(' '))) % 4 != 0) / tl

        # 线性扫描提取嵌套函数 (安全、无回溯)
        nested_funcs = 0
        def_indent = -1
        for line in self.lines:
            stripped = line.lstrip()
            if stripped.startswith('def '):
                indent = len(line) - len(stripped)
                if def_indent != -1 and indent > def_indent:
                    nested_funcs += 1
                def_indent = indent
            elif stripped.startswith('class '):
                def_indent = -1

        human_var_count = sum(len(re.findall(r'\b' + var + r'\b', self.code_lower)) for var in self.human_vars)
        comprehension_nesting = len(re.findall(r'for\s+[^\n]+\s+in\s+[^\n]+\s+for\s+', code))



        return {
            'typo_density': sum(len(re.findall(r'\b' + t + r'\b', self.code_lower)) for t in self.common_typos) / lc,
            'human_atypical_density': sum(
                len(re.findall(r'\b' + t + r'\b', self.code_lower)) for t in self.human_atypical) / lc,
            'informal_tag_density': sum(
                len(re.findall(r'\b' + t + r'\b', self.code_lower)) for t in self.informal_tags) / lc,
            'commented_code_ratio': sum(1 for l in self.lines if re.match(r'^\s*#\s*[a-zA-Z_]', l)) / tl,
            'empty_except_ratio': empty_except / total_except,
            'print_debug_density': debugs / lc,
            'mixed_quotes_ratio': mixed_quotes,
            'consecutive_identical_call_ratio': consecutive_calls / (len(stripped_lines) + 1e-10),
            'perfect_indent_with_crowded_ops': 1.0 if irregular_indent < 0.05 and crowded_ops > 0.01 else 0.0,
            'magic_number_density': magic_nums,
            'human_var_density': human_var_count / lc,
            'nested_functions_density': nested_funcs / tl,
            # 'has_ai_generation_tag': has_ai_tag,
            'comprehension_nesting_density': comprehension_nesting / lc
        }


# ==============================================================================
# 5. 【新增】深度语义特征提取器 (CodeBERTa)
# ==============================================================================
class CodeBERTaFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.model_name = getattr(config, 'CODEBERTA_MODEL', "huggingface/CodeBERTa-small-v1")
        self.hidden_size = 768

        if HAS_TRANSFORMERS:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            logger.info(f"加载 CodeBERTa 模型 ({self.model_name}) 到设备: {self.device}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                self.is_ready = True
                logger.info("✓ 成功加载 CodeBERTa 骨干网络")
            except Exception as e:
                logger.error(f"❌ 加载 CodeBERTa 失败: {e}")
                self.is_ready = False
        else:
            self.is_ready = False

    def get_feature_names(self) -> List[str]:
        return [f"codeberta_dim_{i}" for i in range(self.hidden_size)]

    def extract_batch(self, codes: List[str], batch_size: int = 16) -> np.ndarray:
        if not self.is_ready or not codes:
            logger.warning("CodeBERTa 未就绪，返回全 0 特征矩阵。")
            return np.zeros((len(codes), self.hidden_size))

        all_features = []
        logger.info(f"开始提取 CodeBERTa 深度语义特征 (共 {len(codes)} 条, 批次大小 {batch_size})...")

        with torch.no_grad():
            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i:i + batch_size]
                batch_codes = [c if c and c.strip() else "# empty" for c in batch_codes]

                inputs = self.tokenizer(
                    batch_codes, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(self.device)

                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_features.append(cls_embeddings.cpu().numpy())

                if (i + batch_size) % 100 < batch_size:
                    logger.info(f"  CodeBERTa 进度: {min(i + batch_size, len(codes))} / {len(codes)}")

        return np.vstack(all_features)


# =================================================================================
# 批量提取器包装 (对接 main.py)
# =================================================================================
class PythonBatchExtractorWrapper:
    def __init__(self, config):
        self.config = config
        self.feature_names = []
        self._is_names_initialized = False
        self._codeberta_names_added = False

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        logger.info(f"开始提取 {len(codes)} 个 Python 文件的全量多维特征...")

        features_list = []

        # 在循环外先获取 FEATURES 配置字典
        features_config = getattr(self.config, 'FEATURES', {})

        # 1. 单线程串行提取传统特征
        for idx, code in enumerate(codes):
            if idx % 500 == 0 and idx > 0:
                logger.info(f"  传统特征进度: {idx} / {len(codes)}")

            feat_dict = {}

            # ==========================================
            # 🚨 修复：只传入 code 参数，并修改为 Python 专属提取器
            # ==========================================
            if features_config.get('lexical', True):
                lexical = LexicalFeatureExtractor(code)  # 👈 只留 code
                feat_dict.update(lexical.extract())

            if features_config.get('syntactic', True):
                syntactic = SyntacticFeatureExtractor(code)  # 👈 只留 code
                feat_dict.update(syntactic.extract())

            if features_config.get('python_specific', True):  # 👈 改为 python_specific
                python_specific = PythonSpecificFeatureExtractor(code)  # 👈 调用 Python 类
                feat_dict.update(python_specific.extract())

            if features_config.get('human_error', True):
                human_error = HumanErrorFeatureExtractor(code)  # 👈 只留 code
                feat_dict.update(human_error.extract())
            # ==========================================

            if not self._is_names_initialized:
                self.feature_names = sorted(list(feat_dict.keys()))
                self._is_names_initialized = True

            row = []
            for key in self.feature_names:
                v = feat_dict.get(key, 0.0)
                if math.isnan(v) or math.isinf(v):
                    v = 0.0
                row.append(v)
            features_list.append(row)

        X_manual = np.array(features_list)

        # 2. 批量提取 CodeBERTa 特征并进行矩阵拼接
        if features_config.get('codeberta', False):
            # 注意：初始化特征提取器时，确保 CodeBERTaFeatureExtractor 内部也能正确访问 config
            codeberta = CodeBERTaFeatureExtractor(self.config)
            
            # 提取深度特征
            X_codeberta = codeberta.extract_batch(codes, batch_size=16)

            if not self._codeberta_names_added:
                self.feature_names.extend(codeberta.get_feature_names())
                self._codeberta_names_added = True

            # 水平拼接：手工特征 + 深度特征
            X_final = np.hstack([X_manual, X_codeberta])
            return X_final

        return X_manual


# # =================================================================================
# # 批量提取器包装 (对接 main.py)
# # =================================================================================
# class PythonBatchExtractorWrapper:
#     def __init__(self, config):
#         self.feature_names = []
#         self._is_names_initialized = False
#
#     def get_feature_names(self) -> List[str]:
#         return self.feature_names
#
#     def extract_batch(self, codes: List[str]) -> np.ndarray:
#         logger.info(f"开始提取 {len(codes)} 个 Python 文件的全量多维特征 (四大框架并行)...")
#
#         features_list = []
#         for idx, code in enumerate(codes):
#             if idx % 500 == 0 and idx > 0:
#                 logger.info(f"  已提取: {idx} / {len(codes)}")
#
#             # 实例化四大框架
#             lexical_extractor = LexicalFeatureExtractor(code)
#             syntactic_extractor = SyntacticFeatureExtractor(code)
#             python_specific_extractor = PythonSpecificFeatureExtractor(code)
#             human_error_extractor = HumanErrorFeatureExtractor(code)
#
#             # 获取合并后的字典
#             feat_dict = {}
#             feat_dict.update(lexical_extractor.extract())
#             feat_dict.update(syntactic_extractor.extract())
#             feat_dict.update(python_specific_extractor.extract())
#             feat_dict.update(human_error_extractor.extract())
#
#             if not self._is_names_initialized:
#                 # 固定特征名顺序，保证后续生成的 Numpy 数组顺序不乱
#                 self.feature_names = sorted(list(feat_dict.keys()))
#                 self._is_names_initialized = True
#
#             # 清洗个别异常值
#             row = []
#             for key in self.feature_names:
#                 v = feat_dict.get(key, 0.0)
#                 if math.isnan(v) or math.isinf(v):
#                     v = 0.0
#                 row.append(v)
#
#             features_list.append(row)
#
#         return np.array(features_list)
