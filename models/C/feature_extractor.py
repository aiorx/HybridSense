#!/usr/bin/env python3
"""
特征提取模块 - C语言高阶融合版 (AUC 90+ 冲刺版)
专注挖掘 C 语言中的指针习惯、内存管理严谨度、宏定义偏好与人类排版失误/不一致性
"""

import re
import math
import logging
import numpy as np
from typing import List, Dict, Any
from collections import Counter

# ====== 新增：CodeBERTa 依赖导入与设备检测 ======
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger = logging.getLogger(__name__)
    logger.warning("未安装 torch 或 transformers！CodeBERTa 特征提取将退回全零矩阵。请执行: pip install torch transformers")
# ===============================================

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. 词法特征提取器 (C语言专属 + 信息论 + 长度方差)
# ==============================================================================
class LexicalFeatureExtractor:
    def __init__(self, config):
        self.config = config
        # 纯 C 语言关键字 (C89/C99/C11)
        self.keywords = {
            'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
            'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
            'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof',
            'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void',
            'volatile', 'while', 'inline', 'restrict', '_Bool', '_Complex', '_Static_assert'
        }
        self.operators = {'+', '-', '*', '/', '%', '=', '==', '!=', '>', '<', '>=', '<=',
                          '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--',
                          '+=', '-=', '*=', '/=', '->', '.', '?'}

    def get_feature_names(self) -> List[str]:
        return [
            'log_keywords_per_char', 'log_words_per_char', 'log_comments_per_char',
            'avg_line_length', 'std_line_length', 'operator_density',
            'string_literal_density', 'camel_case_ratio', 'snake_case_ratio',
            'shannon_entropy_char', 'shannon_entropy_word', 'avg_id_length',
            'std_id_length', 'blank_line_ratio', 'inline_comment_ratio',
            'halstead_volume', 'halstead_difficulty'
        ]

    def extract(self, code: str) -> np.ndarray:
        length_chars = len(code) + 1e-10
        if length_chars <= 1e-10:
            return np.zeros(len(self.get_feature_names()))

        keyword_count = sum(len(re.findall(r'\b' + kw + r'\b', code)) for kw in self.keywords)
        words = re.findall(r'\b\w+\b', code)
        comment_lines = len(re.findall(r'//.*|/\*.*?\*/', code, re.MULTILINE | re.DOTALL))

        # 行级别统计
        all_lines = code.split('\n')
        total_lines = len(all_lines) + 1e-10
        lines = [line for line in all_lines if line.strip()]
        line_lengths = [len(line) for line in lines]

        blank_lines = sum(1 for line in all_lines if not line.strip())
        inline_comments = len(re.findall(r';\s*//', code))

        camel_cases = len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z0-9]*\b', code))
        snake_cases = len(re.findall(r'\b[a-z]+_[a-z0-9_]+\b', code))
        total_cases = camel_cases + snake_cases + 1e-10

        # 信息论特征
        char_counts = Counter(code)
        entropy_char = -sum((count / length_chars) * math.log2(count / length_chars) for count in char_counts.values())
        word_counts = Counter(words)
        total_words = len(words) + 1e-10
        entropy_word = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())

        # 标识符长度统计 (加入方差，极其重要的区分点)
        identifiers = [w for w in words if not re.match(r'^\d+$', w)]
        id_lengths = [len(w) for w in identifiers]
        avg_id_len = np.mean(id_lengths) if id_lengths else 0
        std_id_len = np.std(id_lengths) if id_lengths else 0

        # Halstead 估算
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
        ops_in_code = [t for t in tokens if t in self.operators]
        operands_in_code = [t for t in tokens if re.match(r'\b\w+\b', t) and t not in self.operators]
        n1, n2 = len(set(ops_in_code)), len(set(operands_in_code))
        N1, N2 = len(ops_in_code), len(operands_in_code)
        n, N = n1 + n2 + 1e-10, N1 + N2 + 1e-10
        volume = N * math.log2(n) if n >= 1 else 0
        difficulty = (n1 / 2) * (N2 / (n2 + 1e-10))

        features = [
            np.log(keyword_count / length_chars + 1e-10),
            np.log(len(words) / length_chars + 1e-10),
            np.log(comment_lines / length_chars + 1e-10),
            np.mean(line_lengths) if line_lengths else 0,
            np.std(line_lengths) if line_lengths else 0,
            sum(code.count(op) for op in self.operators) / length_chars,
            len(re.findall(r'"(?:\\.|[^"\\])*"', code)) / length_chars,
            camel_cases / total_cases,
            snake_cases / total_cases,
            entropy_char, entropy_word,
            avg_id_len, std_id_len,
            blank_lines / total_lines,
            inline_comments / total_lines,
            volume, difficulty
        ]
        return np.array(features)

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        return np.array([self.extract(code) for code in codes])


# ==============================================================================
# 2. 语法特征提取器 (Tree-sitter C + N-gram)
# ==============================================================================
class SyntacticFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.use_tree_sitter = False
        self.parser = None
        self.ngram_feature_names = {}
        self._load_tree_sitter()

    def _load_tree_sitter(self):
        try:
            # 放弃过时的 tree_sitter_languages，改用官方原生包
            import tree_sitter_c
            from tree_sitter import Language, Parser
            
            # 适配最新版官方 tree-sitter (0.22+ 语法)
            c_language = Language(tree_sitter_c.language())
            self.parser = Parser(c_language)
            
            self.use_tree_sitter = True
            logger.info("✓ 成功加载 AST C语言解析器 (基于官方 tree-sitter-c)")
        except ImportError:
            logger.warning("❌ 未安装官方 tree-sitter 或 tree-sitter-c。退回零矩阵模式。建议: pip install tree-sitter tree-sitter-c")
            self.use_tree_sitter = False
        except Exception as e:
            logger.warning(f"❌ AST 失败: {e}。退回零矩阵模式。")
            self.use_tree_sitter = False

    def get_structural_feature_names(self) -> List[str]:
        return ['ast_max_depth', 'ast_node_density', 'ast_branching_ratio', 'ast_loop_ratio',
                'ast_cyclomatic_complexity', 'ast_pointer_decl_ratio',
                'ast_early_return_ratio', 'ast_assignment_in_cond']

    def extract_structural_features(self, codes: List[str]) -> np.ndarray:
        features = []
        for code in codes:
            if not self.use_tree_sitter or not code.strip():
                features.append(np.zeros(len(self.get_structural_feature_names())))
                continue
            try:
                tree = self.parser.parse(bytes(code, 'utf-8'))
                stats = {'total': 0, 'max_depth': 0, 'branch': 0, 'loop': 0, 'pointer': 0, 'returns': 0,
                         'assign_in_cond': 0}
                self._traverse_ast_structural(tree.root_node, 0, stats)

                length = len(code) + 1e-10
                total = stats['total'] + 1e-10
                branch = stats['branch'] + 1e-10
                cyc_comp = stats['branch'] + stats['loop'] + 1
                features.append([
                    stats['max_depth'], stats['total'] / length, stats['branch'] / total,
                                        stats['loop'] / total, cyc_comp, stats['pointer'] / total,
                                        stats['returns'] / total, stats['assign_in_cond'] / branch
                ])
            except Exception:
                features.append(np.zeros(len(self.get_structural_feature_names())))
        return np.array(features)

    def _traverse_ast_structural(self, node, depth, stats):
        stats['total'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)

        node_type = node.type
        if node_type in ('if_statement', 'switch_statement', 'case_statement'):
            stats['branch'] += 1
        elif node_type in ('for_statement', 'while_statement', 'do_statement'):
            stats['loop'] += 1
        elif node_type == 'pointer_declarator':
            stats['pointer'] += 1
        elif node_type == 'return_statement':
            stats['returns'] += 1

        # 捕捉在条件判断中直接赋值的 C hacker 习惯, 比如 if (a = 1)
        if node_type in ('if_statement', 'while_statement'):
            for child in node.children:
                if child.type == 'parenthesized_expression':
                    cond_text = child.text.decode('utf-8', errors='ignore')
                    # 匹配不含 !=, ==, <=, >= 但包含 = 的赋值行为
                    if re.search(r'[^!=<>]=\s*[^=]', cond_text):
                        stats['assign_in_cond'] += 1

        for child in node.children:
            self._traverse_ast_structural(child, depth + 1, stats)

    # N-gram 序列特征部分保持原样
    def extract_ngram_features(self, codes: List[str], n: int) -> np.ndarray:
        if self.use_tree_sitter and self.parser:
            return self._extract_c_ast_ngrams(codes, n)
        return np.zeros((len(codes), 1))

    def _extract_c_ast_ngrams(self, codes: List[str], n: int) -> np.ndarray:
        all_ngrams = []
        for code in codes:
            if not code.strip():
                all_ngrams.append({})
                continue
            try:
                tree = self.parser.parse(bytes(code, 'utf-8'))
                seq = self._get_ast_sequence(tree.root_node)
                ngrams = {tuple(seq[i:i + n]): 1 for i in range(len(seq) - n + 1)} if len(seq) >= n else {}
                total = sum(ngrams.values())
                all_ngrams.append({k: v / total for k, v in ngrams.items()} if total > 0 else {})
            except Exception:
                all_ngrams.append({})

        vocab = sorted(list(set(k for n_dict in all_ngrams for k in n_dict.keys())))
        self.ngram_feature_names[n] = [f"ast_n{n}_{'_'.join(map(str, g))}" for g in vocab]

        X = np.zeros((len(codes), len(vocab)))
        for i, n_dict in enumerate(all_ngrams):
            for j, v in enumerate(vocab):
                X[i, j] = n_dict.get(v, 0)
        return X

    def _get_ast_sequence(self, node) -> List[str]:
        seq = [node.type] if len(node.type) > 2 else []
        for child in node.children:
            seq.extend(self._get_ast_sequence(child))
        return seq

    def get_feature_names(self, n: int) -> List[str]:
        return self.ngram_feature_names.get(n, [])


# ==============================================================================
# 3. C 语言专属习惯与安全特征 (深入底层与极客习惯)
# ==============================================================================
class CSpecificFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.memory_funcs = set(
            getattr(config, 'C_CONFIG', {}).get('memory_funcs', ['malloc', 'calloc', 'realloc', 'free']))
        self.unsafe_funcs = set(
            getattr(config, 'C_CONFIG', {}).get('unsafe_funcs', ['gets', 'strcpy', 'strcat', 'sprintf']))
        self.safe_funcs = set(
            getattr(config, 'C_CONFIG', {}).get('safe_funcs', ['fgets', 'strncpy', 'strncat', 'snprintf']))

    def get_feature_names(self) -> List[str]:
        return [
            'struct_density', 'typedef_struct_ratio', 'macro_define_density',
            'pointer_op_density', 'memory_func_density', 'malloc_check_ratio',
            'unsafe_func_ratio', 'vla_usage_flag', 'void_main_flag', 'global_large_array_flag',
            'bitwise_op_density', 'c_style_cast_density', 'sizeof_idiom_ratio',
            'pointer_arithmetic_density', 'ternary_op_density',
            'yoda_condition_ratio', 'vertical_decl_ratio'
        ]

    def extract(self, code: str) -> np.ndarray:
        if not code:
            return np.zeros(len(self.get_feature_names()))

        features = []
        length_chars = len(code) + 1e-10

        # ... [保留你原有的 1-15 个特征特征提取代码，这里省略不重复写] ...
        # (确保从 structs 到 ternary_ops 的提取代码都在这里)
        structs = len(re.findall(r'\bstruct\s+\w+', code))
        typedef_structs = len(re.findall(r'\btypedef\s+struct\b', code))
        features.append(structs / length_chars)
        features.append(typedef_structs / (structs + 1e-10))

        features.append(len(re.findall(r'#define\b', code)) / length_chars)
        features.append(len(re.findall(r'(->|\*(?!\s*=)|\&(?!\&))', code)) / length_chars)

        mem_count = sum(len(re.findall(r'\b' + f + r'\b', code)) for f in self.memory_funcs)
        features.append(mem_count / length_chars)

        mallocs = len(re.findall(r'\bmalloc\s*\(', code))
        null_checks = len(re.findall(r'==\s*NULL|NULL\s*==|!([a-zA-Z_]\w*)', code))
        features.append(min(1.0, null_checks / (mallocs + 1e-10)))

        unsafe_cnt = sum(len(re.findall(r'\b' + f + r'\b', code)) for f in self.unsafe_funcs)
        safe_cnt = sum(len(re.findall(r'\b' + f + r'\b', code)) for f in self.safe_funcs)
        features.append(unsafe_cnt / (unsafe_cnt + safe_cnt + 1e-10))

        features.append(1.0 if re.search(r'\b(int|long|char|float|double)\s+[a-zA-Z_]\w*\s*\[\s*[a-zA-Z_]\w*\s*\]', code) else 0.0)
        features.append(1.0 if re.search(r'\bvoid\s+main\s*\(', code) else 0.0)
        features.append(1.0 if re.search(r'^\s*(int|long|long long|double|char)\s+[a-zA-Z_]\w*\s*\[\s*\d{4,}\s*\]', code, re.MULTILINE) else 0.0)

        bitwise_ops = len(re.findall(r'(<<|>>|\&(?!\&)|\|(?!\|)|\^)', code))
        features.append(bitwise_ops / length_chars)

        c_casts = len(re.findall(r'\(\s*(int|char|float|double|long|short|void|struct\s+\w+)\s*\**\s*\)', code))
        features.append(c_casts / length_chars)

        sizeofs = len(re.findall(r'\bsizeof\b', code))
        features.append(sizeofs / length_chars)

        ptr_arithmetic = len(re.findall(r'\*\s*[a-zA-Z_]\w*\s*(\+\+|--)', code)) + len(re.findall(r'\*\s*\(\s*[a-zA-Z_]\w*\s*[\+\-]', code))
        features.append(ptr_arithmetic / length_chars)

        ternary_ops = code.count('?')
        features.append(ternary_ops / length_chars)

        # =========================================================
        # 【新增 16】Yoda 条件 (尤达表达式): 匹配 NULL == t 或 0 == x
        # 这是典型的人类老手防御性编程习惯
        # =========================================================
        yoda_conditions = len(re.findall(r'\b(NULL|nullptr|-?\d+|true|false)\s*(==|!=)\s*[a-zA-Z_]\w*\b', code))
        total_equals = len(re.findall(r'==|!=', code)) + 1e-10
        features.append(yoda_conditions / total_equals)

        # =========================================================
        # 【新增 17】Windows/系统级极端垂直声明: 返回值、宏、函数名各自占一行
        # 例如 HANDLE \n WINAPI \n CreateMutexExA (
        # =========================================================
        vertical_decls = len(re.findall(r'^[A-Za-z_]\w*\*?\s*\n\s*(?:[A-Z_]+\s*\n\s*)?[A-Za-z_]\w*\s*\(', code, re.MULTILINE))
        features.append(vertical_decls / (len(code.split('\n')) + 1e-10))

        return np.array(features)

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        return np.array([self.extract(code) for code in codes])


# ==============================================================================
# 4. 微观习惯与人类失误 (专抓大模型不可能有的一致性破绽)
# ==============================================================================
class HumanErrorFeatureExtractor:
    def __init__(self, config=None):
        self.common_typos = {'lenght', 'heigth', 'recieve', 'occured', 'sucess', 'retun', 'funtion', 'ture', 'flase',
                             'paramter', 'teh'}
        # 补充了一些强情绪词，用来抓取类似于“不知道是故意还是有意”这种长句吐槽背后的情绪发泄
        self.informal_tags = {'todo', 'fixme', 'hack', 'wtf', 'shit', 'fuck', 'lol', 'test', 'bug', 'weird', 'stupid'}
        self.human_vars = {'tmp', 'temp', 'foo', 'bar', 'res', 'ans', 'val', 'arr', 'idx', 'cnt', 'ptr'}
        self.human_atypical = {'aids', 'composed', 'development', 'materials', 'segments', 'utilities', 'content', 'samples' , 'blocks', 'via', 'third-party', 'borrowed'}

    def get_feature_names(self) -> List[str]:
        return [
            'typo_density', 'informal_tag_density', 'human_var_density','human_atypical_density',
            'commented_code_ratio', 'trailing_whitespace_ratio', 'mixed_indentation_ratio',
            'magic_number_density', 'missing_space_after_kw', 'crowded_operator_density',
            'pre_increment_ratio', 'multiple_declaration_ratio',
            'brace_style_inconsistency', 'comma_missing_space_ratio', 'redundant_parens_ratio',
            'consecutive_empty_lines_ratio',
            'consecutive_identical_call_ratio',
        ]

    def extract(self, code: str) -> np.ndarray:
        if not code: return np.zeros(len(self.get_feature_names()))

        code_lower = code.lower()
        lines = code.split('\n')
        total_lines = len(lines) if len(lines) > 0 else 1
        length_chars = len(code) + 1e-10
        features = []

        # ... [保留原有的提取逻辑 1-15，此处省略以节省空间，确保完全保留] ...
        features.append(
            sum(len(re.findall(r'\b' + typo + r'\b', code_lower)) for typo in self.common_typos) / length_chars)
        features.append(
            sum(len(re.findall(r'\b' + tag + r'\b', code_lower)) for tag in self.informal_tags) / length_chars)
        features.append(sum(len(re.findall(r'\b' + var + r'\b', code_lower)) for var in self.human_vars) / length_chars)
        features.append(sum(len(re.findall(r'\b' + atypical + r'\b', code_lower)) for atypical in self.human_atypical) / length_chars)
        features.append(sum(1 for line in lines if re.match(r'^\s*//.*[;{]\s*$', line)) / total_lines)
        features.append(sum(1 for line in lines if re.search(r'[ \t]+$', line)) / total_lines)
        tab_indents = sum(1 for line in lines if line.startswith('\t'))
        space_indents = sum(1 for line in lines if line.startswith(' ') and not line.startswith(' *'))
        features.append(min(tab_indents, space_indents) / (tab_indents + space_indents + 1e-10))
        magic_numbers = [n for n in re.findall(r'\b\d+\b', code) if n not in ('0', '1')]
        features.append(len(magic_numbers) / length_chars)
        missing_space = len(re.findall(r'\b(if|for|while|switch)\(', code))
        has_space = len(re.findall(r'\b(if|for|while|switch)\s+\(', code))
        features.append(missing_space / (missing_space + has_space + 1e-10))
        crowded_ops = len(re.findall(r'[a-zA-Z0-9_](==|!=|<=|>=|\+=|-=|\*=|/=|=(?!=)|<|>)[a-zA-Z0-9_]', code))
        features.append(crowded_ops / length_chars)
        pre_inc = len(re.findall(r'\+\+\s*[a-zA-Z_]\w*', code))
        post_inc = len(re.findall(r'[a-zA-Z_]\w*\s*\+\+', code))
        features.append(pre_inc / (pre_inc + post_inc + 1e-10))
        multiple_decls = len(re.findall(r'\b(int|long|float|double|char)\s+\*?[a-zA-Z_]\w*\s*,\s*\*?[a-zA-Z_]', code))
        total_decls = len(re.findall(r'\b(int|long|float|double|char)\s+\*?[a-zA-Z_]\w*', code)) + 1e-10
        features.append(multiple_decls / total_decls)
        same_line_brace = len(re.findall(r'\)\s*\{', code))
        next_line_brace = len(re.findall(r'\)\s*\n\s*\{', code))
        total_braces = same_line_brace + next_line_brace + 1e-10
        minority_brace_ratio = min(same_line_brace, next_line_brace) / total_braces
        features.append(minority_brace_ratio)
        missing_comma_space = len(re.findall(r',[^\s]', code))
        has_comma_space = len(re.findall(r',\s', code))
        features.append(missing_comma_space / (missing_comma_space + has_comma_space + 1e-10))
        redundant_parens = len(re.findall(r'\(\s*\([^\(\)]+\)\s*\)', code))
        features.append(redundant_parens / length_chars)
        multiple_empty_lines = len(re.findall(r'\n[ \t]*\n[ \t]*\n', code))
        features.append(multiple_empty_lines / total_lines)

        # =========================================================
        # 【新增 16】连续相同函数调用比例 (人类暴力复制粘贴硬编码)
        # 抓取连续 3 行以上以同一个函数名开头的情况（如连续的 mvprintw）
        # =========================================================
        stripped_lines = [line.strip() for line in lines if line.strip()]
        consecutive_identical_calls = 0
        for i in range(len(stripped_lines) - 2):
            m1 = re.match(r'^([a-zA-Z_]\w*)\s*\(', stripped_lines[i])
            m2 = re.match(r'^([a-zA-Z_]\w*)\s*\(', stripped_lines[i + 1])
            m3 = re.match(r'^([a-zA-Z_]\w*)\s*\(', stripped_lines[i + 2])
            if m1 and m2 and m3 and (m1.group(1) == m2.group(1) == m3.group(1)):
                consecutive_identical_calls += 1

        features.append(consecutive_identical_calls / (len(stripped_lines) + 1e-10))


        return np.array(features)

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        return np.array([self.extract(code) for code in codes])


# ==============================================================================
# 5. 深度语义特征提取器 (CodeBERTa 骨干网络)
# ==============================================================================
class CodeBERTaFeatureExtractor:
    def __init__(self, config):
        self.config = config
        # 从配置中读取模型名，默认使用小参数量、表现依然强悍的官方模型
        self.model_name = getattr(config, 'CODEBERTA_MODEL', "huggingface/CodeBERTa-small-v1")
        self.hidden_size = 768  # CodeBERTa-small 的输出维度为 768

        if HAS_TRANSFORMERS:
            # 自动检测可用硬件加速
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            logger.info(f"加载 CodeBERTa 模型 ({self.model_name}) 到设备: {self.device}...")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()  # 切换到评估模式，防止 Dropout 干扰特征提取
                self.is_ready = True
                logger.info("✓ 成功加载 CodeBERTa 骨干网络")
            except Exception as e:
                logger.error(f"❌ 加载 CodeBERTa 失败: {e}")
                self.is_ready = False
        else:
            self.is_ready = False

    def get_feature_names(self) -> List[str]:
        # 生成 768 维的特征列名
        return [f"codeberta_dim_{i}" for i in range(self.hidden_size)]

    def extract_batch(self, codes: List[str], batch_size: int = 16) -> np.ndarray:
        """
        批量提取深层语义特征，加入 batch_size 防止内存爆炸
        """
        if not self.is_ready or not codes:
            logger.warning("CodeBERTa 未就绪或无数据，返回全 0 特征矩阵。")
            return np.zeros((len(codes), self.hidden_size))

        all_features = []
        logger.info(f"开始提取 CodeBERTa 深层特征 (共 {len(codes)} 条, 批次大小 {batch_size})...")

        # 关闭梯度计算，大幅减少显存/内存占用
        with torch.no_grad():
            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i:i + batch_size]

                # 防御性编程：如果有完全为空的代码片段，替换为注释，防止 tokenizer 报错
                batch_codes = [c if c and c.strip() else "// empty file" for c in batch_codes]

                # 统一将代码截断或填充到 512 的长度（BERT架构的标准输入）
                inputs = self.tokenizer(
                    batch_codes,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # 将数据送入模型
                outputs = self.model(**inputs)

                # 提取深层特征：取 [CLS] token 在最后一层的隐藏状态作为全局表示
                # outputs.last_hidden_state 的形状是 (batch_size, sequence_length, hidden_size)
                # 我们取第 0 个 token (即 [CLS])
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

                # 将特征从 GPU/计算设备 转移回 CPU 并转换为 numpy 数组
                all_features.append(cls_embeddings.cpu().numpy())

                # 打印进度
                if (i + batch_size) % 100 < batch_size:
                    logger.info(f"  CodeBERTa 处理进度: {min(i + batch_size, len(codes))} / {len(codes)}")

        return np.vstack(all_features)


# ==============================================================================
# 工厂类
# ==============================================================================
class FeatureExtractorFactory:
    @staticmethod
    def create_extractors(config):
        extractors = {
            'lexical': LexicalFeatureExtractor(config),
            'syntactic': SyntacticFeatureExtractor(config),
            'c_specific': CSpecificFeatureExtractor(config),
            'human_error': HumanErrorFeatureExtractor(config),
            'codeberta': CodeBERTaFeatureExtractor(config),
        }
        return extractors
