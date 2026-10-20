#!/usr/bin/env python3
"""
特征提取模块 - C++专用版 (高阶融合版)
仅包含四大原生板块，但深度融合了信息论、圈复杂度、竞编习惯与微观排版特征
"""

import re
import math
import logging
import numpy as np
from typing import List, Dict, Any
from collections import Counter

# ====== 【新增】CodeBERTa 依赖导入与设备检测 ======
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
# 1. 词法特征提取器 (融合：信息论香农熵、Halstead复杂度)
# ==============================================================================
class LexicalFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.keywords = {
            'do', 'else', 'if', 'switch', 'for', 'while', 'return', 'break', 'continue',
            'struct', 'typedef', 'int', 'char', 'float', 'double', 'void', 'static', 'const',
            'class', 'public', 'private', 'protected', 'virtual', 'override', 'final',
            'namespace', 'using', 'template', 'typename', 'new', 'delete', 'this',
            'try', 'catch', 'throw', 'const_cast', 'dynamic_cast', 'reinterpret_cast', 'static_cast',
            'explicit', 'inline', 'operator', 'friend', 'sizeof', 'typeid', 'nullptr',
            'true', 'false', 'bool', 'wchar_t', 'mutable', 'static_assert', 'alignas', 'alignof',
            'constexpr', 'decltype', 'noexcept', 'thread_local', 'export', 'extern'
        }
        self.operators = {'+', '-', '*', '/', '%', '=', '==', '!=', '>', '<', '>=', '<=',
                          '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--',
                          '+=', '-=', '*=', '/=', '->', '.', '::', '?', ':'}
        self.stl_containers = set(getattr(config, 'CPP_CONFIG', {}).get('stl_containers', []))
        self.stl_algorithms = set(getattr(config, 'CPP_CONFIG', {}).get('stl_algorithms', []))
        self.smart_pointers = set(getattr(config, 'CPP_CONFIG', {}).get('smart_pointers', []))

    def get_feature_names(self) -> List[str]:
        return [
            'log_keywords_per_char', 'log_words_per_char', 'log_comments_per_char',
            'avg_line_length', 'std_line_length', 'cpp_keyword_ratio',
            'operator_density', 'stl_container_usage', 'stl_algorithm_usage', 'smart_pointer_usage',
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
        lines = [line for line in code.split('\n') if line.strip()]
        line_lengths = [len(line) for line in lines]

        cpp_keywords = self.keywords - {'do', 'else', 'if', 'switch', 'for', 'while', 'return', 'break', 'continue',
                                        'struct', 'typedef', 'int', 'char', 'float', 'double', 'void', 'static',
                                        'const'}
        cpp_keyword_count = sum(len(re.findall(r'\b' + kw + r'\b', code)) for kw in cpp_keywords)

        camel_cases = len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z0-9]*\b', code))
        snake_cases = len(re.findall(r'\b[a-z]+_[a-z0-9_]+\b', code))
        total_cases = camel_cases + snake_cases + 1e-10

        features = [
            np.log(keyword_count / length_chars + 1e-10),
            np.log(len(words) / length_chars + 1e-10),
            np.log(comment_lines / length_chars + 1e-10),
            np.mean(line_lengths) if line_lengths else 0,
            np.std(line_lengths) if line_lengths else 0,
            cpp_keyword_count / (keyword_count + 1e-10),
            sum(code.count(op) for op in self.operators) / length_chars,
            sum(len(re.findall(r'\b' + c + r'\b', code)) for c in self.stl_containers) / length_chars,
            sum(len(re.findall(r'\b' + a + r'\b', code)) for a in self.stl_algorithms) / length_chars,
            sum(len(re.findall(r'\b' + p + r'\b', code)) for p in self.smart_pointers) / length_chars,
            len(re.findall(r'"(?:\\.|[^"\\])*"', code)) / length_chars,
            camel_cases / total_cases,
            snake_cases / total_cases
        ]

        # === 融合信息论与结构变异特征 ===
        char_counts = Counter(code)
        entropy_char = -sum((count / length_chars) * math.log2(count / length_chars) for count in char_counts.values())

        word_counts = Counter(words)
        total_words = len(words) + 1e-10
        entropy_word = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())

        identifiers = [w for w in words if not re.match(r'^\d+$', w)]
        avg_id_len = sum(len(w) for w in identifiers) / (len(identifiers) + 1e-10)
        id_lengths = [len(w) for w in identifiers]
        std_id_len = np.std(id_lengths) if id_lengths else 0

        all_lines = code.split('\n')
        total_lines = len(all_lines) + 1e-10
        blank_lines = sum(1 for line in all_lines if not line.strip())
        inline_comments = len(re.findall(r';\s*//', code))

        blank_line_ratio = blank_lines / total_lines
        inline_comment_ratio = inline_comments / total_lines

        # Halstead 估算
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
        ops_in_code = [t for t in tokens if t in self.operators]
        operands_in_code = [t for t in tokens if re.match(r'\b\w+\b', t) and t not in self.operators]
        n1, n2 = len(set(ops_in_code)), len(set(operands_in_code))
        N1, N2 = len(ops_in_code), len(operands_in_code)
        n, N = n1 + n2 + 1e-10, N1 + N2 + 1e-10
        volume = N * math.log2(n) if n >= 1 else 0
        difficulty = (n1 / 2) * (N2 / (n2 + 1e-10))

        features.extend(
            [entropy_char, entropy_word, avg_id_len, std_id_len, blank_line_ratio, inline_comment_ratio, volume,
             difficulty])
        return np.array(features)

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        return np.array([self.extract(code) for code in codes])


# ==============================================================================
# 2. 语法特征提取器 (融合：AST McCabe圈复杂度)
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
            import tree_sitter_cpp
            from tree_sitter import Language, Parser
            
            # 适配最新版官方 tree-sitter (0.22+ 语法)
            cpp_language = Language(tree_sitter_cpp.language())
            self.parser = Parser(cpp_language)
            
            self.use_tree_sitter = True
            logger.info("✓ 成功加载 AST C++语言解析器 (基于官方 tree-sitter-cpp)")
        except ImportError:
            logger.warning("❌ 未安装官方 tree-sitter 或 tree-sitter-cpp。退回零矩阵模式。建议: pip install tree-sitter tree-sitter-cpp")
            self.use_tree_sitter = False
        except Exception as e:
            logger.warning(f"❌ AST 失败: {e}。退回零矩阵模式。")
            self.use_tree_sitter = False

    def get_structural_feature_names(self) -> List[str]:
        return [
            'ast_max_depth', 'ast_node_density', 'ast_branching_ratio',
            'ast_loop_ratio', 'ast_cyclomatic_complexity'
        ]

    def extract_structural_features(self, codes: List[str]) -> np.ndarray:
        features = []
        for code in codes:
            if not self.use_tree_sitter or not code.strip():
                features.append(np.zeros(len(self.get_structural_feature_names())))
                continue
            try:
                tree = self.parser.parse(bytes(code, 'utf-8'))
                stats = {'total': 0, 'max_depth': 0, 'branch': 0, 'loop': 0}
                self._traverse_ast_structural(tree.root_node, 0, stats)

                length = len(code) + 1e-10
                total = stats['total'] + 1e-10

                cyc_comp = stats['branch'] + stats['loop'] + 1

                features.append([
                    stats['max_depth'],
                    stats['total'] / length,
                    stats['branch'] / total,
                    stats['loop'] / total,
                    cyc_comp
                ])
            except Exception:
                features.append(np.zeros(len(self.get_structural_feature_names())))
        return np.array(features)

    def _traverse_ast_structural(self, node, depth, stats):
        stats['total'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)

        node_type = node.type
        if node_type in ('if_statement', 'switch_statement', 'case_statement', 'catch_clause'):
            stats['branch'] += 1
        elif node_type in ('for_statement', 'while_statement', 'do_statement', 'range_for_statement'):
            stats['loop'] += 1

        for child in node.children:
            self._traverse_ast_structural(child, depth + 1, stats)

    def extract_ngram_features(self, codes: List[str], n: int) -> np.ndarray:
        if self.use_tree_sitter and self.parser:
            return self._extract_ast_ngrams(codes, n)
        # 【核心修复】如果AST加载失败，必须返回 0 列，否则会导致 SelectKBest 维度崩溃！
        return np.zeros((len(codes), 0))

    def _extract_ast_ngrams(self, codes: List[str], n: int) -> np.ndarray:
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
# 3. C++ 特有特征提取器
# ==============================================================================
class CppSpecificFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.stl_containers = set(getattr(config, 'CPP_CONFIG', {}).get('stl_containers', []))
        self.stl_algorithms = set(getattr(config, 'CPP_CONFIG', {}).get('stl_algorithms', []))

    def get_feature_names(self) -> List[str]:
        return [
            'class_count', 'template_depth_avg', 'exception_handler_ratio',
            'smart_pointer_density', 'modern_cpp_keyword_density',
            'operator_overload_count', 'stl_container_usage', 'macro_define_count',
            'has_bits_stdcxx', 'fast_io_magic', 'cp_macro_shortcuts',
            'c_style_cast_density', 'cpp_style_cast_density', 'std_prefix_density',
            'endl_vs_newline_ratio', 'has_signed_main', 'const_reference_ratio',
            'yoda_condition_ratio', 'vertical_decl_ratio', 'bitwise_op_density',
            'cpp_sizeof_idiom_ratio', 'ptr_arithmetic_density', 'ternary_op_density'
        ]

    def extract(self, code: str) -> np.ndarray:
        if not code:
            return np.zeros(len(self.get_feature_names()))

        features = []
        length_chars = len(code) + 1e-10

        features.append(len(re.findall(r'\bclass\s+\w+\s*[{:]', code)))
        template_depths = [m.group(1).count('<') + 1 for m in re.finditer(r'template\s*<([^>]*)>', code)]
        features.append(np.mean(template_depths) if template_depths else 0)

        try_cnt = len(re.findall(r'\btry\s*\{', code))
        catch_cnt = len(re.findall(r'\bcatch\s*\([^)]*\)\s*\{', code))
        features.append(catch_cnt / (try_cnt + 1e-10))

        features.append(len(re.findall(r'\b(unique_ptr|shared_ptr|weak_ptr|auto_ptr)\s*<', code)) / length_chars)
        features.append(len(re.findall(r'\b(constexpr|noexcept|decltype|nullptr)\b', code)) / length_chars)
        features.append(len(re.findall(r'\boperator\s*[+\-*/%=<>!&|^~]=?', code)))
        features.append(sum(len(re.findall(r'\b' + c + r'\b', code)) for c in self.stl_containers) / length_chars)
        features.append(len(re.findall(r'#define\b', code)) / length_chars)

        features.append(1.0 if re.search(r'#include\s*<bits/stdc\+\+\.h>', code) else 0.0)
        features.append(1.0 if re.search(r'ios(?:_base)?::sync_with_stdio\s*\(\s*(?:false|0)\s*\)', code) else 0.0)
        features.append(len(re.findall(r'\b(pb|mp|all|sz|rep|FOR)\b', code)) / length_chars)
        features.append(
            len(re.findall(r'\(\s*(int|char|float|double|long|short|void)\s*\**\s*\)', code)) / length_chars)
        features.append(
            len(re.findall(r'\b(static_cast|dynamic_cast|reinterpret_cast|const_cast)\s*<', code)) / length_chars)
        features.append(len(re.findall(r'\bstd::', code)) / length_chars)

        endl_cnt = len(re.findall(r'\bendl\b', code))
        newline_cnt = code.count('\\n')
        features.append(endl_cnt / (endl_cnt + newline_cnt + 1e-10))
        features.append(1.0 if re.search(r'\bsigned\s+main\b', code) else 0.0)
        features.append(len(re.findall(r'\bconst\s+[a-zA-Z_]\w*\s*&', code)) / length_chars)

        # =========================================================
        # 【新增绝杀特征】Yoda条件、系统级垂直排版、底层运算
        # =========================================================
        yoda_conditions = len(re.findall(r'\b(NULL|nullptr|-?\d+|true|false)\s*(==|!=)\s*[a-zA-Z_]\w*\b', code))
        total_equals = len(re.findall(r'==|!=', code)) + 1e-10
        features.append(yoda_conditions / total_equals)

        vertical_decls = len(
            re.findall(r'^[A-Za-z_]\w*\*?\s*\n\s*(?:[A-Z_]+\s*\n\s*)?[A-Za-z_]\w*\s*\(', code, re.MULTILINE))
        features.append(vertical_decls / (len(code.split('\n')) + 1e-10))

        bitwise_ops = len(re.findall(r'(<<|>>|\&(?!&)|\|(?!\|)|\^)', code))
        features.append(bitwise_ops / length_chars)

        sizeofs = len(re.findall(r'\bsizeof\b', code))
        features.append(sizeofs / length_chars)

        ptr_arithmetic = len(re.findall(r'\*\s*[a-zA-Z_]\w*\s*(\+\+|--)', code)) + len(
            re.findall(r'\*\s*\(\s*[a-zA-Z_]\w*\s*[\+\-]', code))
        features.append(ptr_arithmetic / length_chars)

        ternary_ops = code.count('?')
        features.append(ternary_ops / length_chars)

        return np.array(features)

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        return np.array([self.extract(code) for code in codes])


# ==============================================================================
# 4. 微观习惯与人类失误
# ==============================================================================
class HumanErrorFeatureExtractor:
    def __init__(self, config=None):
        self.config = config
        self.common_typos = {'lenght', 'heigth', 'recieve', 'occured', 'sucess', 'retun', 'funtion', 'ture', 'flase',
                             'paramter', 'teh', 'defualt'}
        self.informal_tags = {'todo', 'fixme', 'hack', 'wtf', 'shit', 'fuck', 'lol', 'resources', 'test'}
        self.human_vars = {'tmp', 'temp', 'foo', 'bar', 'res', 'ans', 'val', 'arr', 'idx', 'cnt'}
        self.human_atypical = {'aids', 'composed', 'development', 'materials', 'segments', 'utilities', 'content', 'samples' , 'blocks', 'via', 'third-party', 'borrowed'}


    def get_feature_names(self) -> List[str]:
        return [
            'typo_density', 'informal_tag_density', 'human_var_density','human_atypical_density',
            'commented_code_ratio', 'trailing_whitespace_ratio', 'mixed_indentation_ratio',
            'magic_number_density',
            'missing_space_after_kw', 'crowded_operator_density',
            'pre_increment_ratio', 'multiple_declaration_ratio',
            'brace_style_inconsistency', 'comma_missing_space_ratio',
            'redundant_parens_ratio', 'consecutive_empty_lines_ratio',
            'consecutive_identical_call_ratio','has_ai_generation_tag'
        ]

    def extract(self, code: str) -> np.ndarray:
        if not code or len(code.strip()) == 0:
            return np.zeros(len(self.get_feature_names()))

        code_lower = code.lower()
        lines = code.split('\n')
        total_lines = len(lines) if len(lines) > 0 else 1
        length_chars = len(code) + 1e-10
        features = []

        features.append(
            sum(len(re.findall(r'\b' + typo + r'\b', code_lower)) for typo in self.common_typos) / length_chars)
        features.append(
            sum(len(re.findall(r'\b' + tag + r'\b', code_lower)) for tag in self.informal_tags) / length_chars)
        features.append(sum(len(re.findall(r'\b' + var + r'\b', code_lower)) for var in self.human_vars) / length_chars)
        features.append(sum(len(re.findall(r'\b' + atypical + r'\b', code_lower)) for atypical in self.human_atypical) / length_chars)

        features.append(sum(1 for line in lines if re.match(r'^\s*//.*[;{}]\s*$', line)) / total_lines)
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

        multiple_decls = len(re.findall(r'\b(int|long\s+long|float|double|char)\s+[a-zA-Z_]\w*\s*,\s*[a-zA-Z_]', code))
        total_decls = len(re.findall(r'\b(int|long\s+long|float|double|char)\s+[a-zA-Z_]\w*', code)) + 1e-10
        features.append(multiple_decls / total_decls)

        # =========================================================
        # 【新增绝杀特征】微观排版瑕疵、硬编码
        # =========================================================
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
# 5. 【新增】深度语义特征提取器 (CodeBERTa 骨干网络)
# ==============================================================================
class CodeBERTaFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.model_name = getattr(config, 'CODEBERTA_MODEL', "huggingface/CodeBERTa-small-v1")
        self.hidden_size = 768  # 基础版的输出维度

        if HAS_TRANSFORMERS:
            # 自动检测 GPU
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            logger.info(f"加载 CodeBERTa 模型 ({self.model_name}) 到设备: {self.device}...")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()  # 必须切到 eval 模式
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
        logger.info(f"开始提取 CodeBERTa 深层特征 (共 {len(codes)} 条, 批次大小 {batch_size})...")

        # 禁用梯度计算以节省显存
        with torch.no_grad():
            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i:i + batch_size]

                # 防止空代码片段导致 tokenizer 崩溃
                batch_codes = [c if c and c.strip() else "// empty" for c in batch_codes]

                inputs = self.tokenizer(
                    batch_codes,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512  # C++ 代码较长，直接截断前 512 个 token
                ).to(self.device)

                outputs = self.model(**inputs)

                # 提取 [CLS] 标记的输出作为整个代码片段的全局特征表示
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_features.append(cls_embeddings.cpu().numpy())

                if (i + batch_size) % 100 < batch_size:
                    logger.info(f"  CodeBERTa 处理进度: {min(i + batch_size, len(codes))} / {len(codes)}")

        return np.vstack(all_features)


class FeatureExtractorFactory:
    @staticmethod
    def create_extractors(config):
        return {
            'lexical': LexicalFeatureExtractor(config),
            'syntactic': SyntacticFeatureExtractor(config),
            'cpp_specific': CppSpecificFeatureExtractor(config),
            'human_error': HumanErrorFeatureExtractor(config),
            'codeberta': CodeBERTaFeatureExtractor(config),
        }
