#!/usr/bin/env python3
"""
JavaScript 特征提取模块 - 终极无损四框架版
完全无损保留您的 102 维全量特征！并严格划分为四大类。
彻底修复了因 try-except 导致键名污染引起的特征维度缩水问题。
"""

import re
import math
import statistics
import logging
import numpy as np
from typing import Dict, List
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

try:
    import esprima

    ESPRIMA_AVAILABLE = True
except ImportError:
    ESPRIMA_AVAILABLE = False
    logger.warning("esprima 未安装，AST 特征将置为 0。建议: pip install esprima")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("未安装 torch 或 transformers！CodeBERTa 将退回全零矩阵。")


# ==============================================================================
# 1. 词法特征提取器 (Lexical)
# 包含原版的 basic_stats, lexical_features, style_features
# ==============================================================================
class LexicalFeatureExtractor:
    def __init__(self, code: str, lines: List[str], total_lines: int):
        self.code = code
        self.lines = lines
        self.total_lines = total_lines
        self.length_chars = max(len(code), 1)

    def get_feature_names(self) -> List[str]:
        return [
            # Basic Stats (8)
            'char_density', 'whitespace_ratio', 'digit_ratio', 'alpha_ratio',
            'avg_line_length', 'line_length_std', 'empty_line_ratio', 'comment_ratio',
            # Lexical Features (15)
            'keyword_density', 'conditional_keyword_ratio', 'loop_keyword_ratio', 'declaration_keyword_ratio',
            'operator_density', 'assignment_ratio', 'comparison_ratio', 'bracket_density', 'paren_ratio',
            'curly_ratio', 'identifier_count', 'avg_identifier_length', 'identifier_length_std',
            'unique_identifiers', 'capital_letter_ratio',
            # Style Features (11)
            'camel_case_ratio', 'pascal_case_ratio', 'snake_case_ratio', 'uppercase_ratio',
            'inline_comment_ratio', 'block_comment_ratio', 'space_after_comma',
            'space_before_brace', 'single_quote_ratio', 'code_consistency', 'semicolon_ratio'
        ]

    def extract(self) -> Dict[str, float]:
        features = {k: 0.0 for k in self.get_feature_names()}
        try:
            total_chars = len(self.code)
            whitespace_chars = sum(1 for c in self.code if c.isspace())
            digit_chars = sum(1 for c in self.code if c.isdigit())
            alpha_chars = sum(1 for c in self.code if c.isalpha())
            line_lengths = [len(line.rstrip()) for line in self.lines if line.strip()]
            empty_lines = sum(1 for line in self.lines if not line.strip())
            single_line_comments = sum(1 for line in self.lines if line.strip().startswith('//'))
            multi_line_comments = len(re.findall(r'/\*.*?\*/', self.code, re.DOTALL))

            features['char_density'] = total_chars / self.total_lines
            features['whitespace_ratio'] = whitespace_chars / max(total_chars, 1)
            features['digit_ratio'] = digit_chars / max(total_chars, 1)
            features['alpha_ratio'] = alpha_chars / max(total_chars, 1)
            features['avg_line_length'] = statistics.mean(line_lengths) if line_lengths else 0
            features['line_length_std'] = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0
            features['empty_line_ratio'] = empty_lines / self.total_lines
            features['comment_ratio'] = (single_line_comments + multi_line_comments) / self.total_lines

            js_keywords = ['if', 'else', 'for', 'while', 'function', 'return', 'const', 'let', 'var']
            keyword_counts = {kw: len(re.findall(rf'\b{kw}\b', self.code)) for kw in js_keywords}
            total_keywords = sum(keyword_counts.values())
            operators = {
                'assignment': len(re.findall(r'=(?!=)', self.code)),
                'comparison': len(re.findall(r'[!=]==|[<>]=?|!?=', self.code)),
                'arithmetic': len(re.findall(r'[+\-*/%]|\+\+|\-\-', self.code)),
                'logical': len(re.findall(r'&&|\|\||!', self.code)),
                'bitwise': len(re.findall(r'[&|^~]|[<>]{2}', self.code))
            }
            brackets = {'paren': len(re.findall(r'\(', self.code)), 'curly': len(re.findall(r'\{', self.code))}
            identifiers = re.findall(r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b', self.code)
            identifier_lengths = [len(id) for id in identifiers]

            features['keyword_density'] = total_keywords / self.total_lines
            features['conditional_keyword_ratio'] = (keyword_counts['if'] + keyword_counts['else']) / max(
                total_keywords, 1)
            features['loop_keyword_ratio'] = (keyword_counts['for'] + keyword_counts['while']) / max(total_keywords, 1)
            features['declaration_keyword_ratio'] = (keyword_counts['const'] + keyword_counts['let'] + keyword_counts[
                'var']) / max(total_keywords, 1)
            features['operator_density'] = sum(operators.values()) / self.total_lines
            features['assignment_ratio'] = operators['assignment'] / max(sum(operators.values()), 1)
            features['comparison_ratio'] = operators['comparison'] / max(sum(operators.values()), 1)
            features['bracket_density'] = sum(brackets.values()) / self.total_lines
            features['paren_ratio'] = brackets['paren'] / max(sum(brackets.values()), 1)
            features['curly_ratio'] = brackets['curly'] / max(sum(brackets.values()), 1)
            features['identifier_count'] = len(identifiers) / self.total_lines
            features['avg_identifier_length'] = statistics.mean(identifier_lengths) if identifier_lengths else 0
            features['identifier_length_std'] = statistics.stdev(identifier_lengths) if len(
                identifier_lengths) > 1 else 0
            features['unique_identifiers'] = len(set(identifiers)) / max(len(identifiers), 1)
            features['capital_letter_ratio'] = sum(1 for c in self.code if c.isupper()) / max(len(self.code), 1)

            c1 = sum(1 for i in identifiers if re.match(r'^[a-z][a-zA-Z0-9]*$', i))
            c2 = sum(1 for i in identifiers if re.match(r'^[A-Z][a-zA-Z0-9]*$', i))
            c3 = sum(1 for i in identifiers if re.match(r'^[a-z_][a-z0-9_]*$', i))
            c4 = sum(1 for i in identifiers if re.match(r'^[A-Z_][A-Z0-9_]*$', i))
            sq = len(re.findall(r"'[^']*'", self.code))
            dq = len(re.findall(r'"[^"]*"', self.code))
            bq = len(re.findall(r'`(?:\\`|[^`])*`', self.code))

            features['camel_case_ratio'] = c1 / max(len(identifiers), 1)
            features['pascal_case_ratio'] = c2 / max(len(identifiers), 1)
            features['snake_case_ratio'] = c3 / max(len(identifiers), 1)
            features['uppercase_ratio'] = c4 / max(len(identifiers), 1)
            features['inline_comment_ratio'] = len(re.findall(r'//[^\n]*', self.code)) / self.total_lines
            features['block_comment_ratio'] = len(re.findall(r'/\*.*?\*/', self.code, re.DOTALL)) / self.total_lines
            features['space_after_comma'] = len(re.findall(r',\s', self.code)) / max(self.code.count(','), 1)
            features['space_before_brace'] = len(re.findall(r'\s\{', self.code)) / max(self.code.count('{'), 1)
            features['single_quote_ratio'] = sq / max(sq + dq + bq, 1)
            features['code_consistency'] = 1 - (abs(sq - dq) / max(sq + dq, 1))
            features['semicolon_ratio'] = len(re.findall(r';(?![)}\]])', self.code)) / self.total_lines
        except Exception:
            pass
        return features


# ==============================================================================
# 2. 语法特征提取器 (Syntactic)
# 包含原版的 structural_features, complexity_features, ast_features
# ==============================================================================
class SyntacticFeatureExtractor:
    def __init__(self, code: str, lines: List[str], total_lines: int, ast_tree):
        self.code = code
        self.lines = lines
        self.total_lines = total_lines
        self.ast_tree = ast_tree

    def get_feature_names(self) -> List[str]:
        names = [
            # Structural (15)
            'function_density', 'arrow_function_ratio', 'traditional_function_ratio',
            'control_flow_density', 'if_else_density', 'loop_density', 'max_nesting_depth',
            'avg_nesting_depth', 'semicolon_density_struct', 'statement_density', 'brace_balance',
            'code_compression_ratio', 'class_density', 'interface_to_class_ratio', 'enum_density',
            # Complexity (6)
            'cyclomatic_complexity', 'halstead_volume', 'halstead_difficulty', 'halstead_effort',
            'code_entropy', 'operator_operand_ratio',
            # AST Static (4)
            'ast_node_count', 'ast_max_depth', 'ast_avg_depth', 'ast_node_diversity'
        ]
        # AST Dynamic (10)
        ast_nodes = ['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression', 'VariableDeclaration',
                     'CallExpression', 'MemberExpression', 'IfStatement', 'ForStatement', 'WhileStatement',
                     'ReturnStatement']
        names.extend([f'ast_{n}_ratio' for n in ast_nodes])
        return names

    def extract(self) -> Dict[str, float]:
        features = {k: 0.0 for k in self.get_feature_names()}
        try:
            func_counts = {
                'function_decl': len(re.findall(r'function\s+\w+\s*\(', self.code)),
                'function_expr': len(re.findall(r'=\s*function\s*\(', self.code)),
                'arrow_func': len(re.findall(r'=>', self.code)),
                'method_def': len(re.findall(r'\w+\s*\([^:)]*\)\s*:\s*\w+\s*\{', self.code))
            }
            total_functions = sum(func_counts.values())
            control_flow = {
                'if_else': len(re.findall(r'\bif\s*\(', self.code)) + len(re.findall(r'\belse\b', self.code)),
                'for_loops': len(re.findall(r'\bfor\s*\(', self.code)),
                'while_loops': len(re.findall(r'\bwhile\s*\(', self.code))
            }

            brace_stack, max_nesting, current_nesting = [], 0, 0
            for char in self.code:
                if char == '{':
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                elif char == '}':
                    current_nesting = max(0, current_nesting - 1)

            class_defs = len(re.findall(r'\bclass\s+\w+', self.code))
            interface_defs = len(re.findall(r'\binterface\s+\w+', self.code))
            enum_defs = len(re.findall(r'\benum\s+\w+', self.code))

            features['function_density'] = total_functions / self.total_lines
            features['arrow_function_ratio'] = func_counts['arrow_func'] / max(total_functions, 1)
            features['traditional_function_ratio'] = func_counts['function_decl'] / max(total_functions, 1)
            features['control_flow_density'] = sum(control_flow.values()) / self.total_lines
            features['if_else_density'] = control_flow['if_else'] / self.total_lines
            features['loop_density'] = (control_flow['for_loops'] + control_flow['while_loops']) / self.total_lines
            features['max_nesting_depth'] = max_nesting
            features['avg_nesting_depth'] = max_nesting / self.total_lines if max_nesting > 0 else 0
            features['semicolon_density_struct'] = len(re.findall(r';(?![)}\]])', self.code)) / self.total_lines
            features['statement_density'] = len(re.findall(r'[;{}]', self.code)) / self.total_lines
            features['brace_balance'] = abs(self.code.count('{') - self.code.count('}')) / max(self.total_lines, 1)
            features['code_compression_ratio'] = len(self.code.strip()) / max(self.total_lines * 50, 1)
            features['class_density'] = class_defs / self.total_lines
            features['interface_to_class_ratio'] = interface_defs / max(class_defs, 1)
            features['enum_density'] = enum_defs / self.total_lines

            # Complexity
            cyclomatic = len(re.findall(r'\b(if|else if|for|while|case|catch)\b', self.code)) + 1
            operators = re.findall(r'[+\-*/%=!&|^~<>?:]|\+\+|\-\-|&&|\|\||===|!==|<<|>>', self.code)
            operands = re.findall(r'\b\d+\b|\b[a-zA-Z_$][a-zA-Z0-9_$]*\b', self.code)
            u_ops, u_opr = len(set(operators)), len(set(operands))
            if u_opr == 0: u_opr = 1
            vol = (len(operators) + len(operands)) * math.log2(u_ops + u_opr) if (u_ops + u_opr) > 0 else 0
            diff = (u_ops / 2) * (len(operands) / u_opr)
            char_freq = Counter(self.code)
            total_c = sum(char_freq.values())
            ent = -sum((c / total_c) * math.log2(c / total_c) for c in char_freq.values() if c > 0)

            features['cyclomatic_complexity'] = cyclomatic
            features['halstead_volume'] = vol
            features['halstead_difficulty'] = diff
            features['halstead_effort'] = vol * diff
            features['code_entropy'] = ent
            features['operator_operand_ratio'] = len(operators) / max(len(operands), 1)

            # Esprima AST
            if self.ast_tree:
                node_counts = defaultdict(int)
                depths = []

                def traverse(node, depth=0):
                    if hasattr(node, 'type'):
                        node_counts[node.type] += 1
                        depths.append(depth)
                        for key, value in node.__dict__.items():
                            if key in ['type', 'loc', 'range']: continue
                            if isinstance(value, list):
                                for child in value:
                                    if hasattr(child, 'type'): traverse(child, depth + 1)
                            elif hasattr(value, 'type'):
                                traverse(value, depth + 1)

                traverse(self.ast_tree)
                if depths:
                    features['ast_node_count'] = sum(node_counts.values())
                    features['ast_max_depth'] = max(depths)
                    features['ast_avg_depth'] = statistics.mean(depths)
                    features['ast_node_diversity'] = len(node_counts) / sum(node_counts.values())

                    ast_nodes = ['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression',
                                 'VariableDeclaration',
                                 'CallExpression', 'MemberExpression', 'IfStatement', 'ForStatement', 'WhileStatement',
                                 'ReturnStatement']
                    for n in ast_nodes:
                        features[f'ast_{n}_ratio'] = node_counts.get(n, 0) / sum(node_counts.values())
        except Exception:
            pass
        return features


# ==============================================================================
# 3. JS 专属特性 (JSSpecific)
# 包含原版的 modern_features, js_cp_idioms
# ==============================================================================
class JSSpecificFeatureExtractor:
    def __init__(self, code: str, lines: List[str], total_lines: int):
        self.code = code
        self.lines = lines
        self.total_lines = total_lines
        self.length_chars = max(len(self.code), 1)

    def get_feature_names(self) -> List[str]:
        return [
            # Modern JS Features (8)
            'arrow_function_density_modern', 'template_literal_density', 'destructuring_density',
            'spread_operator_density',
            'modern_class_density', 'module_density', 'async_await_density', 'modern_variable_ratio',
            # JS CP Idioms (5)
            'cp_fast_io_density', 'cp_bitwise_math_floor_density', 'cp_mod_constant_flag',
            'cp_array_fill_density', 'cp_short_circuit_call_density'
        ]

    def extract(self) -> Dict[str, float]:
        features = {k: 0.0 for k in self.get_feature_names()}
        try:
            lc = self.length_chars
            tl = self.total_lines

            # Modern Features
            features['arrow_function_density_modern'] = len(re.findall(r'=>', self.code)) / tl
            features['template_literal_density'] = len(re.findall(r'`(?:\\`|[^`])*`', self.code)) / tl
            features['destructuring_density'] = len(re.findall(r'const\s*\{|let\s*\{|var\s*\{', self.code)) / tl
            features['spread_operator_density'] = len(re.findall(r'\.{3}', self.code)) / tl
            features['modern_class_density'] = len(re.findall(r'\bclass\s+\w+', self.code)) / tl
            features['module_density'] = (len(re.findall(r'\bimport\b', self.code)) + len(
                re.findall(r'\bmodule\.exports\b', self.code))) / tl
            features['async_await_density'] = (len(re.findall(r'\basync\b', self.code)) + len(
                re.findall(r'\bawait\b', self.code))) / tl

            var_cnt = len(re.findall(r'\bvar\b', self.code))
            let_const_cnt = len(re.findall(r'\b(let|const)\b', self.code))
            features['modern_variable_ratio'] = let_const_cnt / max(var_cnt + let_const_cnt, 1)

            # CP Idioms
            features['cp_fast_io_density'] = len(re.findall(r'fs\.readFileSync\s*\(', self.code)) / lc
            features['cp_bitwise_math_floor_density'] = len(re.findall(r'~~|\b\w+\s*\|\s*0\b', self.code)) / lc
            features['cp_mod_constant_flag'] = 1.0 if len(
                re.findall(r'\b(1000000007|1e9\s*\+\s*7|998244353)\b', self.code)) > 0 else 0.0
            features['cp_array_fill_density'] = len(re.findall(r'new\s+Array\s*\([^)]*\)\.fill\(', self.code)) / lc
            features['cp_short_circuit_call_density'] = len(re.findall(r'&&|\|\|', self.code)) / lc

        except Exception:
            pass
        return features


# ==============================================================================
# 4. 人类失误与排版特征 (HumanError)
# 包含原版的 ai_patterns, micro_habits
# ==============================================================================
class HumanErrorFeatureExtractor:
    def __init__(self, code: str, lines: List[str], total_lines: int):
        self.code = code
        self.code_lower = code.lower()
        self.lines = lines
        self.total_lines = total_lines
        self.length_chars = max(len(self.code), 1)

    def get_feature_names(self) -> List[str]:
        return [
            # AI Patterns
            'console_usage', 'simple_function_names', 'generic_comments', 'magic_numbers',
            'long_lines', 'repetitive_patterns', 'over_documented',
            # Micro Habits
            'micro_loose_eq_ratio', 'micro_missing_space_kw_ratio', 'micro_crowded_operator_density',
            'micro_console_log_density', 'micro_typo_density', 'micro_human_atypical_density', 'micro_informal_tag_density',
            # Expert Errors
            'err_empty_catch_ratio', 'err_perfect_indent_with_crowded_ops'
        ]

    def extract(self) -> Dict[str, float]:
        features = {k: 0.0 for k in self.get_feature_names()}
        try:
            lc = self.length_chars
            tl = self.total_lines

            # AI Patterns
            features['console_usage'] = len(re.findall(r'console\.(log|error|warn|info)', self.code)) / tl
            features['simple_function_names'] = len(
                re.findall(r'function\s+(test|demo|example|foo|bar)', self.code, re.I)) / tl
            features['generic_comments'] = len(re.findall(r'//\s*(TODO|FIXME|NOTE|HACK)', self.code)) / tl
            features['magic_numbers'] = len(re.findall(r'\b\d{3,}\b', self.code)) / tl
            features['long_lines'] = sum(1 for line in self.lines if len(line) > 80) / tl
            features['repetitive_patterns'] = sum(
                1 for count in Counter([line.strip() for line in self.lines if line.strip()]).values() if
                count > 2) / tl
            features['over_documented'] = len(re.findall(r'/\*\*(?:[^*]|\*[^/])*\*/', self.code, re.DOTALL)) / tl

            # Micro Habits
            loose_eq = len(re.findall(r'[^=!]==[^=]', self.code)) + len(re.findall(r'[^=]!=[^=]', self.code))
            strict_eq = len(re.findall(r'===', self.code)) + len(re.findall(r'!==', self.code))
            features['micro_loose_eq_ratio'] = loose_eq / (loose_eq + strict_eq + 1e-10)

            missing_space_kw = len(re.findall(r'\b(if|for|while|switch|catch)\(', self.code))
            has_space_kw = len(re.findall(r'\b(if|for|while|switch|catch)\s+\(', self.code))
            features['micro_missing_space_kw_ratio'] = missing_space_kw / (missing_space_kw + has_space_kw + 1e-10)

            crowded_ops = len(
                re.findall(r'[a-zA-Z0-9_](==|===|!=|!==|<=|>=|\+=|-=|\*=|/=|=(?!=)|<|>)[a-zA-Z0-9_]', self.code))
            features['micro_crowded_operator_density'] = crowded_ops / lc
            features['micro_console_log_density'] = len(re.findall(r'\bconsole\.log\s*\(', self.code)) / lc

            code_lower = self.code.lower()
            common_typos = {'lenght', 'heigth', 'recieve', 'occured', 'sucess', 'retun'}
            informal_tags = {'todo', 'fixme', 'hack', 'wtf', 'shit', 'resources', 'test'}
            human_atypical = {'aids', 'composed', 'development', 'materials', 'segments', 'utilities', 'content', 'samples' , 'blocks', 'via', 'third-party', 'borrowed'}
            features['micro_typo_density'] = sum(
                len(re.findall(r'\b' + typo + r'\b', code_lower)) for typo in common_typos) / lc
            features['micro_human_atypical_density'] = sum(
                len(re.findall(r'\b' + atypical + r'\b', self.code_lower)) for atypical in human_atypical) / lc
            features['micro_informal_tag_density'] = sum(
                len(re.findall(r'\b' + tag + r'\b', code_lower)) for tag in informal_tags) / lc

            # Expert additions
            catches = len(re.findall(r'\bcatch\b', self.code)) + 1e-10
            empty_catches = len(re.findall(r'catch\s*\([^)]*\)\s*\{\s*\}', self.code))
            features['err_empty_catch_ratio'] = empty_catches / catches


            irregular_indent = sum(
                1 for l in self.lines if l.startswith(' ') and (len(l) - len(l.lstrip(' '))) % 2 != 0) / tl
            features['err_perfect_indent_with_crowded_ops'] = 1.0 if irregular_indent < 0.05 and (
                        crowded_ops / lc) > 0.01 else 0.0

        except Exception:
            pass
        return features


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
                batch_codes = [c if c and c.strip() else "// empty" for c in batch_codes]

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
# 批量处理包装器 (对接主程序)
# =================================================================================
class JSBatchExtractorWrapper:
    def __init__(self, config=None):
        self.config = config
        self.feature_names = []
        self._is_names_initialized = False
        self._codeberta_names_added = False

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        logger.info(f"开始提取 {len(codes)} 个 JavaScript 文件的全量多维特征...")
        features_list = []

        # 在循环外先获取 FEATURES 配置字典
        features_config = getattr(self.config, 'FEATURES', {})

        # 1. 单线程串行提取传统特征
        for idx, code in enumerate(codes):
            if idx % 500 == 0 and idx > 0:
                logger.info(f"  传统特征进度: {idx} / {len(codes)}")

            lines = code.split('\n')
            total_lines = max(len(lines), 1)
            
            feat_dict = {}

            # ==========================================
            # 🚨 修复：引入 esprima 的 AST 解析，并修改为 JS 专属提取器
            # ==========================================
            if features_config.get('lexical', True):
                lexical = LexicalFeatureExtractor(code, lines, total_lines)
                feat_dict.update(lexical.extract())

            if features_config.get('syntactic', True):
                ast_tree = None
                if ESPRIMA_AVAILABLE:
                    try:
                        # JS版语法特征必须提前解析 AST
                        ast_tree = esprima.parseScript(code, jsx=True)
                    except Exception:
                        pass
                syntactic = SyntacticFeatureExtractor(code, lines, total_lines, ast_tree) # 👈 补齐第4个参数
                feat_dict.update(syntactic.extract())

            if features_config.get('js_specific', True):  # 👈 改为 js_specific
                js_specific = JSSpecificFeatureExtractor(code, lines, total_lines) # 👈 调用 JS 专属类
                feat_dict.update(js_specific.extract())

            if features_config.get('human_error', True):
                human_error = HumanErrorFeatureExtractor(code, lines, total_lines)
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
# # 批量处理包装器 (对接主程序)
# # =================================================================================
# class JSBatchExtractorWrapper:
#     def __init__(self, config=None):
#         self.feature_names = []
#         self._is_names_initialized = False
#
#     def get_feature_names(self) -> List[str]:
#         return self.feature_names
#
#     def extract_batch(self, codes: List[str]) -> np.ndarray:
#         logger.info(f"开始提取 {len(codes)} 个 JavaScript 文件的全量多维特征 (锁定 106 维无损版)...")
#
#         features_list = []
#         for idx, code in enumerate(codes):
#             if idx % 500 == 0 and idx > 0:
#                 logger.info(f"  已提取: {idx} / {len(codes)}")
#
#             lines = code.split('\n')
#             total_lines = max(len(lines), 1)
#
#             ast_tree = None
#             if ESPRIMA_AVAILABLE:
#                 try:
#                     ast_tree = esprima.parseScript(code, jsx=True)
#                 except Exception:
#                     pass
#
#             # 实例化四大提取器
#             lexical = LexicalFeatureExtractor(code, lines, total_lines)
#             syntactic = SyntacticFeatureExtractor(code, lines, total_lines, ast_tree)
#             js_specific = JSSpecificFeatureExtractor(code, lines, total_lines)
#             human_error = HumanErrorFeatureExtractor(code, lines, total_lines)
#
#             # 融合字典
#             feat_dict = {}
#             feat_dict.update(lexical.extract())
#             feat_dict.update(syntactic.extract())
#             feat_dict.update(js_specific.extract())
#             feat_dict.update(human_error.extract())
#
#             if not self._is_names_initialized:
#                 # 记录所有的键名，彻底锁死数据维度
#                 self.feature_names = sorted(list(feat_dict.keys()))
#                 self._is_names_initialized = True
#
#             # 严格按照锁定的 Key 顺序填充
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
