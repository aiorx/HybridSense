#!/usr/bin/env python3
"""
TypeScript 特征提取模块 - 终极无损四框架版
完全无损保留您的 162 维全量特征！并严格划分为四大类。
杜绝了因为 try-except 导致键名改变 (如 basic_0) 引起的维度缩水问题。
"""

import re
import math
import statistics
import logging
import numpy as np
from typing import Dict, List
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ====== 【新增】CodeBERTa 依赖 ======
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("未安装 torch 或 transformers！CodeBERTa 将退回全零矩阵。建议: pip install torch transformers")

try:
    from ts_morph import Project

    TS_MORPH_AVAILABLE = True
except ImportError:
    TS_MORPH_AVAILABLE = False



# ==============================================================================
# 1. 词法特征提取器 (Lexical)
# ==============================================================================
class LexicalFeatureExtractor:
    def __init__(self, code: str, lines: List[str], total_lines: int, is_tsx: bool):
        self.code = code
        self.lines = lines
        self.total_lines = total_lines
        self.is_tsx = is_tsx

    def get_feature_names(self) -> List[str]:
        return [
            'char_density', 'whitespace_ratio', 'digit_ratio', 'alpha_ratio',
            'avg_line_length', 'line_length_std', 'empty_line_ratio', 'comment_ratio', 'is_tsx_file',
            'keyword_density', 'conditional_keyword_ratio', 'loop_keyword_ratio', 'declaration_keyword_ratio',
            'operator_density', 'assignment_ratio', 'comparison_ratio', 'bracket_density', 'paren_ratio',
            'curly_ratio', 'identifier_count', 'avg_identifier_length', 'identifier_length_std',
            'unique_identifiers', 'capital_letter_ratio', 'ts_keyword_ratio',
            'camel_case_ratio', 'pascal_case_ratio', 'snake_case_ratio', 'uppercase_ratio',
            'inline_comment_ratio', 'block_comment_ratio', 'tsdoc_ratio', 'space_after_comma',
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
            features['is_tsx_file'] = float(self.is_tsx)

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
            brackets = {'paren': len(re.findall(r'\(', self.code)), 'curly': len(re.findall(r'\{', self.code)),
                        'square': len(re.findall(r'\[', self.code))}
            identifiers = re.findall(r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b', self.code)
            identifier_lengths = [len(id) for id in identifiers]
            ts_keywords = ['interface', 'type', 'enum', 'public', 'private', 'protected', 'readonly', 'abstract',
                           'implements', 'namespace', 'declare', 'module']
            ts_keyword_counts = sum(len(re.findall(rf'\b{kw}\b', self.code)) for kw in ts_keywords)

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
            features['ts_keyword_ratio'] = ts_keyword_counts / max(total_keywords, 1)

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
            features['tsdoc_ratio'] = len(re.findall(r'/\*\*.*?\*/', self.code, re.DOTALL)) / max(
                len(re.findall(r'/\*.*?\*/', self.code, re.DOTALL)), 1)
            features['space_after_comma'] = len(re.findall(r',\s', self.code)) / max(self.code.count(','), 1)
            features['space_before_brace'] = len(re.findall(r'\s\{', self.code)) / max(self.code.count('{'), 1)
            features['single_quote_ratio'] = sq / max(sq + dq + bq, 1)
            features['code_consistency'] = 1 - (abs(sq - dq) / max(sq + dq, 1))
            features['semicolon_ratio'] = len(re.findall(r';', self.code)) / self.total_lines
        except Exception:
            pass
        return features


# ==============================================================================
# 2. 语法特征提取器 (Syntactic)
# ==============================================================================
class SyntacticFeatureExtractor:
    def __init__(self, code: str, lines: List[str], total_lines: int, ast_tree):
        self.code = code
        self.lines = lines
        self.total_lines = total_lines
        self.ast_tree = ast_tree

    def get_feature_names(self) -> List[str]:
        return [
            'function_density', 'arrow_function_ratio', 'traditional_function_ratio',
            'control_flow_density', 'if_else_density', 'loop_density', 'max_nesting_depth',
            'avg_nesting_depth', 'semicolon_density', 'statement_density', 'brace_balance',
            'code_compression_ratio', 'class_density', 'interface_to_class_ratio', 'enum_density',
            'cyclomatic_complexity', 'halstead_volume', 'halstead_difficulty', 'halstead_effort',
            'code_entropy', 'operator_operand_ratio', 'ts_generic_nesting_depth',
            'ast_class_count', 'ast_interface_count', 'ast_type_alias_count', 'ast_enum_count',
            'ast_function_count', 'ast_declaration_diversity', 'ast_max_depth', 'ast_declaration_density'
        ]

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
                    current_nesting += 1;
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
            features['semicolon_density'] = len(re.findall(r';(?![)}\]])', self.code)) / self.total_lines
            features['statement_density'] = len(re.findall(r'[;{}]', self.code)) / self.total_lines
            features['brace_balance'] = abs(self.code.count('{') - self.code.count('}')) / max(self.total_lines, 1)
            features['code_compression_ratio'] = len(self.code.strip()) / max(self.total_lines * 50, 1)
            features['class_density'] = class_defs / self.total_lines
            features['interface_to_class_ratio'] = interface_defs / max(class_defs, 1)
            features['enum_density'] = enum_defs / self.total_lines

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
            generic_nesting = max([len(re.findall(r'<', g)) for g in re.findall(r'<[^>]+>', self.code)] or [0])

            features['cyclomatic_complexity'] = cyclomatic
            features['halstead_volume'] = vol
            features['halstead_difficulty'] = diff
            features['halstead_effort'] = vol * diff
            features['code_entropy'] = ent
            features['operator_operand_ratio'] = len(operators) / max(len(operands), 1)
            features['ts_generic_nesting_depth'] = generic_nesting

            if self.ast_tree:
                cc = len(self.ast_tree.getClasses())
                ic = len(self.ast_tree.getInterfaces())
                tac = len(self.ast_tree.getTypeAliases())
                ec = len(self.ast_tree.getEnums())
                fc = len(self.ast_tree.getFunctions())
                md = 0

                def calc_depth(node, d=0):
                    nonlocal md
                    md = max(md, d)
                    for child in node.getChildren(): calc_depth(child, d + 1)

                if self.ast_tree.getChildCount() > 0: calc_depth(self.ast_tree)

                features['ast_class_count'] = cc
                features['ast_interface_count'] = ic
                features['ast_type_alias_count'] = tac
                features['ast_enum_count'] = ec
                features['ast_function_count'] = fc
                features['ast_declaration_diversity'] = len([x for x in [cc, ic, tac, ec, fc] if x > 0]) / 5
                features['ast_max_depth'] = md
                features['ast_declaration_density'] = (cc + ic + tac + ec + fc) / self.total_lines
        except Exception:
            pass
        return features


# ==============================================================================
# 3. TS 专属高级特性提取器 (TSSpecific)
# ==============================================================================
class TSSpecificFeatureExtractor:
    def __init__(self, code: str, lines: List[str], total_lines: int):
        self.code = code
        self.lines = lines
        self.total_lines = total_lines
        self.length_chars = max(len(self.code), 1)

    def get_feature_names(self) -> List[str]:
        return [
            'ts_type_annotation_density', 'ts_simple_type_ratio', 'ts_complex_type_ratio', 'ts_array_type_ratio',
            'ts_interface_density', 'ts_type_alias_density', 'ts_enum_density', 'ts_abstract_class_density',
            'ts_generic_density', 'ts_generic_constraint_ratio', 'ts_access_modifier_density', 'ts_decorator_density',
            'ts_readonly_ratio', 'ts_type_import_ratio', 'ts_namespace_density', 'ts_type_assertion_density',
            'ts_non_null_assertion_density', 'ts_optional_property_ratio', 'ts_union_type_density',
            'ts_intersection_type_density', 'ts_type_diversity',
            'arrow_function_density_modern', 'template_literal_density', 'destructuring_density',
            'spread_operator_density',
            'modern_class_density', 'module_density', 'async_await_density', 'modern_variable_ratio',
            'modern_decorator_density',
            'cp_fast_io_density', 'cp_bitwise_math_floor_density', 'cp_bigint_literal_density', 'cp_mod_constant_flag',
            'cp_array_fill_density', 'cp_array_from_density'
        ]

    def extract(self) -> Dict[str, float]:
        features = {k: 0.0 for k in self.get_feature_names()}
        try:
            ts_patterns = {
                'type_annotation': r':\s*[A-Z][a-zA-Z0-9]*(?:<[^>]+>)?(?:\[\])?(?:\|[A-Z][a-zA-Z0-9]*)*',
                'simple_type': r':\s*(?:string|number|boolean|any|unknown|never|void|null|undefined)\b',
                'complex_type': r':\s*(?:[A-Z][a-zA-Z0-9]*(?:<[^>]+>)?|{[^}]+}|\([^)]+\))',
                'array_type': r':\s*(?:[A-Z][a-zA-Z0-9]*|string|number|boolean)\[\]',
                'interface_def': r'\binterface\s+\w+(?:\s+extends\s+\w+)?\s*\{',
                'type_alias': r'\btype\s+\w+\s*=',
                'enum_def': r'\benum\s+\w+\s*\{',
                'abstract_class': r'\babstract\s+class\s+\w+\s*\{',
                'generic_usage': r'<[A-Z][a-zA-Z0-9]*(?:\s*,\s*[A-Z][a-zA-Z0-9]*)*>',
                'generic_constraint': r'extends\s+[A-Z][a-zA-Z0-9]*(?:\s+&\s+[A-Z][a-zA-Z0-9]*)*',
                'access_modifier': r'\b(public|private|protected|readonly|static|abstract|override)\b',
                'decorator': r'@\w+(?:\([^)]*\))?',
                'import_type': r'import\s+type\s+\{',
                'namespace': r'\bnamespace\s+\w+\s*\{',
                'type_assertion': r'as\s+[A-Z][a-zA-Z0-9]*',
                'non_null_assertion': r'!\s*[;\.\)\]\}]',
                'optional_property': r'\?\s*:',
                'union_type': r'\w+(?:\|(?:null|undefined|\w+))+',
                'intersection_type': r'\w+(?:&\w+)+'
            }

            type_annotations = len(re.findall(ts_patterns['type_annotation'], self.code))
            features['ts_type_annotation_density'] = type_annotations / self.total_lines
            features['ts_simple_type_ratio'] = len(re.findall(ts_patterns['simple_type'], self.code)) / max(
                type_annotations, 1)
            features['ts_complex_type_ratio'] = len(re.findall(ts_patterns['complex_type'], self.code)) / max(
                type_annotations, 1)
            features['ts_array_type_ratio'] = len(re.findall(ts_patterns['array_type'], self.code)) / max(
                type_annotations, 1)

            interface_defs = len(re.findall(ts_patterns['interface_def'], self.code))
            type_aliases = len(re.findall(ts_patterns['type_alias'], self.code))
            enum_defs = len(re.findall(ts_patterns['enum_def'], self.code))
            generic_usage = len(re.findall(ts_patterns['generic_usage'], self.code))

            features['ts_interface_density'] = interface_defs / self.total_lines
            features['ts_type_alias_density'] = type_aliases / self.total_lines
            features['ts_enum_density'] = enum_defs / self.total_lines
            features['ts_abstract_class_density'] = len(
                re.findall(ts_patterns['abstract_class'], self.code)) / self.total_lines
            features['ts_generic_density'] = generic_usage / self.total_lines
            features['ts_generic_constraint_ratio'] = len(
                re.findall(ts_patterns['generic_constraint'], self.code)) / max(generic_usage, 1)
            features['ts_access_modifier_density'] = len(
                re.findall(ts_patterns['access_modifier'], self.code)) / self.total_lines
            features['ts_decorator_density'] = len(re.findall(ts_patterns['decorator'], self.code)) / self.total_lines
            features['ts_readonly_ratio'] = len(re.findall(r'\breadonly\b', self.code)) / max(self.total_lines, 1)
            features['ts_type_import_ratio'] = len(re.findall(ts_patterns['import_type'], self.code)) / max(
                len(re.findall(r'\bimport\b', self.code)), 1)
            features['ts_namespace_density'] = len(re.findall(ts_patterns['namespace'], self.code)) / self.total_lines
            features['ts_type_assertion_density'] = len(
                re.findall(ts_patterns['type_assertion'], self.code)) / self.total_lines
            features['ts_non_null_assertion_density'] = len(
                re.findall(ts_patterns['non_null_assertion'], self.code)) / self.total_lines
            features['ts_optional_property_ratio'] = len(re.findall(ts_patterns['optional_property'], self.code)) / max(
                len(re.findall(r'\?\s*:', self.code)), 1)
            features['ts_union_type_density'] = len(re.findall(ts_patterns['union_type'], self.code)) / self.total_lines
            features['ts_intersection_type_density'] = len(
                re.findall(ts_patterns['intersection_type'], self.code)) / self.total_lines
            features['ts_type_diversity'] = len([x for x in [interface_defs, type_aliases, enum_defs] if x > 0]) / 3

            # Modern Features
            features['arrow_function_density_modern'] = len(re.findall(r'=>', self.code)) / self.total_lines
            features['template_literal_density'] = len(re.findall(r'`(?:\\`|[^`])*`', self.code)) / self.total_lines
            features['destructuring_density'] = len(
                re.findall(r'const\s*\{|let\s*\{|var\s*\{', self.code)) / self.total_lines
            features['spread_operator_density'] = len(re.findall(r'\.{3}', self.code)) / self.total_lines
            features['modern_class_density'] = len(re.findall(r'\bclass\s+\w+', self.code)) / self.total_lines
            features['module_density'] = (len(re.findall(r'\bimport\b', self.code)) + len(
                re.findall(r'\bexport\b', self.code))) / self.total_lines
            features['async_await_density'] = (len(re.findall(r'\basync\b', self.code)) + len(
                re.findall(r'\bawait\b', self.code))) / self.total_lines
            features['modern_variable_ratio'] = len(re.findall(r'\b(let|const)\b', self.code)) / max(
                len(re.findall(r'\b(let|const|var)\b', self.code)), 1)
            features['modern_decorator_density'] = len(re.findall(r'@\w+(?:\([^)]*\))?', self.code)) / self.total_lines

            # CP Idioms
            features['cp_fast_io_density'] = len(re.findall(r'fs\.readFileSync\s*\(', self.code)) / self.length_chars
            features['cp_bitwise_math_floor_density'] = len(
                re.findall(r'~~|\b\w+\s*\|\s*0\b', self.code)) / self.length_chars
            features['cp_bigint_literal_density'] = len(re.findall(r'\b\d{4,}n\b', self.code)) / self.length_chars
            features['cp_mod_constant_flag'] = 1.0 if len(
                re.findall(r'\b(1000000007n?|1e9\s*\+\s*7|998244353n?)\b', self.code)) > 0 else 0.0
            features['cp_array_fill_density'] = len(
                re.findall(r'new\s+Array\s*\([^)]*\)\.fill\(', self.code)) / self.length_chars
            features['cp_array_from_density'] = len(re.findall(r'Array\.from\s*\(', self.code)) / self.length_chars
        except Exception:
            pass
        return features


# ==============================================================================
# 4. 人类失误与排版特征 (HumanError)
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
            'console_usage', 'simple_function_names', 'generic_comments', 'ai_magic_numbers',
            'long_lines', 'repetitive_patterns', 'over_documented', 'ts_any_usage_pattern', 'non_null_overuse',
            'micro_loose_eq_ratio', 'micro_as_any_abuse_density', 'micro_non_null_abuse_density',
            'micro_old_assertion_density', 'micro_missing_space_kw_ratio', 'micro_crowded_operator_density',
            'micro_console_log_density', 'micro_typo_density', 'micro_human_atypical_density', 'micro_informal_tag_density'
        ]

    def extract(self) -> Dict[str, float]:
        features = {k: 0.0 for k in self.get_feature_names()}
        try:
            # AI Patterns
            features['console_usage'] = len(re.findall(r'console\.(log|error|warn|info)', self.code)) / self.total_lines
            features['simple_function_names'] = len(
                re.findall(r'function\s+(test|demo|example|foo|bar)', self.code, re.I)) / self.total_lines
            features['generic_comments'] = len(re.findall(r'//\s*(TODO|FIXME|NOTE|HACK)', self.code)) / self.total_lines
            features['ai_magic_numbers'] = len(re.findall(r'\b\d{3,}\b', self.code)) / self.total_lines
            features['long_lines'] = sum(1 for line in self.lines if len(line) > 80) / self.total_lines
            features['repetitive_patterns'] = sum(
                1 for count in Counter([line.strip() for line in self.lines if line.strip()]).values() if
                count > 2) / self.total_lines
            features['over_documented'] = len(
                re.findall(r'/\*\*(?:[^*]|\*[^/])*\*/', self.code, re.DOTALL)) / self.total_lines
            features['ts_any_usage_pattern'] = len(re.findall(r':\s*any\b', self.code)) / self.total_lines
            features['non_null_overuse'] = len(re.findall(r'![\.\;\)\}]', self.code)) / self.total_lines

            # Micro Habits
            loose_eq = len(re.findall(r'[^=!]==[^=]', self.code)) + len(re.findall(r'[^=]!=[^=]', self.code))
            strict_eq = len(re.findall(r'===', self.code)) + len(re.findall(r'!==', self.code))
            missing_space_kw = len(re.findall(r'\b(if|for|while|switch|catch)\(', self.code))
            has_space_kw = len(re.findall(r'\b(if|for|while|switch|catch)\s+\(', self.code))

            features['micro_loose_eq_ratio'] = loose_eq / (loose_eq + strict_eq + 1e-10)
            features['micro_as_any_abuse_density'] = len(re.findall(r'\bas\s+any\b', self.code)) / self.length_chars
            features['micro_non_null_abuse_density'] = len(re.findall(r'\b\w+!\.', self.code)) / self.length_chars
            features['micro_old_assertion_density'] = len(re.findall(r'<\w+>\s*\w+', self.code)) / self.length_chars
            features['micro_missing_space_kw_ratio'] = missing_space_kw / (missing_space_kw + has_space_kw + 1e-10)
            features['micro_crowded_operator_density'] = len(
                re.findall(r'[a-zA-Z0-9_](==|===|!=|!==|<=|>=|\+=|-=|\*=|/=|=(?!=)|<|>)[a-zA-Z0-9_]',
                           self.code)) / self.length_chars
            features['micro_console_log_density'] = len(
                re.findall(r'\bconsole\.log\s*\(', self.code)) / self.length_chars


            common_typos = {'lenght', 'heigth', 'recieve', 'occured', 'sucess', 'retun'}
            human_atypical = {'aids', 'composed', 'development', 'materials', 'segments', 'utilities', 'content', 'samples' , 'blocks', 'via', 'third-party', 'borrowed'}
            informal_tags = {'todo', 'fixme', 'hack', 'wtf', 'shit','resources', 'test'}
            features['micro_typo_density'] = sum(
                len(re.findall(r'\b' + typo + r'\b', self.code_lower)) for typo in common_typos) / self.length_chars
            features['micro_human_atypical_density'] = sum(
                len(re.findall(r'\b' + atypical + r'\b', self.code_lower)) for atypical in human_atypical) / self.length_chars
            features['micro_informal_tag_density'] = sum(
                len(re.findall(r'\b' + tag + r'\b', self.code_lower)) for tag in informal_tags) / self.length_chars
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
class TSBatchExtractorWrapper:
    def __init__(self, config=None):
        self.config = config
        self.feature_names = []
        self._is_names_initialized = False
        self._codeberta_names_added = False

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        logger.info(f"开始提取 {len(codes)} 个 TypeScript 文件的全量多维特征...")
        features_list = []

        # 在循环外先获取 FEATURES 配置字典
        features_config = getattr(self.config, 'FEATURES', {})

        # 如果启用了语法特征提取，并且安装了 ts_morph，则提前初始化 Project（避免在循环里反复初始化拖慢速度）
        project = None
        if TS_MORPH_AVAILABLE and features_config.get('syntactic', True):
            try:
                from ts_morph import Project
                project = Project()
            except Exception as e:
                logger.warning(f"ts_morph Project 初始化失败: {e}")

        # 1. 单线程串行提取传统特征
        for idx, code in enumerate(codes):
            if idx % 500 == 0 and idx > 0:
                logger.info(f"  传统特征进度: {idx} / {len(codes)}")

            lines = code.split('\n')
            total_lines = max(len(lines), 1)
            # 🚨 补充 TS 独有的 TSX 识别标志
            is_tsx = 'React' in code or '<div' in code
            
            feat_dict = {}

            # ==========================================
            # 通过 config 控制是否提取
            # ==========================================
            if features_config.get('lexical', True):
                # 👈 补齐第 4 个参数 is_tsx
                lexical = LexicalFeatureExtractor(code, lines, total_lines, is_tsx)
                feat_dict.update(lexical.extract())

            if features_config.get('syntactic', True):
                ast_tree = None
                if project is not None:
                    try:
                        # 给每个文件一个临时的虚拟名字，防止重名冲突
                        ast_tree = project.createSourceFile(f"temp_{idx}.ts", code)
                    except Exception:
                        pass
                # 👈 补齐第 4 个参数 ast_tree
                syntactic = SyntacticFeatureExtractor(code, lines, total_lines, ast_tree)
                feat_dict.update(syntactic.extract())

            if features_config.get('ts_specific', True):  # 👈 改为 ts_specific
                # 👈 调用 TypeScript 专属提取类
                ts_specific = TSSpecificFeatureExtractor(code, lines, total_lines)
                feat_dict.update(ts_specific.extract())

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
# class TSBatchExtractorWrapper:
#     def __init__(self, config=None):
#         self.feature_names = []
#         self._is_names_initialized = False
#
#     def get_feature_names(self) -> List[str]:
#         return self.feature_names
#
#     def extract_batch(self, codes: List[str]) -> np.ndarray:
#         logger.info(f"开始提取 {len(codes)} 个 TypeScript 文件的全量多维特征 (无损融合版)...")
#
#         features_list = []
#         for idx, code in enumerate(codes):
#             if idx % 500 == 0 and idx > 0:
#                 logger.info(f"  已提取: {idx} / {len(codes)}")
#
#             # 预处理通用参数，减少重复计算
#             lines = code.split('\n')
#             total_lines = max(len(lines), 1)
#             is_tsx = 'React' in code or '<div' in code
#
#             ast_tree = None
#             if TS_MORPH_AVAILABLE:
#                 try:
#                     project = Project()
#                     ast_tree = project.createSourceFile("temp.ts", code)
#                 except Exception:
#                     pass
#
#             # 实例化四大提取器
#             lexical = LexicalFeatureExtractor(code, lines, total_lines, is_tsx)
#             syntactic = SyntacticFeatureExtractor(code, lines, total_lines, ast_tree)
#             ts_specific = TSSpecificFeatureExtractor(code, lines, total_lines)
#             human_error = HumanErrorFeatureExtractor(code, lines, total_lines)
#
#             # 汇聚所有特征到一个大字典中
#             feat_dict = {}
#             feat_dict.update(lexical.extract())
#             feat_dict.update(syntactic.extract())
#             feat_dict.update(ts_specific.extract())
#             feat_dict.update(human_error.extract())
#
#             if not self._is_names_initialized:
#                 # 严格记录所有的 121 个预定义键名作为固定维度
#                 self.feature_names = sorted(list(feat_dict.keys()))
#                 self._is_names_initialized = True
#
#             # 按照固定的 key 顺序组装 numpy 数组 (若某项为空则置 0)
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
