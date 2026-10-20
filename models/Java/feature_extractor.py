#!/usr/bin/env python3
"""
Java 特征提取模块
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

# ====== 【新增】CodeBERTa 依赖 ======
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("未安装 torch 或 transformers！CodeBERTa 将退回全零矩阵。")

try:
    import javalang

    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False
    logger.warning("javalang 未安装，部分 AST 特征将置为 0。建议: pip install javalang")


# ==============================================================================
# 1. 词法特征提取器 (Lexical)
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
            # Lexical Features (14)
            'keyword_density', 'conditional_keyword_ratio', 'loop_keyword_ratio',
            'operator_density', 'assignment_ratio', 'comparison_ratio', 'bracket_density', 'paren_ratio',
            'curly_ratio', 'identifier_count', 'avg_identifier_length', 'identifier_length_std',
            'unique_identifiers', 'capital_letter_ratio',
            # Style Features (12)
            'camel_case_ratio', 'pascal_case_ratio', 'snake_case_ratio', 'uppercase_ratio',
            'inline_comment_ratio', 'block_comment_ratio', 'javadoc_ratio', 'space_after_comma',
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

            java_keywords = ['if', 'else', 'for', 'while', 'public', 'private', 'class', 'int', 'String', 'return']
            keyword_counts = {kw: len(re.findall(rf'\b{kw}\b', self.code)) for kw in java_keywords}
            total_keywords = sum(keyword_counts.values())
            operators = {
                'assignment': len(re.findall(r'=(?!=)', self.code)),
                'comparison': len(re.findall(r'[!=]==|[<>]=?|!?=', self.code)),
                'arithmetic': len(re.findall(r'[+\-*/%]|\+\+|\-\-', self.code)),
                'logical': len(re.findall(r'&&|\|\||!', self.code)),
                'bitwise': len(re.findall(r'[&|^~]|[<>]{2,3}', self.code))
            }
            brackets = {'paren': len(re.findall(r'\(', self.code)), 'curly': len(re.findall(r'\{', self.code))}
            identifiers = re.findall(r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b', self.code)
            identifier_lengths = [len(id) for id in identifiers]

            features['keyword_density'] = total_keywords / self.total_lines
            features['conditional_keyword_ratio'] = (keyword_counts['if'] + keyword_counts['else']) / max(
                total_keywords, 1)
            features['loop_keyword_ratio'] = (keyword_counts['for'] + keyword_counts['while']) / max(total_keywords, 1)
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

            features['camel_case_ratio'] = c1 / max(len(identifiers), 1)
            features['pascal_case_ratio'] = c2 / max(len(identifiers), 1)
            features['snake_case_ratio'] = c3 / max(len(identifiers), 1)
            features['uppercase_ratio'] = c4 / max(len(identifiers), 1)
            features['inline_comment_ratio'] = len(re.findall(r'//[^\n]*', self.code)) / self.total_lines
            features['block_comment_ratio'] = len(re.findall(r'/\*.*?\*/', self.code, re.DOTALL)) / self.total_lines
            features['javadoc_ratio'] = len(re.findall(r'/\*\*.*?\*/', self.code, re.DOTALL)) / max(
                len(re.findall(r'/\*.*?\*/', self.code, re.DOTALL)), 1)
            features['space_after_comma'] = len(re.findall(r',\s', self.code)) / max(self.code.count(','), 1)
            features['space_before_brace'] = len(re.findall(r'\s\{', self.code)) / max(self.code.count('{'), 1)
            features['single_quote_ratio'] = sq / max(sq + dq, 1)
            features['code_consistency'] = 1 - (abs(sq - dq) / max(sq + dq, 1))
            features['semicolon_ratio'] = len(re.findall(r';(?![)}\]])', self.code)) / self.total_lines
        except Exception:
            pass
        return features


# ==============================================================================
# 2. 语法特征提取器 (Syntactic)
# ==============================================================================
class SyntacticFeatureExtractor:
    def __init__(self, code: str, lines: List[str], total_lines: int):
        self.code = code
        self.lines = lines
        self.total_lines = total_lines
        self.length_chars = max(len(self.code), 1)
        self.ast_tree = None
        if JAVALANG_AVAILABLE:
            try:
                # 兼容片段：如果没有类声明，包一层壳
                if "class " not in self.code and "interface " not in self.code:
                    parsed_code = f"public class DummyClass {{\n{self.code}\n}}"
                else:
                    parsed_code = self.code
                self.ast_tree = javalang.parse.parse(parsed_code)
            except Exception:
                self.ast_tree = None

    def get_feature_names(self) -> List[str]:
        names = [
            # Structural (15)
            'method_density', 'control_flow_density', 'if_else_density', 'loop_density',
            'max_nesting_depth', 'avg_nesting_depth', 'semicolon_density_struct', 'statement_density',
            'brace_balance', 'code_compression_ratio', 'class_density', 'interface_density', 'enum_density',
            # Complexity (6)
            'cyclomatic_complexity', 'halstead_volume', 'halstead_difficulty', 'halstead_effort',
            'code_entropy', 'operator_operand_ratio',
            # AST Static (4)
            'ast_node_count', 'ast_max_depth', 'ast_avg_depth', 'ast_node_diversity'
        ]
        # AST Dynamic (10)
        ast_nodes = ['MethodDeclaration', 'ClassDeclaration', 'VariableDeclarator', 'IfStatement',
                     'ForStatement', 'WhileStatement', 'ReturnStatement', 'MethodInvocation',
                     'Assignment', 'BinaryOperation']
        names.extend([f'ast_{n}_ratio' for n in ast_nodes])
        return names

    def extract(self) -> Dict[str, float]:
        features = {k: 0.0 for k in self.get_feature_names()}
        try:
            total_methods = len(re.findall(r'\b(?:public|private|protected)?\s*\w+\s+\w+\s*\(', self.code))
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

            features['method_density'] = total_methods / self.total_lines
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
            features['interface_density'] = interface_defs / self.total_lines
            features['enum_density'] = enum_defs / self.total_lines

            # Complexity
            cyclomatic = len(re.findall(r'\b(if|else if|for|while|case|catch)\b', self.code)) + 1
            operators = re.findall(r'[+\-*/%=!&|^~<>?:]|\+\+|\-\-|&&|\|\||==|!=|<<|>>', self.code)
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

            # Javalang AST Features
            if self.ast_tree:
                node_counts = defaultdict(int)
                depths = []

                def traverse(node, depth=0):
                    if not node: return
                    node_type = type(node).__name__
                    node_counts[node_type] += 1
                    depths.append(depth)
                    if hasattr(node, 'children'):
                        for child in node.children:
                            if isinstance(child, list):
                                for item in child:
                                    if isinstance(item, javalang.tree.Node): traverse(item, depth + 1)
                            elif isinstance(child, javalang.tree.Node):
                                traverse(child, depth + 1)

                traverse(self.ast_tree)
                if depths:
                    features['ast_node_count'] = sum(node_counts.values())
                    features['ast_max_depth'] = max(depths)
                    features['ast_avg_depth'] = statistics.mean(depths)
                    features['ast_node_diversity'] = len(node_counts) / sum(node_counts.values())

                    ast_nodes = ['MethodDeclaration', 'ClassDeclaration', 'VariableDeclarator', 'IfStatement',
                                 'ForStatement', 'WhileStatement', 'ReturnStatement', 'MethodInvocation',
                                 'Assignment', 'BinaryOperation']
                    for n in ast_nodes:
                        features[f'ast_{n}_ratio'] = node_counts.get(n, 0) / sum(node_counts.values())
        except Exception:
            pass
        return features


# ==============================================================================
# 3. Java 专属特性 (JavaSpecific)
# ==============================================================================
class JavaSpecificFeatureExtractor:
    def __init__(self, code: str, lines: List[str], total_lines: int):
        self.code = code
        self.lines = lines
        self.total_lines = total_lines
        self.length_chars = max(len(self.code), 1)

    def get_feature_names(self) -> List[str]:
        return [
            # Modern Java (6)
            'lambda_density', 'stream_density', 'annotation_density',
            'generics_density', 'var_keyword_density', 'method_reference_density',
            # CP Idioms (6)
            'cp_scanner_density', 'cp_bufferedreader_density', 'cp_printwriter_density',
            'cp_math_density', 'cp_arrays_density', 'cp_collections_density'
        ]

    def extract(self) -> Dict[str, float]:
        features = {k: 0.0 for k in self.get_feature_names()}
        try:
            lc = self.length_chars
            tl = self.total_lines

            # Modern Features
            features['lambda_density'] = len(re.findall(r'->', self.code)) / tl
            features['stream_density'] = len(re.findall(r'\.stream\(\)', self.code)) / tl
            features['annotation_density'] = len(re.findall(r'@\w+', self.code)) / tl
            features['generics_density'] = len(re.findall(r'<[A-Z][a-zA-Z0-9]*>', self.code)) / tl
            features['var_keyword_density'] = len(re.findall(r'\bvar\s+', self.code)) / tl
            features['method_reference_density'] = len(re.findall(r'::', self.code)) / tl

            # CP Idioms
            features['cp_scanner_density'] = len(re.findall(r'\bnew\s+Scanner\b', self.code)) / lc
            features['cp_bufferedreader_density'] = len(re.findall(r'\bnew\s+BufferedReader\b', self.code)) / lc
            features['cp_printwriter_density'] = len(re.findall(r'\bnew\s+PrintWriter\b', self.code)) / lc
            features['cp_math_density'] = len(re.findall(r'\bMath\.(min|max|abs|pow|sqrt)\b', self.code)) / lc
            features['cp_arrays_density'] = len(re.findall(r'\bArrays\.(sort|fill|binarySearch)\b', self.code)) / lc
            features['cp_collections_density'] = len(re.findall(r'\bCollections\.(sort|reverse)\b', self.code)) / lc

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
            # AI Patterns
            'print_usage', 'simple_method_names', 'generic_comments', 'magic_numbers',
            'long_lines', 'repetitive_patterns',
            # Micro Habits
            'micro_missing_space_kw_ratio', 'micro_crowded_operator_density',
            'micro_print_debug_density', 'micro_typo_density', 'micro_human_atypical_density', 'micro_informal_tag_density',
            # Expert Errors
            'err_empty_catch_ratio', 'err_perfect_indent_with_crowded_ops', 'err_consecutive_calls_density'
        ]

    def extract(self) -> Dict[str, float]:
        features = {k: 0.0 for k in self.get_feature_names()}
        try:
            lc = self.length_chars
            tl = self.total_lines

            # AI Patterns
            features['print_usage'] = len(re.findall(r'System\.out\.print(?:ln)?', self.code)) / tl
            features['simple_method_names'] = len(
                re.findall(r'(?:public|private)\s+\w+\s+(test|demo|example|foo|bar)\s*\(', self.code, re.I)) / tl
            features['generic_comments'] = len(re.findall(r'//\s*(TODO|FIXME|NOTE|HACK)', self.code)) / tl
            features['magic_numbers'] = len(re.findall(r'\b\d{3,}\b', self.code)) / tl
            features['long_lines'] = sum(1 for line in self.lines if len(line) > 80) / tl
            features['repetitive_patterns'] = sum(
                1 for count in Counter([line.strip() for line in self.lines if line.strip()]).values() if
                count > 2) / tl

            # Micro Habits
            missing_space_kw = len(re.findall(r'\b(if|for|while|switch|catch)\(', self.code))
            has_space_kw = len(re.findall(r'\b(if|for|while|switch|catch)\s+\(', self.code))
            features['micro_missing_space_kw_ratio'] = missing_space_kw / (missing_space_kw + has_space_kw + 1e-10)

            crowded_ops = len(re.findall(r'[a-zA-Z0-9_](==|!=|<=|>=|\+=|-=|\*=|/=|=(?!=)|<|>)[a-zA-Z0-9_]', self.code))
            features['micro_crowded_operator_density'] = crowded_ops / lc
            features['micro_print_debug_density'] = len(
                re.findall(r'\bSystem\.out\.print(?:ln)?\s*\(\s*["\']\s*(?:test|debug|here|111|aaa|ok)\s*["\']\s*\)',
                           self.code_lower)) / lc

            common_typos = {'lenght', 'heigth', 'recieve', 'occured', 'sucess', 'retun'}
            informal_tags = {'todo', 'fixme', 'hack', 'wtf', 'shit', 'resources', 'test'}
            human_atypical = {'aids', 'composed', 'development', 'materials', 'segments', 'utilities', 'content', 'samples' , 'blocks', 'via', 'third-party', 'borrowed'}
            features['micro_typo_density'] = sum(
                len(re.findall(r'\b' + typo + r'\b', self.code_lower)) for typo in common_typos) / lc
            features['micro_human_atypical_density'] = sum(
                len(re.findall(r'\b' + atypical + r'\b', self.code_lower)) for atypical in human_atypical) / lc
            features['micro_informal_tag_density'] = sum(
                len(re.findall(r'\b' + tag + r'\b', self.code_lower)) for tag in informal_tags) / lc


            # Expert Additions
            catches = len(re.findall(r'\bcatch\b', self.code)) + 1e-10
            empty_catches = len(re.findall(r'catch\s*\([^)]*\)\s*\{\s*\}', self.code))
            features['err_empty_catch_ratio'] = empty_catches / catches

            irregular_indent = sum(
                1 for l in self.lines if l.startswith(' ') and (len(l) - len(l.lstrip(' '))) % 4 != 0) / tl
            features['err_perfect_indent_with_crowded_ops'] = 1.0 if irregular_indent < 0.05 and (
                        crowded_ops / lc) > 0.01 else 0.0


            stripped_lines = [l.strip() for l in self.lines if l.strip()]
            consecutive_calls = 0
            for i in range(len(stripped_lines) - 2):
                m1 = re.match(r'^([a-zA-Z_][\w\.]*)\s*\(', stripped_lines[i])
                m2 = re.match(r'^([a-zA-Z_][\w\.]*)\s*\(', stripped_lines[i + 1])
                m3 = re.match(r'^([a-zA-Z_][\w\.]*)\s*\(', stripped_lines[i + 2])
                if m1 and m2 and m3 and (m1.group(1) == m2.group(1) == m3.group(1)):
                    consecutive_calls += 1
            features['err_consecutive_calls_density'] = consecutive_calls / (len(stripped_lines) + 1e-10)



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
# 批量处理包装器 (重写版：支持 CodeBERTa 矩阵合并)
# =================================================================================
class JavaBatchExtractorWrapper:
    def __init__(self, config=None):
        self.config = config
        self.feature_names = []
        self._is_names_initialized = False
        self._codeberta_names_added = False

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        logger.info(f"开始提取 {len(codes)} 个 Java 文件的传统多维特征...")
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
            # 通过 config 控制是否提取
            # ==========================================
            if features_config.get('lexical', True):
                lexical = LexicalFeatureExtractor(code, lines, total_lines)
                feat_dict.update(lexical.extract())

            if features_config.get('syntactic', True):
                syntactic = SyntacticFeatureExtractor(code, lines, total_lines)
                feat_dict.update(syntactic.extract())

            if features_config.get('java_specific', True):
                java_specific = JavaSpecificFeatureExtractor(code, lines, total_lines)
                feat_dict.update(java_specific.extract())

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
# class JavaBatchExtractorWrapper:
#     def __init__(self, config=None):
#         self.feature_names = []
#         self._is_names_initialized = False
#
#     def get_feature_names(self) -> List[str]:
#         return self.feature_names
#
#     def extract_batch(self, codes: List[str]) -> np.ndarray:
#         logger.info(f"开始提取 {len(codes)} 个 Java 文件的全量多维特征 (锁定 123 维无损版)...")
#
#         features_list = []
#         for idx, code in enumerate(codes):
#             if idx % 500 == 0 and idx > 0:
#                 logger.info(f"  已提取: {idx} / {len(codes)}")
#
#             lines = code.split('\n')
#             total_lines = max(len(lines), 1)
#
#             # 实例化四大提取器
#             lexical = LexicalFeatureExtractor(code, lines, total_lines)
#             syntactic = SyntacticFeatureExtractor(code, lines, total_lines)
#             java_specific = JavaSpecificFeatureExtractor(code, lines, total_lines)
#             human_error = HumanErrorFeatureExtractor(code, lines, total_lines)
#
#             # 融合字典
#             feat_dict = {}
#             feat_dict.update(lexical.extract())
#             feat_dict.update(syntactic.extract())
#             feat_dict.update(java_specific.extract())
#             feat_dict.update(human_error.extract())
#
#             if not self._is_names_initialized:
#                 # 记录所有的 123 个预定义键名，彻底锁死数据维度
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
