"""Structural feature extraction from Solidity source code."""
import re
import numpy as np
import pandas as pd


def extract_structural_features(source_text: str) -> dict:
    """Extract structural features from Solidity source code."""
    lines = source_text.split('\n')
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('//')]

    loc_total = len(lines)
    loc_code = len(code_lines)
    code_text = source_text

    features = {
        'loc_total': loc_total,
        'loc_code': loc_code,

        # Function counts
        'function_count_total': len(re.findall(r'\bfunction\s+\w+', code_text)),
        'function_count_external_public': len(re.findall(
            r'\bfunction\s+\w+[^{]*\b(public|external)\b', code_text)),
        'function_count_private_internal': len(re.findall(
            r'\bfunction\s+\w+[^{]*\b(private|internal)\b', code_text)),

        # State variables
        'state_var_count': len(re.findall(
            r'^\s*(uint|int|bool|address|string|bytes|mapping|struct)\w*\s', code_text, re.MULTILINE)),
        'mapping_count': len(re.findall(r'\bmapping\s*\(', code_text)),
        'array_state_var_count': len(re.findall(r'\[\]\s+(public|private|internal)?\s*\w+\s*;', code_text)),

        # Modifiers and events
        'modifier_count': len(re.findall(r'\bmodifier\s+\w+', code_text)),
        'event_count': len(re.findall(r'\bevent\s+\w+', code_text)),

        # Inheritance
        'inheritance_depth_max': len(re.findall(r'\bis\s+\w+', code_text)),
        'contract_count': len(re.findall(r'\bcontract\s+\w+', code_text)),

        # Ether handling
        'fallback_exists': int(bool(re.search(r'\bfallback\s*\(', code_text))),
        'receive_exists': int(bool(re.search(r'\breceive\s*\(', code_text))),
        'payable_function_count': len(re.findall(r'\bpayable\b', code_text)),

        # External calls (reentrancy signals)
        'external_call_count': (
            len(re.findall(r'\.call\s*[({]', code_text)) +
            len(re.findall(r'\.send\s*\(', code_text)) +
            len(re.findall(r'\.transfer\s*\(', code_text))
        ),
        'low_level_call_count': len(re.findall(r'\.(call|delegatecall|callcode|staticcall)\s*[({]', code_text)),
        'delegatecall_count': len(re.findall(r'\.delegatecall\s*[({]', code_text)),
        'call_value_count': len(re.findall(r'\.call\s*\{.*value', code_text)),

        # State writes after external calls (reentrancy pattern)
        'state_write_count': len(re.findall(r'[a-zA-Z_]\w*\s*(=|\+=|-=|\*=|/=)\s*', code_text)),

        # Reentrancy guard
        'reentrancy_guard_present': int(bool(
            re.search(r'nonReentrant|ReentrancyGuard|_locked|mutex', code_text))),

        # Arithmetic patterns
        'arithmetic_op_count': len(re.findall(r'[\+\-\*/%]', code_text)),
        'unchecked_block_count': len(re.findall(r'\bunchecked\s*\{', code_text)),
        'safemath_usage': int(bool(re.search(r'SafeMath|using\s+SafeMath', code_text))),

        # Control flow
        'loop_count': len(re.findall(r'\b(for|while|do)\s*[\({]', code_text)),
        'if_count': len(re.findall(r'\bif\s*\(', code_text)),
        'require_count': len(re.findall(r'\brequire\s*\(', code_text)),
        'assert_count': len(re.findall(r'\bassert\s*\(', code_text)),
        'revert_count': len(re.findall(r'\brevert\s*[\(;]', code_text)),

        # Inline assembly
        'inline_assembly_count': len(re.findall(r'\bassembly\s*\{', code_text)),

        # Timestamp patterns
        'timestamp_read_count': len(re.findall(r'\bblock\.timestamp\b|\bnow\b', code_text)),
        'block_number_read_count': len(re.findall(r'\bblock\.number\b', code_text)),

        # Compiler version
        'compiler_version_raw': '',
        'compiler_age_proxy': 0,
    }

    # Extract compiler version
    pragma_match = re.search(r'pragma\s+solidity\s+[\^>=<]*\s*(\d+\.\d+\.\d+)', code_text)
    if pragma_match:
        version_str = pragma_match.group(1)
        features['compiler_version_raw'] = version_str
        parts = version_str.split('.')
        major, minor = int(parts[0]), int(parts[1])
        # Proxy for age: older versions = higher risk
        features['compiler_age_proxy'] = max(0, 8 - minor) if major == 0 else 0
        features['solc_legacy_lt_0_8'] = int(major == 0 and minor < 8)
    else:
        features['solc_legacy_lt_0_8'] = 0

    # Derived densities
    loc_safe = max(loc_code, 1)
    features['arithmetic_op_density'] = features['arithmetic_op_count'] / loc_safe
    features['external_call_density'] = features['external_call_count'] / loc_safe
    features['require_density'] = features['require_count'] / loc_safe
    features['loop_density'] = features['loop_count'] / loc_safe

    # Cyclomatic complexity proxy
    features['cyclomatic_complexity_proxy'] = (
        features['if_count'] + features['loop_count'] +
        features['require_count'] + features['assert_count'] + 1
    )

    # State-write-after-call pattern detection
    features['state_write_after_call'] = _detect_write_after_call(code_text)

    return features


def _detect_write_after_call(source: str) -> int:
    """Detect state-write-after-external-call pattern (reentrancy signal)."""
    # Simple heuristic: look for functions with call followed by assignment
    functions = re.split(r'\bfunction\s+', source)
    count = 0
    for func in functions[1:]:  # skip preamble
        call_pos = -1
        for m in re.finditer(r'\.(call|send|transfer)\s*[\({]', func):
            call_pos = m.start()
        if call_pos >= 0:
            # Check for state write after the call
            after_call = func[call_pos:]
            if re.search(r'[a-zA-Z_]\w*\s*=\s*', after_call):
                count += 1
    return count


def extract_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structural features for all contracts in batch."""
    feature_records = []
    for idx, row in df.iterrows():
        try:
            feats = extract_structural_features(row['source_text'])
            feats['contract_id'] = row['contract_id']
            feats['source_hash'] = row['source_hash']
            feature_records.append(feats)
        except Exception as e:
            print(f"[WARN] Feature extraction failed for {row['contract_id']}: {e}")
            feature_records.append({
                'contract_id': row['contract_id'],
                'source_hash': row['source_hash'],
            })

    feat_df = pd.DataFrame(feature_records)
    # Drop non-numeric columns for modeling
    non_feature_cols = ['contract_id', 'source_hash', 'compiler_version_raw']
    feature_cols = [c for c in feat_df.columns if c not in non_feature_cols]
    feat_df[feature_cols] = feat_df[feature_cols].fillna(0)

    print(f"[INFO] Extracted {len(feature_cols)} structural features for {len(feat_df)} contracts")
    return feat_df
