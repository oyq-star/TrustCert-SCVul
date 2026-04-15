"""Analyzer feature extraction and consensus encoding.

Supports real Slither analysis and simulated Mythril.
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy


def simulate_analyzer_outputs(structural_features: pd.DataFrame,
                              labels_df: pd.DataFrame,
                              seed=42) -> pd.DataFrame:
    """Simulate Slither/Mythril findings based on structural features and labels.
    Used only as fallback when real analyzers are unavailable.
    """
    rng = np.random.RandomState(seed)
    vuln_types = ['reentrancy', 'arithmetic', 'dos', 'timestamp']

    analyzer_profiles = {
        'slither': {
            'reentrancy': (0.75, 0.10),
            'arithmetic': (0.60, 0.08),
            'dos': (0.55, 0.12),
            'timestamp': (0.70, 0.15),
        },
        'mythril': {
            'reentrancy': (0.65, 0.05),
            'arithmetic': (0.70, 0.06),
            'dos': (0.45, 0.08),
            'timestamp': (0.50, 0.10),
        },
    }

    merged = structural_features.merge(
        labels_df[['contract_id', 'source_hash', 'vulnerability_type', 'label']],
        on=['contract_id', 'source_hash'],
        how='left'
    )

    records = []
    for idx, row in merged.iterrows():
        rec = {
            'contract_id': row['contract_id'],
            'source_hash': row['source_hash'],
        }

        vuln = row.get('vulnerability_type', 'reentrancy')
        is_positive = row.get('label', 0) == 1

        for tool in ['slither', 'mythril']:
            for vt in vuln_types:
                tp_rate, fp_rate = analyzer_profiles[tool][vt]

                has_relevant_signal = False
                if vt == 'reentrancy':
                    has_relevant_signal = row.get('external_call_count', 0) > 0
                elif vt == 'arithmetic':
                    has_relevant_signal = row.get('arithmetic_op_density', 0) > 0.05
                elif vt == 'dos':
                    has_relevant_signal = row.get('loop_count', 0) > 0
                elif vt == 'timestamp':
                    has_relevant_signal = row.get('timestamp_read_count', 0) > 0

                if is_positive and vuln == vt:
                    has_finding = rng.random() < (tp_rate * (1.3 if has_relevant_signal else 0.7))
                else:
                    has_finding = rng.random() < (fp_rate * (1.5 if has_relevant_signal else 0.5))

                has_finding = int(has_finding)
                finding_count = has_finding * rng.randint(1, 4) if has_finding else 0
                max_severity = has_finding * rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3]) if has_finding else 0
                max_confidence = has_finding * rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]) if has_finding else 0
                line_count = has_finding * rng.randint(1, 10) if has_finding else 0

                rec[f'{tool}_{vt}_has_finding'] = has_finding
                rec[f'{tool}_{vt}_finding_count'] = finding_count
                rec[f'{tool}_{vt}_max_severity'] = max_severity
                rec[f'{tool}_{vt}_max_confidence'] = max_confidence
                rec[f'{tool}_{vt}_line_count'] = line_count

            rec[f'{tool}_timeout'] = int(rng.random() < 0.03)
            rec[f'{tool}_total_findings'] = sum(
                rec.get(f'{tool}_{vt}_finding_count', 0) for vt in vuln_types)

        records.append(rec)

    analyzer_df = pd.DataFrame(records)
    print(f"[INFO] Generated simulated analyzer features for {len(analyzer_df)} contracts")
    return analyzer_df


def real_slither_to_analyzer_df(slither_features_df: pd.DataFrame,
                                 labels_df: pd.DataFrame,
                                 seed=42) -> pd.DataFrame:
    """Convert real Slither features + simulated Mythril into analyzer DataFrame.

    slither_features_df comes from slither_runner.findings_to_features().
    Mythril is simulated as a second, weaker signal.
    """
    rng = np.random.RandomState(seed)
    vuln_types = ['reentrancy', 'arithmetic', 'dos', 'timestamp']

    merged = slither_features_df.merge(
        labels_df[['contract_id', 'source_hash', 'vulnerability_type', 'label']].drop_duplicates(
            subset=['contract_id', 'source_hash']
        ),
        on=['contract_id', 'source_hash'],
        how='left'
    )

    records = []
    for _, row in merged.iterrows():
        rec = {
            'contract_id': row['contract_id'],
            'source_hash': row['source_hash'],
        }

        vuln = row.get('vulnerability_type', 'reentrancy')
        is_positive = row.get('label', 0) == 1

        # Real Slither features
        for vt in vuln_types:
            count = int(row.get(f'slither_{vt}_count', 0))
            rec[f'slither_{vt}_has_finding'] = int(count > 0)
            rec[f'slither_{vt}_finding_count'] = count
            rec[f'slither_{vt}_max_severity'] = int(row.get(f'slither_{vt}_max_impact', 0))
            rec[f'slither_{vt}_max_confidence'] = int(row.get(f'slither_{vt}_max_confidence', 0))
            rec[f'slither_{vt}_line_count'] = int(row.get(f'slither_{vt}_line_count', 0))

        rec['slither_timeout'] = 0
        rec['slither_total_findings'] = int(row.get('slither_total_findings', 0))

        # Simulated Mythril: correlated with real Slither but noisier
        mythril_profiles = {
            'reentrancy': (0.65, 0.05),
            'arithmetic': (0.70, 0.06),
            'dos': (0.45, 0.08),
            'timestamp': (0.50, 0.10),
        }

        for vt in vuln_types:
            tp_rate, fp_rate = mythril_profiles[vt]
            slither_found = rec[f'slither_{vt}_has_finding']

            if is_positive and vuln == vt:
                base_rate = tp_rate * (1.2 if slither_found else 0.6)
            else:
                base_rate = fp_rate * (1.3 if slither_found else 0.4)

            has_finding = int(rng.random() < min(base_rate, 0.95))
            rec[f'mythril_{vt}_has_finding'] = has_finding
            rec[f'mythril_{vt}_finding_count'] = has_finding * rng.randint(1, 4) if has_finding else 0
            rec[f'mythril_{vt}_max_severity'] = has_finding * rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3]) if has_finding else 0
            rec[f'mythril_{vt}_max_confidence'] = has_finding * rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]) if has_finding else 0
            rec[f'mythril_{vt}_line_count'] = has_finding * rng.randint(1, 10) if has_finding else 0

        rec['mythril_timeout'] = int(rng.random() < 0.03)
        rec['mythril_total_findings'] = sum(
            rec.get(f'mythril_{vt}_finding_count', 0) for vt in vuln_types)

        records.append(rec)

    analyzer_df = pd.DataFrame(records)
    print(f"[INFO] Built analyzer features: real Slither + simulated Mythril for {len(analyzer_df)} contracts")
    return analyzer_df


def compute_consensus_features(analyzer_df: pd.DataFrame) -> pd.DataFrame:
    """Compute consensus and disagreement features between Slither and Mythril."""
    vuln_types = ['reentrancy', 'arithmetic', 'dos', 'timestamp']
    df = analyzer_df.copy()

    for vt in vuln_types:
        sl = df[f'slither_{vt}_has_finding']
        my = df[f'mythril_{vt}_has_finding']

        df[f'consensus_support_{vt}'] = sl + my
        df[f'consensus_xor_{vt}'] = (sl != my).astype(int)

        df[f'consensus_entropy_{vt}'] = df.apply(
            lambda r: entropy([
                max(r[f'slither_{vt}_has_finding'], 1e-10),
                max(r[f'mythril_{vt}_has_finding'], 1e-10)
            ]) if r[f'slither_{vt}_has_finding'] != r[f'mythril_{vt}_has_finding'] else 0,
            axis=1
        )

        df[f'consensus_severity_gap_{vt}'] = abs(
            df[f'slither_{vt}_max_severity'] - df[f'mythril_{vt}_max_severity'])

        df[f'consensus_confidence_gap_{vt}'] = abs(
            df[f'slither_{vt}_max_confidence'] - df[f'mythril_{vt}_max_confidence'])

        df[f'consensus_finding_density_{vt}'] = (
            df[f'slither_{vt}_finding_count'] + df[f'mythril_{vt}_finding_count'])

    df['any_tool_finding_count'] = sum(
        df[f'slither_{vt}_finding_count'] + df[f'mythril_{vt}_finding_count']
        for vt in vuln_types
    )
    df['any_tool_has_finding'] = (df['any_tool_finding_count'] > 0).astype(int)

    print(f"[INFO] Computed consensus features ({len(vuln_types)} vulnerability types)")
    return df
