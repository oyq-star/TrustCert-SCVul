"""Load and label SmartBugs-Wild contracts using Slither silver labels."""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from trustcert_scvul.data.ingest import hash_source, TARGET_VULNS


def sample_wild_contracts(wild_dir, n_sample=200, seed=42):
    """Sample n contracts from SmartBugs-Wild."""
    rng = np.random.RandomState(seed)
    contracts_dir = Path(wild_dir) / 'contracts'

    if not contracts_dir.exists():
        print(f"[WARN] SmartBugs-Wild not found at {wild_dir}")
        return pd.DataFrame()

    all_sols = sorted([f for f in os.listdir(contracts_dir) if f.endswith('.sol')])
    if len(all_sols) == 0:
        return pd.DataFrame()

    chosen = rng.choice(all_sols, size=min(n_sample, len(all_sols)), replace=False)

    records = []
    for fname in chosen:
        fpath = contracts_dir / fname
        try:
            source = fpath.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        if len(source) < 50:
            continue
        records.append({
            'contract_id': fpath.stem,
            'source_path': str(fpath),
            'source_text': source,
            'source_hash': hash_source(source),
            'dataset': 'smartbugs_wild',
        })

    df = pd.DataFrame(records)
    print(f"[INFO] Sampled {len(df)} contracts from SmartBugs-Wild")
    return df


def label_with_slither(wild_df, slither_findings_dict):
    """Create silver-labeled dataset from Slither findings.

    For each contract, for each target vuln type:
    - If Slither found a HIGH/MEDIUM impact finding: label=1
    - Otherwise: label=0
    """
    records = []

    for _, row in wild_df.iterrows():
        source_hash = row['source_hash']
        findings = slither_findings_dict.get(source_hash, [])

        for vt in TARGET_VULNS:
            vt_findings = [
                f for f in findings
                if f.get('mapped_vuln') == vt and f.get('impact_score', 0) >= 2
            ]
            label = 1 if len(vt_findings) > 0 else 0
            records.append({
                'contract_id': row['contract_id'],
                'source_path': row['source_path'],
                'source_text': row['source_text'],
                'source_hash': source_hash,
                'vulnerability_type': vt,
                'label': label,
                'dataset': 'smartbugs_wild_silver',
            })

    result = pd.DataFrame(records)
    for vt in TARGET_VULNS:
        vt_df = result[result['vulnerability_type'] == vt]
        pos = (vt_df['label'] == 1).sum()
        print(f"  Wild silver labels - {vt}: {pos} positive, {len(vt_df) - pos} negative")
    return result
