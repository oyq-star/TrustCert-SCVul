"""Data ingestion: load raw datasets into canonical schema."""
import os
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path


def hash_source(source_text: str) -> str:
    return hashlib.sha256(source_text.encode('utf-8', errors='replace')).hexdigest()


VULN_LABEL_MAP = {
    # BCCC labels -> canonical
    'reentrancy': 'reentrancy',
    'Re-entrancy': 'reentrancy',
    'RE': 'reentrancy',
    'integer_overflow': 'arithmetic',
    'integer_underflow': 'arithmetic',
    'arithmetic': 'arithmetic',
    'IO': 'arithmetic',
    'overflow_underflow': 'arithmetic',
    'dos': 'dos',
    'denial_of_service': 'dos',
    'DOS': 'dos',
    'unchecked_low_level_calls': 'dos',
    'timestamp': 'timestamp',
    'time_manipulation': 'timestamp',
    'timestamp_dependence': 'timestamp',
    'TD': 'timestamp',
    'block_timestamp_dependency': 'timestamp',
    'TOD': 'timestamp',
}

TARGET_VULNS = ['reentrancy', 'arithmetic', 'dos', 'timestamp']


def load_smartbugs_curated(data_dir: str) -> pd.DataFrame:
    """Load SmartBugs-Curated dataset from local directory."""
    records = []
    base = Path(data_dir)

    # Check for dataset subdirectory
    if (base / 'dataset').exists():
        base = base / 'dataset'

    if not base.exists():
        print(f"[WARN] SmartBugs-Curated not found at {data_dir}")
        return pd.DataFrame()

    for vuln_dir in base.iterdir():
        if not vuln_dir.is_dir():
            continue
        vuln_type_raw = vuln_dir.name
        vuln_type = VULN_LABEL_MAP.get(vuln_type_raw, vuln_type_raw.lower())
        if vuln_type not in TARGET_VULNS:
            continue

        for sol_file in vuln_dir.rglob('*.sol'):
            try:
                source = sol_file.read_text(encoding='utf-8', errors='replace')
            except Exception:
                continue
            records.append({
                'contract_id': sol_file.stem,
                'source_path': str(sol_file),
                'source_text': source,
                'source_hash': hash_source(source),
                'vulnerability_type': vuln_type,
                'label': 1,
                'dataset': 'smartbugs_curated',
            })

    df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(df)} samples from SmartBugs-Curated")
    return df


def load_bccc_dataset(data_dir: str) -> pd.DataFrame:
    """Load BCCC-SCsVuls-2024 dataset."""
    records = []
    base = Path(data_dir)

    if not base.exists():
        print(f"[WARN] BCCC dataset not found at {data_dir}")
        return pd.DataFrame()

    # Try CSV/JSON format first
    for csv_file in base.rglob('*.csv'):
        try:
            df_raw = pd.read_csv(csv_file)
            for _, row in df_raw.iterrows():
                vuln_raw = str(row.get('vulnerability_type', row.get('label', '')))
                vuln_type = VULN_LABEL_MAP.get(vuln_raw, vuln_raw.lower())
                if vuln_type not in TARGET_VULNS:
                    continue
                source = str(row.get('source_code', row.get('code', '')))
                records.append({
                    'contract_id': str(row.get('contract_name', row.get('id', len(records)))),
                    'source_path': str(csv_file),
                    'source_text': source,
                    'source_hash': hash_source(source),
                    'vulnerability_type': vuln_type,
                    'label': int(row.get('vulnerable', row.get('label', 1))),
                    'dataset': 'bccc',
                })
        except Exception as e:
            print(f"[WARN] Failed to read {csv_file}: {e}")

    # Try directory structure (like SmartBugs)
    if not records:
        for vuln_dir in base.iterdir():
            if not vuln_dir.is_dir():
                continue
            vuln_type_raw = vuln_dir.name
            vuln_type = VULN_LABEL_MAP.get(vuln_type_raw, vuln_type_raw.lower())
            if vuln_type not in TARGET_VULNS:
                continue
            for sol_file in vuln_dir.rglob('*.sol'):
                try:
                    source = sol_file.read_text(encoding='utf-8', errors='replace')
                except Exception:
                    continue
                records.append({
                    'contract_id': sol_file.stem,
                    'source_path': str(sol_file),
                    'source_text': source,
                    'source_hash': hash_source(source),
                    'vulnerability_type': vuln_type,
                    'label': 1,
                    'dataset': 'bccc',
                })

    df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(df)} samples from BCCC")
    return df


def generate_synthetic_dataset(n_contracts=2000, seed=42) -> pd.DataFrame:
    """Generate synthetic dataset for prototyping when real data unavailable."""
    rng = np.random.RandomState(seed)
    records = []

    for i in range(n_contracts):
        vuln_type = rng.choice(TARGET_VULNS, p=[0.35, 0.25, 0.20, 0.20])
        is_vulnerable = rng.random() < 0.4  # 40% positive rate

        # Simulate source code characteristics
        loc = rng.randint(20, 500)
        source_text = f"// Synthetic contract {i}\npragma solidity ^0.8.0;\ncontract C{i} {{\n"
        source_text += f"  // LOC: {loc}, vuln: {vuln_type if is_vulnerable else 'none'}\n}}"

        records.append({
            'contract_id': f'synthetic_{i}',
            'source_path': f'synthetic/contract_{i}.sol',
            'source_text': source_text,
            'source_hash': hash_source(source_text),
            'vulnerability_type': vuln_type,
            'label': int(is_vulnerable),
            'dataset': 'synthetic',
        })

    df = pd.DataFrame(records)
    print(f"[INFO] Generated {len(df)} synthetic samples")
    return df


def expand_with_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """For each vulnerability type, add negative samples from other categories.

    SmartBugs-Curated only contains vulnerable contracts organized by type.
    A contract vulnerable to 'reentrancy' can serve as a negative sample
    for 'arithmetic' detection, etc.
    """
    records = []
    unique_contracts = df.drop_duplicates(subset=['contract_id', 'source_hash'])

    for vuln_type in TARGET_VULNS:
        # Positive: contracts labeled with this vulnerability
        pos = df[df['vulnerability_type'] == vuln_type].copy()
        records.append(pos)

        # Negative: contracts from other vulnerability types
        other = unique_contracts[
            ~unique_contracts['contract_id'].isin(pos['contract_id'])
        ].copy()
        other['vulnerability_type'] = vuln_type
        other['label'] = 0
        records.append(other)

    result = pd.concat(records, ignore_index=True)
    print(f"[INFO] Expanded dataset with negatives: {len(df)} -> {len(result)} samples")
    return result


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact-duplicate contracts by source hash."""
    before = len(df)
    df = df.drop_duplicates(subset=['source_hash', 'vulnerability_type'], keep='first')
    after = len(df)
    if before > after:
        print(f"[INFO] Deduplicated: {before} -> {after} ({before - after} removed)")
    return df.reset_index(drop=True)


def create_splits(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    """Create leakage-safe train/val/cal/test splits grouped by source_hash."""
    rng = np.random.RandomState(seed)

    # Group by source_hash to prevent leakage
    unique_hashes = df['source_hash'].unique()
    rng.shuffle(unique_hashes)

    n = len(unique_hashes)
    train_end = int(n * 0.60)
    val_end = int(n * 0.75)
    cal_end = int(n * 0.85)

    hash_to_split = {}
    for h in unique_hashes[:train_end]:
        hash_to_split[h] = 'train'
    for h in unique_hashes[train_end:val_end]:
        hash_to_split[h] = 'val'
    for h in unique_hashes[val_end:cal_end]:
        hash_to_split[h] = 'cal'
    for h in unique_hashes[cal_end:]:
        hash_to_split[h] = 'test'

    df['split'] = df['source_hash'].map(hash_to_split)
    for split_name in ['train', 'val', 'cal', 'test']:
        count = (df['split'] == split_name).sum()
        print(f"  Split {split_name}: {count} samples")

    return df
