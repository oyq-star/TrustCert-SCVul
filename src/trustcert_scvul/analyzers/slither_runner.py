"""Run real Slither analysis on Solidity contracts."""
import os
import json
import subprocess
import tempfile
import csv
from pathlib import Path
from collections import defaultdict

DETECTOR_TO_VULN = {
    'reentrancy-eth': 'reentrancy',
    'reentrancy-no-eth': 'reentrancy',
    'reentrancy-benign': 'reentrancy',
    'reentrancy-unlimited-gas': 'reentrancy',
    'reentrancy-events': 'reentrancy',
    'controlled-delegatecall': 'reentrancy',
    'calls-loop': 'dos',
    'costly-loop': 'dos',
    'msg-value-loop': 'dos',
    'unchecked-lowlevel': 'dos',
    'unchecked-send': 'dos',
    'unchecked-transfer': 'dos',
    'locked-ether': 'dos',
    'controlled-array-length': 'dos',
    'arbitrary-send-eth': 'dos',
    'timestamp': 'timestamp',
    'weak-prng': 'timestamp',
    'block-timestamp': 'timestamp',
    'incorrect-equality': 'timestamp',
    'divide-before-multiply': 'arithmetic',
    'tautology': 'arithmetic',
}

IMPACT_SCORE = {'High': 3, 'Medium': 2, 'Low': 1, 'Informational': 0}
CONFIDENCE_SCORE = {'High': 3, 'Medium': 2, 'Low': 1}


def _load_version_map(curated_dir):
    """Load solc versions from versions.csv."""
    vmap = {}
    csv_path = Path(curated_dir).parent / 'versions.csv'
    if not csv_path.exists():
        csv_path = Path(curated_dir) / '..' / 'versions.csv'
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get('file', '').strip()
                ver = row.get('compiled version', '').strip()
                if fname and ver:
                    basename = Path(fname).name
                    vmap[basename] = ver
    return vmap


def _detect_solc_version(source_text):
    """Extract solc version from pragma."""
    import re
    m = re.search(r'pragma\s+solidity\s*[\^~>=<]*\s*(0\.\d+\.\d+)', source_text)
    if m:
        return m.group(1)
    return '0.4.25'


def _ensure_solc(version):
    """Install and select solc version."""
    try:
        subprocess.run(
            ['solc-select', 'install', version],
            capture_output=True, timeout=60
        )
        subprocess.run(
            ['solc-select', 'use', version],
            capture_output=True, timeout=10
        )
        return True
    except Exception:
        return False


def run_slither_on_file(sol_path, solc_version=None, timeout_sec=60):
    """Run slither on a single .sol file. Returns list of findings."""
    sol_path = str(sol_path)
    if solc_version:
        _ensure_solc(solc_version)

    out_file = sol_path + '.slither.json'
    try:
        result = subprocess.run(
            ['slither', sol_path, '--json', out_file],
            capture_output=True, text=True, timeout=timeout_sec
        )
    except subprocess.TimeoutExpired:
        return []
    except FileNotFoundError:
        return []

    findings = []
    if os.path.exists(out_file):
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for det in data.get('results', {}).get('detectors', []):
                check = det.get('check', '')
                impact = det.get('impact', 'Informational')
                confidence = det.get('confidence', 'Low')
                vuln_type = DETECTOR_TO_VULN.get(check)
                elements = det.get('elements', [])
                lines = set()
                for el in elements:
                    sl = el.get('source_mapping', {})
                    lines.update(range(
                        sl.get('starting_line', 0),
                        sl.get('ending_line', 0) + 1
                    ))
                findings.append({
                    'detector': check,
                    'mapped_vuln': vuln_type,
                    'impact': impact,
                    'impact_score': IMPACT_SCORE.get(impact, 0),
                    'confidence': confidence,
                    'confidence_score': CONFIDENCE_SCORE.get(confidence, 0),
                    'lines': sorted(lines),
                    'description': det.get('description', '')[:200],
                })
        except Exception:
            pass
        finally:
            try:
                os.remove(out_file)
            except OSError:
                pass
    return findings


def run_slither_batch(df, data_dir=None, version_map=None):
    """Run slither on all contracts in df. Returns dict: source_hash -> findings."""
    if version_map is None:
        version_map = {}
        if data_dir:
            version_map = _load_version_map(data_dir)

    unique = df.drop_duplicates(subset=['contract_id', 'source_hash'])
    results = {}
    n = len(unique)
    succeeded = 0
    failed = 0

    installed_versions = set()

    for idx, (_, row) in enumerate(unique.iterrows()):
        source_path = row.get('source_path', '')
        source_hash = row['source_hash']
        contract_id = row['contract_id']

        if idx % 20 == 0:
            print(f"  Slither progress: {idx}/{n} (ok={succeeded}, fail={failed})")

        sol_path = None
        if source_path and os.path.isfile(source_path):
            sol_path = source_path
        elif data_dir:
            candidates = list(Path(data_dir).rglob(f'{contract_id}.sol'))
            if candidates:
                sol_path = str(candidates[0])

        if sol_path is None:
            source_text = row.get('source_text', '')
            if not source_text:
                results[source_hash] = []
                failed += 1
                continue
            tmp = tempfile.NamedTemporaryFile(
                suffix='.sol', delete=False, mode='w', encoding='utf-8'
            )
            tmp.write(source_text)
            tmp.close()
            sol_path = tmp.name

        basename = Path(sol_path).name
        solc_ver = version_map.get(basename)
        if not solc_ver:
            source_text = row.get('source_text', '')
            if not source_text and os.path.isfile(sol_path):
                source_text = Path(sol_path).read_text(encoding='utf-8', errors='replace')
            solc_ver = _detect_solc_version(source_text)

        if solc_ver not in installed_versions:
            _ensure_solc(solc_ver)
            installed_versions.add(solc_ver)

        findings = run_slither_on_file(sol_path, solc_version=solc_ver, timeout_sec=90)
        results[source_hash] = findings

        if findings:
            succeeded += 1
        else:
            failed += 1

    print(f"  Slither complete: {succeeded} succeeded, {failed} failed out of {n}")
    return results


def findings_to_features(findings_dict, df):
    """Convert slither findings dict into per-contract feature columns.

    Returns DataFrame with columns:
        slither_<vuln>_count, slither_<vuln>_max_impact, slither_<vuln>_max_confidence,
        slither_total_findings, slither_high_count, slither_relevant_flag
    """
    import pandas as pd
    import numpy as np

    TARGET_VULNS = ['reentrancy', 'arithmetic', 'dos', 'timestamp']
    rows = []
    unique = df.drop_duplicates(subset=['contract_id', 'source_hash'])

    for _, row in unique.iterrows():
        source_hash = row['source_hash']
        findings = findings_dict.get(source_hash, [])
        feat = {'contract_id': row['contract_id'], 'source_hash': source_hash}

        feat['slither_total_findings'] = len(findings)
        feat['slither_high_count'] = sum(
            1 for f in findings if f['impact'] == 'High'
        )

        for vt in TARGET_VULNS:
            vt_findings = [f for f in findings if f.get('mapped_vuln') == vt]
            feat[f'slither_{vt}_count'] = len(vt_findings)
            feat[f'slither_{vt}_max_impact'] = (
                max((f['impact_score'] for f in vt_findings), default=0)
            )
            feat[f'slither_{vt}_max_confidence'] = (
                max((f['confidence_score'] for f in vt_findings), default=0)
            )
            all_lines = set()
            for f in vt_findings:
                all_lines.update(f.get('lines', []))
            feat[f'slither_{vt}_line_count'] = len(all_lines)

        rows.append(feat)

    return pd.DataFrame(rows)
