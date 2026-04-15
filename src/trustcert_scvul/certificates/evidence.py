"""Evidence certificate generation and Merkle tree verification."""
import hashlib
import json
import time
import numpy as np


def generate_certificate(contract_id: str, source_hash: str, vuln_type: str,
                         prediction: int, confidence: float,
                         top_features: list, model_version: str = "v1.0") -> dict:
    """Generate an evidence certificate for an accepted prediction.

    Args:
        contract_id: Unique contract identifier
        source_hash: SHA-256 of source code
        vuln_type: Vulnerability type being assessed
        prediction: 0 (safe) or 1 (vulnerable)
        confidence: Model confidence score
        top_features: List of (feature_name, contribution) tuples
        model_version: Version string for the model

    Returns:
        Certificate dict with all fields
    """
    cert = {
        'contract_id': contract_id,
        'source_hash': source_hash,
        'vulnerability_type': vuln_type,
        'prediction': 'vulnerable' if prediction == 1 else 'safe',
        'confidence': float(confidence),
        'model_version': model_version,
        'timestamp': time.time(),
        'evidence': {
            'top_features': [
                {'name': name, 'contribution': float(contrib)}
                for name, contrib in top_features[:5]
            ],
        },
    }

    # Compute certificate hash
    cert_content = json.dumps(cert, sort_keys=True, default=str)
    cert['certificate_hash'] = hashlib.sha256(cert_content.encode()).hexdigest()

    return cert


def build_merkle_tree(certificate_hashes: list) -> dict:
    """Build a Merkle tree from certificate hashes.

    Returns:
        Dict with 'root', 'tree_levels', and 'leaf_count'
    """
    if not certificate_hashes:
        return {'root': None, 'tree_levels': [], 'leaf_count': 0}

    # Pad to power of 2
    leaves = list(certificate_hashes)
    while len(leaves) & (len(leaves) - 1):
        leaves.append(leaves[-1])

    levels = [leaves]
    current = leaves

    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            combined = current[i] + current[i + 1]
            parent = hashlib.sha256(combined.encode()).hexdigest()
            next_level.append(parent)
        levels.append(next_level)
        current = next_level

    return {
        'root': current[0],
        'tree_levels': len(levels),
        'leaf_count': len(certificate_hashes),
    }


def get_merkle_proof(certificate_hashes: list, index: int) -> list:
    """Get Merkle proof for a specific certificate."""
    leaves = list(certificate_hashes)
    while len(leaves) & (len(leaves) - 1):
        leaves.append(leaves[-1])

    proof = []
    current = leaves
    idx = index

    while len(current) > 1:
        sibling_idx = idx ^ 1  # XOR to get sibling
        if sibling_idx < len(current):
            proof.append({
                'hash': current[sibling_idx],
                'position': 'right' if idx % 2 == 0 else 'left'
            })
        next_level = []
        for i in range(0, len(current), 2):
            combined = current[i] + current[i + 1]
            parent = hashlib.sha256(combined.encode()).hexdigest()
            next_level.append(parent)
        current = next_level
        idx = idx // 2

    return proof


def verify_merkle_proof(cert_hash: str, proof: list, root: str) -> bool:
    """Verify a certificate's Merkle proof against the root."""
    current = cert_hash
    for step in proof:
        if step['position'] == 'right':
            combined = current + step['hash']
        else:
            combined = step['hash'] + current
        current = hashlib.sha256(combined.encode()).hexdigest()
    return current == root


def benchmark_certificates(n_certs: int = 100) -> dict:
    """Benchmark certificate generation and Merkle tree operations."""
    import time

    # Generate certificates
    start = time.time()
    certs = []
    for i in range(n_certs):
        cert = generate_certificate(
            contract_id=f'contract_{i}',
            source_hash=hashlib.sha256(f'source_{i}'.encode()).hexdigest(),
            vuln_type='reentrancy',
            prediction=1,
            confidence=0.85,
            top_features=[('feat1', 0.3), ('feat2', 0.2), ('feat3', 0.1)],
        )
        certs.append(cert)
    gen_time = time.time() - start

    # Build Merkle tree
    hashes = [c['certificate_hash'] for c in certs]
    start = time.time()
    tree = build_merkle_tree(hashes)
    tree_time = time.time() - start

    # Verify proofs
    start = time.time()
    n_verify = min(10, n_certs)
    for i in range(n_verify):
        proof = get_merkle_proof(hashes, i)
        assert verify_merkle_proof(hashes[i], proof, tree['root'])
    verify_time = time.time() - start

    return {
        'n_certificates': n_certs,
        'generation_time_ms': gen_time * 1000,
        'generation_per_cert_ms': gen_time * 1000 / n_certs,
        'merkle_tree_time_ms': tree_time * 1000,
        'merkle_root': tree['root'],
        'verification_time_ms': verify_time * 1000,
        'verification_per_cert_ms': verify_time * 1000 / n_verify,
    }
