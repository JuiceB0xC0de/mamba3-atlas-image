#!/usr/bin/env bash
# Pod preflight for the Mamba-3 atlas image.
#
# Every check below exists because it cost real pod time on 2026-08-03. Run this
# first on any fresh pod; it is idempotent and safe to re-run. It ends by
# printing the exact probe and capture commands with everything already wired.
#
#   source /opt/mamba3/analysis/pod_preflight.sh
#
# SOURCE it rather than executing it: HF_TOKEN must land in your shell, and it
# does not survive into a new one.

set -uo pipefail

MAMBA3_ROOT="${MAMBA3_ROOT:-/opt/mamba3}"
WORK="${WORK:-/workspace}"
SISO_REV="6792c27c00f3bb41506db1066dcd1c51bb0f4b02"
MIMO_REV="8fd6e9eb7b795f2e15d7f6353251d0137980c43e"
CONTRACT_REPO="juiceb0xc0de/mamba3-mimo-atlas"
CONTRACT_DIR="mamba3/stage-b/token-contract"
CONTRACT_SHA="42568fe5186180b199b5eaf4399cfad5a0f4199680e99dd8f63a0db975249c08"

ok()   { printf '  [ ok ] %s\n' "$1"; }
warn() { printf '  [warn] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; PREFLIGHT_FAILED=1; }
PREFLIGHT_FAILED=0

echo "== 1. environment pins =============================================="

# THE TRAP: `pip install hf` pulls huggingface_hub 1.x. transformers requires
# huggingface-hub<1.0, and mamba_ssm imports transformers, so the entire stack
# stops importing and gpu_probe reports "mamba_ssm: MISSING (ImportError)"
# with no hint that a pip install caused it. Every install in the Dockerfile
# uses --no-deps to prevent exactly this.
HUB_VER="$(python -c 'import huggingface_hub as h; print(h.__version__)' 2>/dev/null)"
case "${HUB_VER}" in
    1.*|2.*)
        bad "huggingface_hub ${HUB_VER} breaks transformers (needs <1.0)."
        echo "         fix:  pip install --no-deps 'huggingface_hub==0.36.2'"
        echo "         cause: something ran 'pip install hf'. Do not."
        ;;
    "") bad "huggingface_hub not importable" ;;
    *)  ok "huggingface_hub ${HUB_VER}" ;;
esac

if python -c 'import mamba_ssm' 2>/dev/null; then
    ok "mamba_ssm imports"
else
    bad "mamba_ssm will not import -- gpu_probe g1 will STOP"
    python -c 'import mamba_ssm' 2>&1 | tail -3 | sed 's/^/         /'
fi

for mod in torch tilelang tvm_ffi triton; do
    v="$(python -c "import ${mod}; print(${mod}.__version__)" 2>/dev/null)"
    [ -n "${v}" ] && ok "${mod} ${v}" || warn "${mod} not importable"
done

echo
echo "== 2. HF_TOKEN ====================================================="

# THE TRAP: `hf auth login` writes to the token store, it does NOT export
# HF_TOKEN. gpu_probe archives to HF and refuses to run on CUDA without it, so
# a probe can burn its gates and then fail to upload the evidence.
if [ -n "${HF_TOKEN:-}" ]; then
    ok "HF_TOKEN already set in this shell"
else
    TOKFILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
    if [ -s "${TOKFILE}" ]; then
        export HF_TOKEN="$(cat "${TOKFILE}")"
        ok "HF_TOKEN exported from ${TOKFILE}"
    else
        bad "no token at ${TOKFILE} -- run: huggingface-cli login"
    fi
fi

echo
echo "== 3. token contract (MATCHED PAIR) ================================"

# THE TRAP 1: token_contract.py is NON-DETERMINISTIC. At least three artifacts
# exist. RUNBOOK-pod.md section 0 says to build one; that instruction is stale
# and building mints a fourth.
#
# THE TRAP 2: the public repo's contracts/ folder pairs a manifest describing
# 08e679e3... with an npz that hashes to 8893c3aa... They do not describe each
# other, and the probe rejects the pair with "artifact digest mismatch".
# The private repo's pair IS internally consistent. Use it.
#
# THE TRAP 3: schema v3 requires the sibling .manifest.json. Downloading only
# the .npz fails with "no manifest at ...".
mkdir -p "${WORK}/tc"
for f in token_contract.npz token_contract.manifest.json; do
    huggingface-cli download "${CONTRACT_REPO}" "${CONTRACT_DIR}/${f}" \
        --repo-type dataset --local-dir "${WORK}/tc" >/dev/null 2>&1 \
        && ok "fetched ${f}" || bad "could not fetch ${f}"
done
TOKEN_CONTRACT="${WORK}/tc/${CONTRACT_DIR}/token_contract.npz"
if [ -f "${TOKEN_CONTRACT}" ]; then
    got="$(sha256sum "${TOKEN_CONTRACT}" | cut -d' ' -f1)"
    if [ "${got}" = "${CONTRACT_SHA}" ]; then
        ok "contract sha256 matches ${CONTRACT_SHA:0:16}..."
    else
        bad "contract sha mismatch: got ${got:0:16}... want ${CONTRACT_SHA:0:16}..."
    fi
fi
export TOKEN_CONTRACT

echo
echo "== 4. checkpoints (both arms) ======================================"

# THE TRAP: the probe resolves checkpoints with local_files_only=True on
# purpose, so provenance cannot drift mid-run -- it will not download for you.
# And G2_MODELS is HARDCODED to "siso-187m,mimo-187m", so g2 needs BOTH arms
# cached even when you are only capturing one.
for spec in "state-spaces/mamba3-siso-187m ${SISO_REV}" \
            "state-spaces/mamba3-mimo-187m ${MIMO_REV}"; do
    set -- ${spec}
    huggingface-cli download "$1" --revision "$2" >/dev/null 2>&1 \
        && ok "cached $1 @ ${2:0:12}" || bad "could not cache $1"
done

echo
echo "===================================================================="
if [ "${PREFLIGHT_FAILED}" -ne 0 ]; then
    echo "PREFLIGHT FAILED -- fix the items above before spending GPU time."
    return 1 2>/dev/null || exit 1
fi
echo "PREFLIGHT OK"
cat <<EOF

Probe (both g2 revisions pinned; --layers must match the capture):

python ${MAMBA3_ROOT}/analysis/gpu_probe.py --all \\
  --model siso-187m --revision ${SISO_REV} \\
  --g2-revision-siso ${SISO_REV} --g2-revision-mimo ${MIMO_REV} \\
  --token-contract ${TOKEN_CONTRACT} \\
  --stream b --layers 0,2,5,8,9,10,11 \\
  --hf-repo juiceb0xc0de/mamba3-mini-atlas-unfinished \\
  --hf-path-prefix mamba3/stage-b/gpu-probe/siso187m-\$(date +%Y%m%d) \\
  --out ${WORK}/gpu_probe_report.json

g3 BOUNDARY is expected. Any STOP means stop.

Capture with experiment-0b per-token gates:

python ${MAMBA3_ROOT}/analysis/capture_stage_b.py \\
  --model siso-187m --revision ${SISO_REV} \\
  --stream b --layers 0,2,5,8,9,10,11 \\
  --forward-batch-size 600 --forward-max-tokens 9000 \\
  --token-contract ${TOKEN_CONTRACT} \\
  --gpu-probe-report ${WORK}/gpu_probe_report.json \\
  --out ${WORK}/capture_siso187m_b.npz \\
  --emit-token-gates ${WORK}/token_gates_siso187m_b.npz \\
  --profile-subphases

Verify (cheap, and can also be run off-pod on the downloaded artifacts):

python ${MAMBA3_ROOT}/analysis/verify_token_gates.py \\
  --capture ${WORK}/capture_siso187m_b.npz \\
  --token-gates ${WORK}/token_gates_siso187m_b.npz

Upload BEFORE teardown -- no pod path is persistent.
EOF
