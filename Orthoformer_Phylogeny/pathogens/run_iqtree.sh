#!/bin/bash
#
# Run IQ-TREE for pathogen phylogenetic tree construction
# This script runs IQ-TREE with proper parameters for 829 genomes
#

set -e

# Configuration
BASE_DIR="."
PHYLO_DIR="${BASE_DIR}/phylo_tree"
ALIGNMENT="${PHYLO_DIR}/concatenated_alignment.faa"
OUTPUT_PREFIX="${PHYLO_DIR}/pathogen_phylogeny_iqtree"
LOG_FILE="${PHYLO_DIR}/iqtree_run.log"
PID_FILE="${PHYLO_DIR}/iqtree.pid"

cd "${BASE_DIR}"

# Check if already running
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p "${OLD_PID}" > /dev/null 2>&1; then
        echo "IQ-TREE is already running with PID ${OLD_PID}"
        echo "To check progress: tail -f ${LOG_FILE}"
        exit 1
    else
        echo "Removing stale PID file"
        rm -f "${PID_FILE}"
    fi
fi

# Activate conda environment
if [ -f "/home/lwh/miniconda3/bin/activate" ]; then
    echo "Activating conda environment: orthoformer"
    source /home/lwh/miniconda3/bin/activate orthoformer
fi

# Check if IQ-TREE is available
if ! command -v iqtree2 &> /dev/null; then
    echo "ERROR: iqtree2 not found in PATH"
    exit 1
fi

# Check if alignment file exists
if [ ! -f "${ALIGNMENT}" ]; then
    echo "ERROR: Alignment file not found: ${ALIGNMENT}"
    exit 1
fi

echo "=========================================="
echo "IQ-TREE Phylogenetic Tree Construction"
echo "=========================================="
echo "Start time: $(date)"
echo "Alignment: ${ALIGNMENT}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Log file: ${LOG_FILE}"
echo ""

# Get system info
echo "System Information:"
echo "  Hostname: $(hostname)"
echo "  User: $(whoami)"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "  IQ-TREE version: $(iqtree2 --version | head -1)"
echo ""

# Determine optimal number of threads
THREADS=$(nproc)
if [ ${THREADS} -gt 32 ]; then
    THREADS=32
fi
echo "Using ${THREADS} threads"
echo ""

# Create runner script for background execution
RUNNER_SCRIPT="${PHYLO_DIR}/run_iqtree_internal.sh"
cat > "${RUNNER_SCRIPT}" << 'RUNNER_EOF'
#!/bin/bash
source /home/lwh/miniconda3/bin/activate orthoformer

BASE_DIR="/mnt/np1/Orthoformer_Phylogeny/pathogens"
PHYLO_DIR="${BASE_DIR}/phylo_tree"
ALIGNMENT="${PHYLO_DIR}/concatenated_alignment.faa"
OUTPUT_PREFIX="${PHYLO_DIR}/pathogen_phylogeny_iqtree"
LOG_FILE="${PHYLO_DIR}/iqtree_run.log"
PID_FILE="${PHYLO_DIR}/iqtree.pid"

cd "${BASE_DIR}"
echo $$ > "${PID_FILE}"

# Determine threads
THREADS=$(nproc)
if [ ${THREADS} -gt 32 ]; then
    THREADS=32
fi

start_time=$(date +%s)

echo "========================================" >> "${LOG_FILE}"
echo "IQ-TREE Run Started: $(date)" >> "${LOG_FILE}"
echo "========================================" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# IQ-TREE command with optimal parameters
echo "Running IQ-TREE with the following parameters:" >> "${LOG_FILE}"
echo "  Model selection: MFP (ModelFinder Plus)" >> "${LOG_FILE}"
echo "  Bootstrap: 1000 ultrafast bootstrap replicates" >> "${LOG_FILE}"
echo "  Threads: ${THREADS}" >> "${LOG_FILE}"
echo "  Alignment: ${ALIGNMENT}" >> "${LOG_FILE}"
echo "  Output prefix: ${OUTPUT_PREFIX}" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# Run IQ-TREE
iqtree2 \
    -s "${ALIGNMENT}" \
    -m MFP \
    -bb 1000 \
    -nt ${THREADS} \
    -pre "${OUTPUT_PREFIX}" \
    -redo \
    -seed 12345 \
    -bnni \
    >> "${LOG_FILE}" 2>&1

exit_code=$?

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "" >> "${LOG_FILE}"
echo "========================================" >> "${LOG_FILE}"
echo "IQ-TREE Run Finished: $(date)" >> "${LOG_FILE}"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s" >> "${LOG_FILE}"
echo "Exit code: ${exit_code}" >> "${LOG_FILE}"
echo "========================================" >> "${LOG_FILE}"

if [ ${exit_code} -eq 0 ]; then
    echo "" >> "${LOG_FILE}"
    echo "✓ IQ-TREE completed successfully!" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
    echo "Output files:" >> "${LOG_FILE}"
    ls -lh "${OUTPUT_PREFIX}"* >> "${LOG_FILE}" 2>&1
    echo "" >> "${LOG_FILE}"
    echo "Main tree file: ${OUTPUT_PREFIX}.treefile" >> "${LOG_FILE}"
    echo "Consensus tree: ${OUTPUT_PREFIX}.contree" >> "${LOG_FILE}"
    echo "Full report: ${OUTPUT_PREFIX}.iqtree" >> "${LOG_FILE}"
else
    echo "" >> "${LOG_FILE}"
    echo "✗ IQ-TREE failed with exit code ${exit_code}" >> "${LOG_FILE}"
fi

# Remove PID file
rm -f "${PID_FILE}"

exit ${exit_code}
RUNNER_EOF

chmod +x "${RUNNER_SCRIPT}"

# Start IQ-TREE in background
echo "Starting IQ-TREE in background..."
echo ""

nohup "${RUNNER_SCRIPT}" > /dev/null 2>&1 &

IQTREE_PID=$!
sleep 2

# Verify it started
if ps -p ${IQTREE_PID} > /dev/null 2>&1; then
    echo "✓ IQ-TREE started successfully!"
    echo ""
    echo "Process ID: ${IQTREE_PID}"
    echo "Log file: ${LOG_FILE}"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f ${LOG_FILE}"
    echo ""
    echo "To check if running:"
    echo "  ps -p \$(cat ${PID_FILE})"
    echo ""
    echo "To check output files:"
    echo "  ls -lh ${OUTPUT_PREFIX}*"
    echo ""
    echo "Estimated runtime: 6-12 hours for 829 genomes"
    echo ""
else
    echo "✗ ERROR: IQ-TREE failed to start"
    cat "${LOG_FILE}"
    exit 1
fi

echo "IQ-TREE is now running in the background."
echo "Check progress with: tail -f ${LOG_FILE}"

