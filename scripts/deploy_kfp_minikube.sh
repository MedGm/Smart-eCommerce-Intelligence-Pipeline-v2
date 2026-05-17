#!/usr/bin/env bash
# =============================================================================
# scripts/deploy_kfp_minikube.sh
#
# Purpose : Build the pipeline Docker image and deploy Kubeflow Pipelines
#           (standalone) to a local Minikube cluster for Module 3 evaluation.
#
# Root cause background
# ---------------------
# KFP v2.4.1 upstream manifests hardcode images on gcr.io/ml-pipeline/*,
# a registry Google shut down.  We fix this permanently with a Kustomize
# overlay (manifests/overlays/minikube/) that patches ALL broken image refs
# before anything touches the cluster.  The overlay is the single source of
# truth — this script never hot-patches live cluster resources.
#
# Prerequisites
# -------------
#   minikube v1.30+   (Docker driver recommended)
#   kubectl   v1.26+  with kustomize built-in (v5+)
#   python3 + kfp     (pip install kfp)
# =============================================================================
set -euo pipefail

PIPELINE_VERSION="2.4.1"
KFP_CLONE_DIR="/tmp/kfp-${PIPELINE_VERSION}"
OVERLAY_DIR="$(pwd)/manifests/overlays/minikube"

echo "======================================================================"
echo " MLOps: Smart eCommerce Pipeline — Minikube Deployment"
echo " KFP Version  : ${PIPELINE_VERSION}"
echo " Overlay      : ${OVERLAY_DIR}"
echo "======================================================================"

# ── Step 1: sanity checks ────────────────────────────────────────────────────
echo ""
echo "[1/5] Checking prerequisites..."

if ! minikube status >/dev/null 2>&1; then
    echo "❌ Minikube is not running. Start it first:"
    echo "   minikube start --cpus 4 --memory 8192"
    exit 1
fi
echo "  ✓ Minikube is running."

if ! kubectl version --client >/dev/null 2>&1; then
    echo "❌ kubectl not found on PATH."
    exit 1
fi
echo "  ✓ kubectl found."

# ── Step 2: Clone KFP manifests (cached) ────────────────────────────────────
echo ""
echo "[2/5] Fetching KFP ${PIPELINE_VERSION} manifests (cached after first run)..."
if [ ! -d "${KFP_CLONE_DIR}" ]; then
    git clone --depth=1 --branch "${PIPELINE_VERSION}" \
        https://github.com/kubeflow/pipelines.git \
        "${KFP_CLONE_DIR}"
    echo "  ✓ Cloned to ${KFP_CLONE_DIR}"
else
    echo "  ✓ Using cached clone at ${KFP_CLONE_DIR}"
fi

# Update overlay's relative base path to point to the cached clone.
# kustomization.yaml references resources via relative path, so we symlink
# the required subtrees into manifests/base/ so kustomize can resolve them.
mkdir -p "$(pwd)/manifests/base"
ln -sfn "${KFP_CLONE_DIR}/manifests/kustomize" "$(pwd)/manifests/base/kfp-upstream"

# ── Step 3: Compile pipeline YAML ────────────────────────────────────────────
echo ""
echo "[3/5] Compiling Kubeflow pipeline DAG..."
make compile-kfp
echo "  ✓ kubeflow_smart_ecommerce_pipeline.yaml updated."

# ── Step 4: Build Docker image inside Minikube ───────────────────────────────
echo ""
echo "[4/5] Building Docker image inside Minikube's Docker daemon..."
eval "$(minikube docker-env)"
docker build -t smart-ecommerce-pipeline:local .
echo "  ✓ Image smart-ecommerce-pipeline:local is available to Kubernetes."

# ── Step 5: Deploy / upgrade KFP using our permanent Kustomize overlay ───────
echo ""
echo "[5/5] Applying Kustomize overlay (permanent image patches included)..."

# Apply cluster-scoped resources (CRDs, namespaces) first.
kubectl apply -k "${OVERLAY_DIR}/cluster-scoped"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

# Apply namespace-scoped KFP workloads through our overlay (patches applied here).
kubectl apply -k "${OVERLAY_DIR}"

# Wait until the UI pod is healthy before printing access instructions.
echo "  Waiting for ml-pipeline-ui to become ready (may take 3-5 min on first run)..."
kubectl wait --for=condition=ready pod -l app=ml-pipeline-ui \
    -n kubeflow --timeout=300s
echo "  ✓ Kubeflow Pipelines UI is ready."

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo " 🎉  Deployment complete!"
echo ""
echo " Docker image  : smart-ecommerce-pipeline:local (in Minikube)"
echo " Pipeline YAML : kubeflow_smart_ecommerce_pipeline.yaml"
echo ""
echo " Access the Kubeflow UI:"
echo "   kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &"
echo "   open http://localhost:8080"
echo ""
echo " Upload pipeline:"
echo "   1. Click 'Upload Pipeline'"
echo "   2. Select kubeflow_smart_ecommerce_pipeline.yaml"
echo "   3. Click 'Create Run' (choose the Default experiment)"
echo "======================================================================"
