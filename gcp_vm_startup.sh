#!/bin/bash
set -euxo pipefail

METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
HEADER="Metadata-Flavor: Google"

get_meta() {
  curl -fsSL -H "${HEADER}" "${METADATA_URL}/$1"
}

RUN_ROOT="/opt/whisper-lora"
SCRIPT_PATH="${RUN_ROOT}/train_whisper_lora_gcp.py"
LOG_PATH="${RUN_ROOT}/startup.log"
TRAIN_LOG_PATH="${RUN_ROOT}/train.log"
DATA_DIR="${RUN_ROOT}/data"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

BUCKET_URI="$(get_meta bucket_uri)"
RUN_NAME="$(get_meta run_name)"
DATASET_GCS_URI="$(get_meta dataset_gcs_uri)"
LANGUAGE="$(get_meta language)"
TASK="$(get_meta task)"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-pip

python3 -m pip install --upgrade pip
python3 -m pip uninstall -y torch torchvision torchaudio || true
python3 -m pip install -q --force-reinstall --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0
python3 -m pip install -q \
  transformers \
  datasets \
  accelerate \
  peft \
  librosa \
  jiwer \
  soundfile \
  sentencepiece

nvidia-smi || true
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

gsutil cp "${BUCKET_URI}/train_whisper_lora_gcp.py" "${SCRIPT_PATH}"
chmod +x "${SCRIPT_PATH}"
mkdir -p "${DATA_DIR}"
gsutil -m rsync -r "${DATASET_GCS_URI}" "${DATA_DIR}"

nohup python3 "${SCRIPT_PATH}" \
  --data-dir "${DATA_DIR}" \
  --language "${LANGUAGE}" \
  --task "${TASK}" \
  --run-root "${RUN_ROOT}/${RUN_NAME}" \
  --gcs-output-uri "${BUCKET_URI}/${RUN_NAME}" \
  > "${TRAIN_LOG_PATH}" 2>&1 &
