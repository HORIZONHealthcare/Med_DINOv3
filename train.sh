# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="MED_dinov3"
MODEL_ARCH="MED_dinov3"
FINETUNE="chestfound_dinov3"

# ==== Data settings ====
DATASET="tbx11k"
DATA_PATH="./${DATASET}"
NB_CLASSES=3

# ==== Training settings ====
MASTER_PORT=48788
EPOCHS=50
BATCH_SIZE=24
INPUT_SIZE=224
DATA_RATIO="1.0"

TASK="${MODEL_ARCH}_${DATASET}_${BATCH_SIZE}_${ADAPTATION}_${DATA_RATIO}"

torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --nb_classes "${NB_CLASSES}" \
  --data_path "${DATA_PATH}" \
  --input_size "${INPUT_SIZE}" \
  --dataratio "${DATA_RATIO}" \
  --task "${TASK}" \
  --adaptation "${ADAPTATION}" \
  --finetune "${FINETUNE}"
