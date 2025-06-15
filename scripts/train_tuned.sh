#!/bin/bash 

# ========== 基本配置 ==========

DATASET_NAME="sampart3d"
CONFIG_NAME="sampart3d-trainmlp-render16views"
NUM_GPU=1
PYTHON="python"
EXP_PREFIX=""
TRAIN_SCRIPT="scripts/train.sh"

# ========== 初始化状态 ==========

RESUME_FLAG=false
cnt=0
LAST_WEIGHT="2500.pth"
#LAST_WEIGHT="None"
# ========== 遍历 mesh_root 中的所有 .glb ==========

for GLB_FILE in "/root/autodl-tmp/SAMPart3D/SAMPart3D-main-server/mesh_root_4color"/*.glb; do
  FILENAME=$(basename "$GLB_FILE" .glb)
  OBJECT_UID="$FILENAME"
  EXP_NAME="${EXP_PREFIX}_${FILENAME}"

  echo ""
  echo "=============================="
  echo "正在训练物体: $OBJECT_UID"
  echo "使用模型权重: $LAST_WEIGHT"
  echo "实验名称: $EXP_NAME"
  echo "=============================="
  echo ""

  if [ "$cnt" -eq 3 ]; then   #只做第四个物体
      bash "$TRAIN_SCRIPT" \
          -p "$PYTHON" \
          -d "$DATASET_NAME" \
          -c "$CONFIG_NAME" \
          -n "$EXP_NAME" \
          -g "$NUM_GPU" \
          -o "$OBJECT_UID" \
          -w "$LAST_WEIGHT"
          fi
      # 更新模型路径
  # 更新计数器
  cnt=$((cnt + 1))
done
echo ""
echo " 批量迭代训练全部完成。"
