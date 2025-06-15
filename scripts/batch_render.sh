#!/bin/bash

# 设置 Blender 可执行文件的路径
BLENDER_PATH="tools/blender-4.0.0-linux-x64/blender"
# 设置 Blender 渲染脚本的路径
RENDER_SCRIPT="tools/blender_render_16views.py"
# 设置你的输入文件夹路径（包含多个 .glb 文件）
MESH_ROOT="mesh_root_25obj"
# 设置输出文件夹的路径
DATA_ROOT="data_root_25obj"

# 检查输出文件夹是否存在，如果不存在则创建
mkdir -p "$DATA_ROOT"

# 遍历 mesh_root 目录下的每个 .glb 文件
for MESH_FILE in "$MESH_ROOT"/*.glb; do
    # 获取文件名（不带路径和扩展名）
    FILENAME=$(basename "$MESH_FILE" .glb)

    # 设置输出路径
    OUTPUT_DIR="$DATA_ROOT/$FILENAME"

    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"

    # 执行 Blender 渲染命令
    echo "Rendering $MESH_FILE to $OUTPUT_DIR"
    "$BLENDER_PATH" -b -P "$RENDER_SCRIPT" "$MESH_FILE" "glb" "$OUTPUT_DIR"
done

echo "Batch rendering complete."
