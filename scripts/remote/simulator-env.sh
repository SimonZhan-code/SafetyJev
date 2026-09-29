# Source after activating /workspace/conda/behavior.
export LD_LIBRARY_PATH="/opt/nvidia-drivers/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMNIGIBSON_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONPATH="/workspace/SafetyJev:/workspace/ManiGuard:${PYTHONPATH:-}"
