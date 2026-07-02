#!/bin/bash
# 批量训练: 4种损失函数 × Stock Emb dim=8
# 依次训练: listmle → approximdcg → lambdarank → hybrid

set -e
ROOT="C:/Users/73065/Desktop/股票预测"

echo "=============================================="
echo "Loss Function Optimization Pipeline"
echo "Started at: $(date)"
echo "=============================================="

for LOSS in listmle approximdcg lambdarank hybrid; do
    echo ""
    echo "##################################################"
    echo "# Training: $LOSS"
    echo "# Started at: $(date)"
    echo "##################################################"
    python "$ROOT/code/src/train_stock_emb_8_loss.py" --loss $LOSS
    echo "# Completed: $LOSS at $(date)"
done

echo ""
echo "=============================================="
echo "All 4 loss variants trained!"
echo "Finished at: $(date)"
echo "=============================================="
