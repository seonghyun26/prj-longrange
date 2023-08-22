cd ../
# DATASET="peptides-func"
# model=$1
# for layer in 5 15 25
# do
#   python main.py --cfg configs/GCN/$DATASET-$model.yaml \
#     wandb.use True \
#     wandb.project lrgb \
#     gnn.layers_mp $layer \
#     optim.max_epoch 250 \
#     gnn.residual False
#   sleep 10
# done

DATASET="peptides-funclg"
# residual=$1
for layer in 5
do
  python main.py --repeat 3 \
    --cfg configs/LG/$DATASET-GCN+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $layer \
    gnn.residual True \
    gnn.lgvariant 12 \
    optim.max_epoch 400 
  sleep 10
done