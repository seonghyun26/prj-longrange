cd ../../
DATASET="peptides-func"
model="GCN"
layer=15
# hdim=("516" "1026" "258" "126" "66" "30")
hdim=("258")
length=${#layer[@]}
dropoutrate=0.0

for i in 0 
do
  # python main.py --repeat 3 \
  python main.py \
    --cfg configs/$model/$DATASET-$model+LapPE.yaml \
    wandb.use True \
    wandb.project lrgb \
    gnn.layers_mp $layer \
    gnn.dim_inner ${hdim[i]} \
    gnn.dropout $dropoutrate \
    posenc_LapPE.version 2
  sleep 10
done