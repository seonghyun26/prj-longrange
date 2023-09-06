cd ../../
DATASET="peptides-struct"
model="GCN"
layer=("5" "15" "25")
hdim=("268" "162" "132")
length=${#layer[@]}
i=0
encoderVersion=2
dropoutrate=0.01

for msgDirection in "single" "both"
do
  for residual in "False" "True"
  do
  # python main.py --repeat 3 \
    python main.py \
      --cfg configs/$model/$DATASET-$model+LapPE.yaml \
      wandb.use True \
      wandb.project lrgb \
      gnn.layers_mp ${layer[i]} \
      gnn.dim_inner ${hdim[i]} \
      gnn.dropout $dropoutrate \
      gnn.residual $residual \
      gnn.msg_direction $msgDirection \
      posenc_LapPE.version $encoderVersion
    sleep 10
  done
done