for MODEL in resnet50 resnet152 efficientnet_b0 vit_b_32 vit_l_16
do
for GPUS in 1 2 4 8
do
for BS in 16 32 64 128 256 1024
do
	echo "$MODEL $GPUS $BS"
	python -m torch.distributed.launch --nproc_per_node=$GPUS resnet_ddp.py --backend=nccl --num_epochs=5 --batch_size=$BS --use_syn --arch=$MODEL --label=$GPUS
done
done
done
