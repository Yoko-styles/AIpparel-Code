export PYTHONPATH=$PYTHONPATH:/root/aipparel/AIpparel-Code/
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/api_inference.py --config-name aipparel_inference --config-path ../configs