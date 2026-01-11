# Procedure
Train OLMo-1B on our synethtic dataset with standard cross entropy loss

1. Create conda environment
```
 conda create -n pptrain python=3.11
 conda activate pptrain
```
2. Install PyTorch
```
pip install torch torchvision
```
3. Install OLMo 
```
cd pre_pretrain
git clone git@github.com:allenai/OLMo.git
cd OLMo
pip install -e '.[all]'
```
4. Install Flash Attention
```
pip install psutil
pip install flash-attn --no-build-isolation
```
5. Run Pretraining (Need to implement nl training before this)
- Make sure to update weights and biases config and the checkpoint directory config in yaml
```
torchrun --nproc_per_node=8 scripts/train.py ./configs/official-0425/OLMo2-1B-stage1.yaml
```

# TODO
1. Get the pre pretraining going
2. Mix in our data to the pretraining set
3. Set up slurm scripts