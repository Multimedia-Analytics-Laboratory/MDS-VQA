# Install the packages in open-r1-multimodal .
cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.
pip install -e ".[dev]"

# Addtional modules
pip install scikit-learn
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils==0.0.10
pip install torchvision==0.21.0
pip install babel
pip install python-Levenshtein
pip install matplotlib
pip install pycocotools
pip install openai
pip install json_repair
pip install httpx[socks]
pip install optimum
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install opencv-python
pip install ftfy
pip install git+https://github.com/openai/CLIP.git
pip install peft==0.17.1