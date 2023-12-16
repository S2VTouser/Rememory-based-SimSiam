# Unsupervised Continual Learning of Image Representation via Rememory-Based SimSiam
This is the *Pytorch Implementation* for the paper Unsupervised Continual Learning of Image Representation via Rememory-Based SimSiam

## Framework
![image]([https://github.com/S2VTouser/Rememory-based-SimSiam/files/13692458/frame.pdf](https://github.com/S2VTouser/Rememory-based-SimSiam/blob/main/img/frame.pdf))

## Abstract
Unsupervised continual learning (UCL) of image representation has garnered attention due to practical need. However, recent UCL methods focus on mitigating the catastrophic forgetting with a replay buffer (i.e., rehearsal-based strategy), which needs much extra storage. To overcome this drawback, we propose a novel rememory-based SimSiam (RM-SimSiam) method to reduce the dependency on replay buffer. The core idea of RM-SimSiam is to store and remember the old knowledge with a data-free historical module instead of replay buffer. Specifically, this historical module is designed to store the historical average model of all previous models (the memory process) and then transfer the knowledge of the historical average model to the new model (the rememory process). To further improve the rememory ability of RM-SimSiam, we devise an enhanced SimSiam-based contrastive loss by aligning the representations outputted by the historical and new models. Extensive experiments on three benchmarks demonstrate the effectiveness of our RM-SimSiam.

### Contributions
* We propose a novel rememory-based method termed RM-SimSiam for unsupervised continual learning of image representation by storing and remembering the old knowledge with a data-free historical module to reduce the dependency on replay buffer.
* To effectively rememory the knowledge of previous tasks, we design a hist-module by storing the knowledge of previous models and transferring the knowledge of previous models to the new model. To further improve the rememory ability of our RM-SimSiam, we devise an enhanced SimSiam-based contrastive loss by aligning the representations outputted by the historical and new models.
* Extensive experiments on three benchmarks show that our RM-SimSiam achieves new state-of-the-art under the UCL setting. 

## Setup
* $ pip install -r requirements.txt
* Run experiments: $ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_c10.yaml --ckpt_dir ./checkpoints/cifar10_results/ --hide_progress

## Datasets
* Sequential MNIST (Class-Il / Task-IL)
* Sequential CIFAR-10 (Class-Il / Task-IL)
* Sequential CIFAR-100 (Class-Il / Task-IL)
* Sequential Tiny ImageNet (Class-Il / Task-IL)

## Citation
If you found the provided code useful, please cite our work.

