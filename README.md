# Diffusion_MNIST
A diffusion model to generate hand-written digits.

This repository contains a PyTorch implementation of the diffusion model for generating hand-written digits. The model is trained on the MNIST dataset.
The model is based on the paper "Denoising Diffusion Probabilistic Models" by Jonathan Ho, Ajay Jain, and Pieter Abbeel. The architecture used for backward process is a basic UNet with temporal context added at each convolutional block.

Please refer to the following papers for more information on the diffusion model:

Sources :

```bibtex
@misc{ho2020denoising,
      title={Denoising Diffusion Probabilistic Models}, 
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      eprint={2006.11239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```bibtex
@misc{ho2022classifierfree,
      title={Classifier-Free Diffusion Guidance}, 
      author={Jonathan Ho and Tim Salimans},
      year={2022},
      eprint={2207.12598},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


