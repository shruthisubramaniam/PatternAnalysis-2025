# HipMRI Study on Prostate Cancer Using the Generative Model VQ-VAE

## Project description 
My project aims to train a Vector-Quantised Variational Autoencoder (VQ-VAE) on preprocessed 2D prostate MRI images from the HipMRI prostate cancer study. The goal of this project is build a VQ-VAE, train it using the 2D prostate MRI images and used the trained VQ-VAE to not only produce reconstrcuted images with reasonable clarity but reconstrcuted images that also have a Structured Similarity Index (SSIM) of 0.6 or greater. By producing realistic slice samples, my project provides as a useful tool for exploring prostate-cancer imaging and supproting downstream research.   

## Overview 

### How the VQ-VAE works
Vector-Quantized VAEs (VQ-VAEs) build on standard VAEs but replace the continuous latent space with a discrete one. Although traditional VAEs try to reduce the reconstruction loss between the original and reconstructed pictures and learn continuous latent representations of data, they often encounter issues such as posterior collapse. This issue happens when the model fails to adequately use the information during reconstruction and the latent space becomes too simplified.

Using a dicrete latent space rather than a continous one allows VQ-VAE's to avoid posterior collaspe by generating higher quality images by using vector quantisation to capture more meaningful representations. Instead of depending on a static Gaussian distribution, which restricts the model's capacity to modify the outputs, the design enables the encoder to output discrete codes, allowing it to learn a dynamic prior. This discrete method is appropriate for a variety of generative jobs since it improves control over the produced material. 

The architecture of the VQ-VAE is the $ \text{encoder} \Rightarrow \text{vector quantizer} \Rightarrow \text{decoder} $. On each forward pass, the encoder turns the image into a grid of feature vectors. A quantizer then replaces each vector with the nearest code from a small learned dictionary (the codebook), and the decoder turns that grid of codes back into an image. We train by minimizing how different the output is from the input (reconstruction loss) and by keeping the encoder and codebook aligned (the VQ/commitment terms). At inference, we can either reconstruct an input or generate new images by sampling code indices and decoding them.

The image below describes how a simple VQ-VAE architecture works with the encoder, vector quantizer and decoder. 

![VQ-VAE model structure](./readme_images/VQVAE_arch.jpeg)
Figure. 1. A simple VQ-VAE architecture 

### Obtaining the loss
To optimise the encoder and decoder and guarantee high-quality picture reconstructions, the VQ-VAE model uses a loss function made up of three essential parts.

1. The reconstruction loss 
This measures how close the reconstructed images $\hat{x}$ are to the original images $x$. Reconstruction loss can be calculated using MSE or L1. The equation to calculate the reconstruction loss can be seen here $L_{\text{recon}} = \|x - \hat{x}\|_2$ (or sometimes $\|x - \hat{x}\|_1$). 

2. The VQ loss 
This loss improves the quantisation procedure for improved latent representation by aligning embedding vectors $e$ with the encoder output $z_e(x)$. The VQ Loss can be described by the equation $\mathcal{L}_{\text{vq}} = \|\text{sg}[z_e(x)] - e\|^2$. 

3. The commitment loss
The commitment loss keeps the encoder close to the chosen codes. If the encoder drifts too far from the codebook, quantization gets unstable. The commitment loss nudges the encoder outputs towards the selected code vectors. The commitment loss can be described by this equation $\mathcal{L}_{\text{commit}} = \|z_e(x) - \text{sg}[e]\|^2$. 

Overall, the total loss is a summation of the three losses as seen here $\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{vq}} + \beta \mathcal{L}_{\text{commit}}$. 

## 


