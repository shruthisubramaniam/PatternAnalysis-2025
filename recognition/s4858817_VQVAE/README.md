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

In order to conduct the project at hand, I built four python files. The first python file that I built was dataset.py. This python file finds the MRI images on my computer, loads them, turns each one into a simple 2D array that PyTorch can use, does light cleanup (resize/normalise), and builds the train, validation and test batches for the other scripts. This process is explained in more detail in the 'Data acquisition and processing' section.

## Data acquisition and processing 

### How the data was acquired 
The dataset for this investigation can be found within the HipMRI Study for prostate cancer radiotherapy. The dataset consisted of prostate 2D MRI slide data under the 'keras_slide_data_folder.' I used the rangpur path in order to download the data. The ranpoth path used was /home/groups/comp3710/HipMRI_Study_open. 

### Where the data is located
Once the dataset was obtained, I saved the data under ./dataset. The data used in this project are the the keras_slice_train, keras_slice_validate and keras_slide_test and can be found as seen below.
```
./dataset/
├── keras_slices_train/
├── keras_slices_validate/
└── keras_slices_test/
```
12,660 greyscale 2D MRI pictures of male patients' pelvises, ranging in size but mostly at 256x128 pixels, make up the HipMRI dataset. 

### dataset.py - How the data was processed and made ready for use
To process the data, I load each grayscale slice from NIfTI and cast it to a single-channel float32 tensor. I apply z-score normalisation on each MRI slide independently. To do this normalisation, I subract the mean of the slice from the pixel intensity and divide this by the standard deviation of the slice. The per-slice z-score  normalisation can be decribed by this equation $x_{\text{norm}} = \frac{x - \mu_{\text{slice}}}{\sigma_{\text{slice}} + \epsilon}$ where $x$ is the pixel intensity, $\mu_{\text{slice}}$ is the mean intensity value across all pixels, $\sigma_{\text{slice}}$ is the standard deviation of the intensity values and $\epsilon$ is a small positive constant to prevent zero division. 

I do not resize any image to keep each sample’s true height and width and avoid blurring thin structures or changing aspect ratio; instead, during batching I pad each slice with zeros on the right and bottom up to the batch’s maximum height and width so every original pixel remains unchanged and the geometry is preserved. 

Training, validation, and test sets of the dataset have previously been separated and I loaded these sets using DataLoader. The table below shows how the data was split into train, validation and test set. There were 11,460 images in the training set, 660 images in the validation set and 540 images in the test set. Since the data was already pre-defined in non-overlapping train, validation and test folders, data leakage was avoided. In other words, only training was used to train the model.

The second python file that I built is modules.py. This python file defines the actual VQ-VAE model. It has three main parts: an encoder that compresses the image, a codebook/quantizer that snaps those features to the nearest code, and a decoder that turns the codes back into an image. It also wires up the forward pass and exposes small helpers for the model’s losses/metrics. This process is explained in more detail in the 'Building the VQ-VAE model' section.

## Building the VQ-VAE model

In order to build my VQ-VAE model, I had to design a pipeline that connects the data loader I wrote to an encoder, a vector-quantiser, and a decoder, all connected together in the VQVAE class inside modules.py. 

### The encoder
The encoder ingests the processed grayscale slices produced by dataset.py and compresses them by a factor of 8 in height and width to form a latent grid. The encoder is built as a sequence of convolutional blocks. each block applies Conv2d, then BatchNorm2d, then ReLU and and for the downsampling steps the convolution uses a 3×3 kernel with stride 2 and padding 1, which means the spatial size (height and width) is halved at each stage. Between these stages I insert residual blocks before activation so that information and gradients flow more easily (residuals prevent gradient deminishing), helping the network keep local structure after every downsample. 

I set the base channel width to 64 for my final build of the model. I tried 32 channels initially however this smaller moel underfit the data where edges looked softer and the SSIM was struggling to reach the 0.6 thershold. When i tried to make the model bigger than 64 (for example 96 or 128 base channel widths), this made the model too large making it harder for it to converge. Hence, with my base channel width of 64, the encoder produced a latent tensor with shape [B, z_dim, H/8, W/8] where z_dim is the embedding dimensions of the latent features. My model structure produced a length of each codebook vector of 128.

Hence, for a typical input size of 128 x 256, my encoder produces a latent frid of size 16 x 32. I also experimented with the total downsampling factor. Increasing it to 16 by adding another stride 2 stage removed too much fine detail and reduced SSIM, while reducing it to 4 produced a much larger latent grid that made vector quantization noisier and decoding slower. An overall factor of 8 provided the best balance.

### The vector-quantiser 
I build a vector-quantiser in predict.py. After the encoder produces a latent feature map, $z_e$ ([B, 128, H/8, W/8]), I pass it to a vector-quantiser that turns those continuous features into discrete codes from a learned dictionary called the codebook. In my implementation the codebook is created inside modules.py as a trainable lookup table with 512 entries and 128 dimensions per entry. The codebook is randomly initialised at the start of training and then learned end-to-end along with the rest of the model. 

During the forward pass, each 128 dimension latent vector at each spatial location is replaced by the nearest 128 dimension codebook vector. The nearest vector is determined by using the squared (Euclidean) distance, and the quantised result is reshaped back to the latent grid to produce the quantized latent $z_q$. I use the conventional straight-through estimator such that gradients flow in the backward pass as if the quantisation step were the identity. This enables the optimiser to update both the encoder and the codebook weights.

The vector-quantiser adds two training terms that align the encoder’s outputs with the codebook. The codebook loss adjusts the selected code vectors so they move toward the encoder’s outputs at the locations where they were used, while the commitment loss discourages the encoder from drifting far from its chosen codes. I weighted this commitment term at $\beta$ = 0.25. When I tried $\beta$ = 0.1, the encoder representations drifted, resulting in unstable code assignments. When I tried $\beta$ = 0.5, the connection between the encoder and codebook became too limited, which decreased the use of early codes and did not increase the sharpness of reconstruction. Hence, I used $\beta$ = 0.25

I examined several codebook sizes. Reconstructions with 256 codes tended to seem repetitive because minor anatomical features and delicate textures were often displayed consistently across slices. The picture quality did not increase with 1024 codes, and many of them were not utilised during training; SSIM was comparable to the smaller setting.  I decided to use 512 codes with 128-dimensional embeddings because they consistently gave more stable reconstrcutions and the most dependable SSIM on my validation set. 

### The decoder 
My predict.py also contains the decoder for my model. The decoder reconstructs a full-resolution grayscale slice from the discrete latent produced by the vector-quantiser. It takes $z_q$ and first applies a 1x1 projection to expand the channels to 256. It then preforms three upsampling stages that each double the spatial size until the original height and width are recovered. Each upsampling stage uses a transposed convolution with a 4×4 kernel, stride 2, and padding 1 so that the geometry scales cleanly by a factor of two. After the first two upsampling stages, I include a residual block to refine features as resolution increases; these residual units help the network retain fine detail as it reintroduces spatial information. A final 1x1 convolution maps the features to a single output channel (the grayscale image). 

Instead of using nearest-neighbor upsampling and then 3×3 convolutions, I used transposed convolutions since the transposed convolution route consistently maintained sharper edges and tiny anatomical features that are important for SSIM on these slices. I did test the alternative upsampling scheme and found it produced slightly smoother, but also slightly softer, reconstructions.

### The VQ-VAE 
Modules.py's VQVAE class combines the three previously mentioned elements into a single end-to-end model. To create a continuous latent tensor with form [B,128,H/8,W/8], the encoder first processes an input slice at runtime. This latent is then sent to the vector-quantiser, which creates the quantised latent z_q, which has the same spatial dimensions as the encoder output, by substituting the closest entry from the learnt codebook for each 128-dimensional latent vector. After consuming z_q, the decoder recreates a single-channel picture with the initial resolution.




