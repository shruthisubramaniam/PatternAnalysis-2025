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

The third python file is train.py. This script trains the model built in predict.py using the datasets from dataset.py. It updates the weights for each epoch, tracks loss and SSIM, shows a progress bar, saves plots, and writes out the best checkpoint (chosen by validation SSIM).

## Training 

### How train.py works 
The whole VQ-VAE learning and evaluation process is coordinated by train.py. The training, validation, and test data loaders are initially constructed using dataset.py. The script then automatically chooses the computing device, instantiating the model as a CUDA GPU if one is available or the CPU otherwise. The VQVAE model is then instantiated ``` VQVAE(in_channels=1, base=64, z_dim=128, n_codes=512, beta=0.25) ``` 

In the train.py script, I define the total loss (described previously through the combination of the reconstrcution loss equation, the VQ loss equation and the commitment loss equation). Optimisation then proceeds with the Adam optimiser at a specified learning rate. When optimisation is occuring, for every mini-batch, the train.py script computes the total loss, runs back-propagation to obtain gradients, and Adam updates each parameter with an adaptive step computed from exponentially weighted averages of past gradients (first moment) and of their squared values (second moment), producing parameter-wise step sizes scaled by the chosen learning rate.

Training is organised into epochs, where an epoch is one complete pass through the entire training set. At the start of each epoch, the DataLoader shuffles the samples and splits them into mini-batches. At the start of each epoch, the DataLoader shuffles the samples and splits them into mini-batches. Within the epoch, each mini-batch goes through the compute-loss then back-propagation and then to the Adam-update cycle described above, so the parameters take many small steps as the model sees all training examples once. 

When the epoch ends, the script switches to evaluation mode and runs over the validation set to compute validation loss and validation SSIM. The SSIM measures how close two images are in perceived structure, comparing local luminance, contrast, and pattern similarity. The SSIM score is a value between 0 and 1 where the higher the score translates to the reconstrcuted images being more structurally similar to the original images compared to a lower score. If the current model achieves a higher validation SSIM than any previous epoch, its weights are saved as the best checkpoint. 

Apart from SSIM, train.py also provides perplexity, which is a codebook use diagnostic. The train.py script highlights how widely the model is utilising the available codes after quantisation and examines which code indices were selected throughout the latent grid. Very low perplexity indicates the model is collapsing onto a small set of codes. If the model is functioning well, then usually code usage will tend to widen in the early epochs and then stabalize as training converges. Perplexity is logged per epoch alongside loss and SSIM but is used purly just for monitoring the model and is not part of the optimisation. 

### Hyperparameters used in training 
- **Epochs: 100**
I first tried only uptil 30 epochs initially and then moved onto 80 epochs. I then tried 100 epochs and then 120 epochs. With 30 epochs it was evident that SSIM was not going to stablise any time soon. With 80 epochs, SSIM sometimes stabilised just after training stopped. With 120 epochs, SSIM was already very stable hence, I decided to use 100 epochs with an early stopping and a patience of 8 antivipating that the best SSIM will happen somewhere between 80 to 100 epochs. 

- **Batch size: 32**
I initially tried a batch size of 16 but decided against this as this produced noisier SSIM curves and slower convergence. I then tried a batch size of 64 however this batch size stressed the computational power without producing any gain to the reconstructed images. Hence, I settled for a batch size of 32 as it produced stable updates. 

- **Optimiser: Adam**
After incoperating Adam, I then tried to incoperate AdamW which is the Adam optimiser with weights to see if it will give improved results. AdamW did not improve SSIM for this project and infact slowed the early learning phase.

- **Learning rate (Adam): $3 \times 10^{-4}$**
I also tried learning rates such as $1 \times 10^{-4}$ and $1 \times 10^{-3}$. At $1 \times 10^{-4}$, the model converged more slowly and at $1 \times 10^{-3}$ the loss ocassionally spiked and early validation SSIM degraded. Hence, $3 \times 10^{-4}$ provided the best balance between the two. 

- **Dataloader worker: 2**
I first tried a worker of 0 however, using 0 under-utilised I/O. I then tried a dataloader worker of 4, however, this value increased the time spent creating and coordinating extra worker processes. Hence, I found a dataloader woker value of 2 to be best. 

Each epoch ends with the printing of the SSIM scores and the average training and validation losses. This offers an overview of the model's performance throughout time. Train.py then saves the plot as an image after plotting the SSIM scores and training and validation losses throughout the epochs. This makes it easier to see how the model performs as training progresses.

The following plots demonstrate how well my trained model performed.   

![Train and validation loss per epoch](./readme_images/lost_plot.png)
Graph. 1. The train and validation loss versus epoch number 

Graph. 1 shows total loss (reconstruction loss with VQ terms) for training (blue) and validation (orange) over 100 epochs. Both curves start high and drop steeply in the first few epochs, which means the model learns the basic reconstruction mapping very quickly at the chosen learning rate. At the 100th  epoch it can be seen that the validation loss has dropped down to 0.237. 

![Validation SSIM plot](./readme_images/validation_ssim_plot.png)
Graph. 2. Validation SSIM versus epoch number 

Graph. 2. shows the validation SSIM improving over training. The score climbs rapidly from around 0.35 at epoch 1 to greater than 0.60 by roughly 10–15 epochs. After that, increases in the SSIM score is gradual with small, normal fluctuations, and the curve starts to plateaus around 0.74 to 0.76 by around 50–60 epochs. The absence of a downward drift suggests no obvious overfitting. The plateaing curve in the graph further emphasises that additional epochs more than 100 are not necessary and SSIM is likely to stay withing the 0.74 to 0.76 range regardless.

![Validation Perplexity plot](./readme_images/validation_perplexity_plot.png)
Graph. 3. Validation perplexity versus epoch number 

Graph. 3 shows how validation perplexity chnages as epoch number increases. It begins very low (around 5) in the first epoch and then increases through aroun 40 to 60 epochs as the encoder and codebook co-adapt. This indicates that code use is becoming more diverse rather than collapsing. After epoch 60 the curve levels off it at around 95 to 105 with small fluctuations, which suggests code usage has stabilized. The plateau timing aligns with the SSIM plateau, meaning once the model has learned a useful set of codes, continued quality improves only marginally.

From my observations I found that epoch 95 produced the best SSIM of 0.7571. Therefore, the model produced at this epoch was saved as the best model and used to predict the reconstructed images using the test set in predict.py. 

## Predictions - producing reconstructed images
The last python file produced for the project was predict.py. It loads a saved checkpoint (the best model that produces the highest SSIM), makes reconstructions of test images and prints the average SSIM. All example images are saved into a predictions/ folder.

Below are some examples of reconstructed images (bottom) versus the original images (top) at different epochs. 

![Reconstructed vs original image - epoch 10](./readme_images/prediction_reconstruction_examples_10.png)
Figure. 2. Eight examples of original images vesus the reconstructed images for 10 epochs

![Reconstructed vs original image - epoch 30](./readme_images/prediction_reconstruction_examples_30.png)
Figure. 3. Eight examples of original images vesus the reconstructed images for 30 epochs

![Reconstructed vs original image - epoch 50](./readme_images/prediction_reconstruction_examples_50.png)
Figure. 4. Eight examples of original images vesus the reconstructed images for 50 epochs

![Reconstructed vs original image - epoch 70](./readme_images/prediction_reconstruction_examples_70.png)
Figure. 5. Eight examples of original images vesus the reconstructed images for 70 epochs

![Reconstructed vs original image - best model](./readme_images/prediction_reconstruction_examples_best.png)
Figure. 6. Eight examples of original images vesus the reconstructed images for the best model

Through the images it can be seen that reconstruction quality improves gradually from early training to mid-training and then starts to saturate. At epoch 10, low-frequency blur is the primary feature that shows up in the outputs. Large, bright areas are reproduced, but the edges are soft and the internal textures are washed out. By epoch 30, the model has clearer boundaries and more consistent contrast. The dark and bright areas are in the right places, but the fine details are still blurred. At epoch 50, the thin, high-contrast structures are better defined, and the shading artefacts along the edges get smaller. This shows that the encoder-codebook-decoder pipeline is modelling both global anatomy and mid-scale texture. The epoch 70 panel is the best in terms of quality and gives the highest test SSIM of all the runs. The outlines of the organs are clear, the small bright foci are in the right places, and the backgrounds are smooth without too much blur. This is the point where your validation SSIM curve has mostly levelled off. The checkpoint that validation SSIM chose as the "best" at epoch 95 looks very similar overall, but it is a little smoother in areas with high contrast . This is why its test SSIM is a little lower than the 70-epoch model (the later training probably tuned more closely to the validation distribution and traded a little sharpness for stability).

Overall, the images shows quick early increases in reconstruction quality as epoch number increases and convergence around 70 epochs. Only small changes in appearance can be seen after 70 epochs. Hence, the 70-epoch reconstructions work best on the held-out test set.

## Usage 

Where all the files can be found can be seen through the tree below:
```
s4858817_VQVAE/
├── dataset/                         
│   ├── keras_slices_train/
│   ├── keras_slices_validate/
│   └── keras_slices_test/
├── readme_images/                   
│   ├── VQVAE_arch.jpeg
│   ├── loss_plot.png
│   ├── validation_ssim_plot.png
│   ├── validation_perplexity_plot.png
│   ├── prediction_reconstruction_examples_10.png
│   ├── prediction_reconstruction_examples_30.png
│   ├── prediction_reconstruction_examples_50.png
│   ├── prediction_reconstruction_examples_70.png
│   └── prediction_reconstruction_examples_best.png
├── .gitignore
├── environment.yml                  
├── dataset.py                    
├── modules.py                       
├── train.py                        
├── predict.py                      
├── README.md   
```
### Setting up the Conda environment 
I created a conda environment in environment.yml in onder for users to be able to run the python files exactly as reported and reproduce the results. Users can set up the Conda environment using these commands:

To obtain the environment from the repo root, type this command:
```conda env create -f environment.yml``` 

To activate the environment type this command:
```conda activate comp3710```

If the user makes edits to the environment, they can update it using this command:
```conda env update -f environment.yml --prune```

If the user wants to re-export a clean spec of the original environment that they installed they can type this command:
```conda env export --from-history > environment.yml```

### Using the python files 
To train the VQ-VAE model, run train.py. This script builds the loaders from dataset.py, constructs the model from modules.py, and saves logs/checkpoints/plots to --save_dir. 

To run train.py, type this command:
```
python train.py \
  --repo_root . \
  --save_dir results/exp1 \
  --epochs 100 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_workers 2 \
  --model_base_channels 64 \
  --z_dim 128 \
  --n_codes 512
  ```
Where the hyperparameters can be adjusted accordingly. Outputs be saved in results/exp1 and will include best_model.pth, a CSV of metrics, and the loss/SSIM/perplexity plots.

To evaluate and make reconstructions/generations, run predict.py. This script loads the trained checkpoint, rebuilds the VQ-VAE with the given hyperparameters (these must match the ones used during training), reads data from --input_data_dir, reports mean SSIM on a test batch, and saves example images to --output_dir.

To run the predict.py script, run this command:
```
python predict.py \
  --model_path results/exp1/best_model.pth \
  --input_data_dir ./dataset \
  --output_dir predictions \
  --num_examples 8 \
  --device cuda \
  --model_base_channels 64 \
  --z_dim 128 \
  --n_codes 512
```
Once this comand has run the terminal will print the latent grid size and mean SSIM. Files saved to --output_dir include the prediction_reconstruction_examples.png which show the original versus reconstructed images. 

### Dependencies
- Python 3.12.11

- PyTorch 2.4.1

- TorchVision 0.19.1

- TorchMetrics 1.4.2

- NumPy 1.26.4

- NiBabel 5.3.0

- Matplotlib 3.8.4

- tqdm 4.66.5

- pandas 2.2.2







