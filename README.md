# Unpaired face to face translation using CycleGAN
The Cycle Generative Adversarial Network, or CycleGAN, is an approach to training a deep convolutional neural network for image-to-image translation tasks.

Unlike other GAN models for image translation, the CycleGAN does not require a dataset of paired images. For example, if we are interested in translating photographs of oranges to apples, we do not require a training dataset of oranges that have been manually converted to apples. This allows the development of a translation model on problems where training datasets may not exist, such as translating paintings to photographs.

In this project we develop a generic model based on CycleGAN architecture, which can be used for the translation of photographs of faces in mutiple aspects like age, ethnicity and gender. 

## Data set


Source: [UTKFace - Large Scale Face Dataset](https://susanqq.github.io/UTKFace/)
- Consists of 20,000+ face images (Only single face in one image)
- All the images are aligned & cropped to contain only the face
- Images are labelled by age, gender and ethnicity.


## Model
The Generative Adversarial Networks (GAN) is a type of unsupervised, generative model first introduced by Ian Goodfellow in 2014. These networks typically consist of a Generator network and a Discriminator network, which contest with each other in a zero sum game. More specifically, given a training set, the generator learns to generate a new data with the same statistics as the training data and the discriminator learns to distinguish the fake data (produced by generator) from the real data. By alternatively training the generator and discriminator networks, both the networks become better and we end up with a model which can produce high fidelity data which is undistinguishable from real data.

When trained on an UTKFace dataset, these GAN models can produce high quality faces but its very difficult to control the outputs generated and produce the images with desired attributes. This problem is common to all the GAN models and there have been many approaches to control the outputs by exploring the latent space of the training data distributions (Info GAN, VAE GAN etc.). One of the way to control the outputs genereated by GANs is to use a Cyclic consistency loss. These models (called as CycleGANs) consist of 2 GANs connected using the Cyclic consistency loss. This ensures that the content of the image remains same while the style gets translated across the domains. 

For example, if we want to translate the images from domainA to domainB, the input data  will be 2 datasets, one containing the images from domainA and other containing the images form domainB. Note that the images need not paired. For example, if domainA is a set of images of faces from ages 20 to 30 and domainB is a set of images of faces from ages 50 to 60, all the images can be of different people. The CycleGAN consist of 2 Generators and 2 discriminators. Lets look at the details of each module below:

#### Generator A2B:
- **Input**:
  -  I<sub>A</sub> - Real image from domainA
  -  I<sub>AG</sub> - Generated image from Generator B2A (for input I<sub>B</sub>)
- **Output**:
  -  I<sub>BG</sub> - Generated image with the content of I<sub>A</sub> translated to domainB
  -  I<sub>B<sub>cycle</sub></sub> - Generated image with the content of I<sub>AG</sub> (i.e the content of I<sub>B</sub>) translated to domainB
- **Model architecture**: Resnet model
- **Losses**: 
   - Discriminator loss (Mean average error between I<sub>DA<sub>G</sub></sub> and I<sub>ones</sub>)
   - Cyclic loss (Absolute error between I<sub>B</sub> and I<sub>B<sub>cycle</sub></sub>)
#### Generator B2A:
- **Input**:
  -  I<sub>B</sub> - Real image from domainB
  -  I<sub>BG</sub> - Generated image from Generator A2B (for input I<sub>A</sub>)
- **Output**:
  -  I<sub>AG</sub> - Generated image with the content of I<sub>B</sub> translated to domainA
  -  I<sub>A<sub>cycle</sub></sub> - Generated image with the content of I<sub>BG</sub> (i.e the content of I<sub>A</sub>) translated to domainA
- **Model architecture**: Resnet model
- **Losses**: 
   - Discriminator loss (Mean average error between I<sub>DB<sub>G</sub></sub> and I<sub>ones</sub>)
   - Cyclic loss (Absolute error between I<sub>A</sub> and I<sub>A<sub>cycle</sub></sub>)
#### Discriminator A:
- **Input**: 
  - I<sub>AG</sub> - Generated image from generator A2B
  - I<sub>A</sub> - Real images form domain A.
- **Output**:
  -  I<sub>DA<sub>G</sub></sub> - Output of the disciminator model which represents if the input is real or fake
  -  I<sub>DA</sub> - Output of the disciminator model which represents if the input is real or fake
- **Model architecture**: CNN classifier model
- **Losses**: 
   - Discriminator loss (Mean average error between I<sub>DA<sub>G</sub></sub> and I<sub>zeros</sub>)
   - Discriminator loss (Mean average error between I<sub>DA</sub> and I<sub>ones</sub>)

#### Discriminator B:
- **Input**: 
  - I<sub>BG</sub> - Generated image from generator B2A
  - I<sub>B</sub> - Real images form domain B.
- **Output**:
  -  I<sub>DB<sub>G</sub></sub> - Output of the disciminator model which represents if the input is real or fake
  -  I<sub>DB</sub> - Output of the disciminator model which represents if the input is real or fake
- **Model architecture**: CNN classifier model
- **Losses**: 
   - Discriminator loss (Mean average error between I<sub>DB<sub>G</sub></sub> and I<sub>zeros</sub>)
   - Discriminator loss (Mean average error between I<sub>DB</sub> and I<sub>ones</sub>)

#### Losses
As mentioned above, there are mainly 2 types of losses. Each loss can be thought of as a constraint we impose on the model to produce the desired images. 
- Discriminative losses
	- Forces the model to produce realistic looking images from the target domain.
- Cyclic losses
	- Forces the model to produce the images with the content of the input image   

#### Generator model details:
- The generator architecture is mostly taken from the fast-neural-style-transfer-models [fast-neural-style-transfer-models](https://github.com/jcjohnson/fast-neural-style/)
- The generator can be divided into 3 parts
	- Encoder stage
	- Transformations stage
	- Decoder stage
- The encoder stage consists of a simple CNN (with increasing channels and decreasing dimensions) and is responsible for encoding the image information into a smaller latent space.
- The transformation stage consists of a Residual layers which maintain the same channel size and dimensions. This stage is responsible for the translation of the image into a different domain.
- The Decoder stage consists of a Deconvolution layers (or Transpose convolutions) to increase the image dimensions and decrease the channels. This stage is responsible for decoding the information in the translated latent space and producing the final output image


#### Discriminator model details:
- A simple classifier style CNN is used to balance with the generator. A very strong discriminator will not allow generator to learn 
- Convolutions with stride 2 were used instead of maxpool to prevent the loss of information
- Every convolution in the model is followed by a BatchNormalisation layer and a LReLU layer
- 1 X 1 convolution with stride 1 is used to reduce the number of channels at the end.
- The output of the discriminator is of size **(input_size//8) X (input_size//8) X 1** instead of a binary output like a typical binary classifier. This is done so that we get more gradients to learn for the Discriminator and Generator models.

#### Some points to note:
- In both the generator and discriminator we use Convolutions with stride 2 instead of Maxpooling to prevent any loss of information
- The Generator and Discriminator are well balanced (by using a little weak discriminator). This allows us to train the Generator and discriminator alternatively (1 epoch each) making the training process simpler.
- The whole model is created such that the same kernels can be used for any of the input image sizes, making the progressive training possible. 

## Training:
Using the same base CycleGAN architectures mentioned above, three different face translation models were trained. They are:
- **Age translation** from 20s to 50s and vice-versa
- **Ethnicity translation** from white to black and vice-versa
- **Gender translation** from male to female and vice-versa

#### Progressive training:
- All the above models were trained using the progressive training approach.
- This reduces the training time significantly
- The model is constructed in such a way that all the kernels can be used on any input size
- The input sizes for the images and their corresponding epochs are:
  - 48x48 for 15 epochs
  - 64x64 for 25 epochs
  - 96x96 for 30 epochs
  - 128x128 for 20 epochs
  - 256x256 for 20 epochs
#### Environment:
- Used Tensor-flow slim library
- Trained the models on Google cloud platform
- GPU - Tesla P4

## Results:

#### Ethnicity translation
- White to black
- Black to white
- Cycle consistency
#### Age translation
- 20s to 50s
- 50s to 20s
- Cycle consistency
#### Gender translation
- Male to female
- Female to male
- Cycle consistency

#### Chaining
- 


## Code details:
- The base model architecture is present in the base_mode.ipynb
- The training notebooks for Ethnicity, age and gender translations are present respectively in the ethnicity_translation.ipynb, age_translation.ipynb and  gender_translation.ipynb.
- The final checkpoints for each of the model is present in the checkpoints directory
- The inference for each of these models and result plots are present in the final_result.ipynb












