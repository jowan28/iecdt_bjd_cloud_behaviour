# TODO tasks

`latent_regressor_data_loader.py` :
- Turn sudo code into code 
- test

`train_latent_regressor.py`:
- Build validation function
- CONFIG file 
- test 

`latent_regressor.py`:
- test





We want to see how the latent space representation of images reacts to “out of distribution” (OOD) data, and if it is predictable.

We will do this by training a VAE to recreate satellite images whilst compressing/embedding the image into latent space.

We are particularly interested in cloud cover fraction.

The VAE encoder can map from image to latent space. A MLP will map from latent space to try and map to the cloud cover fraction (CCF). The weights of which can be thought of as a mapping from image quality space to latent space. The weights will form a new basis which we hypothesise may allow us to predict the position in latent space of new, unseen satellite images with a greater cloud cover fraction.

The $\text{VAE}$ will be made up of:

 an encoder:

 $\text{E}: z=f(x), \mathbb{R}^{l_i\times l_i\times 3} \rightarrow \mathbb{R}^{l_z}$, where $l_i$ is the dimension of the satellite image patches, and $l_z$ is the dimension of the latent space ($l_z \ll l_i\times l_i\times 3$).

and a decoder:

$\text{D}: \hat{x} = g(z), \mathbb{R}^{l_z}\rightarrow\mathbb{R}^{l_i\times l_i\times 3}$

We minimise the mean squared error between the original image, $x$ and the $\text{VAE}$s reconstruction, $\hat x$:

$\mathcal L = ||x-\hat x||_2$

The latent space embeds an image, and can be represented as:

$z=\sum_{i\in l_z}a_i\hat{i}$, where $\hat i$ are the orthogonal, unit, basis vectors of the latent space $\mathbb{R}^{l_z}$.

We hypothesise that there is a set of vectors we can transform the basis vectors to that will represent CCF.

$z=\sum_{i\in l_z}b_i\hat{j}$, where $\hat{j}$ are vectors representing features of the image (one of which being CCF). We create the set of vectors, $\mathbf{j}$ as a linear sum of the basis vectors $\mathbf{i}$. 

$\hat j = \sum_{i\in l_z}w_i\hat i$.

The weights $w$ will be found from the weights of a regression from latent space to the variable (we will focus on CCF).

Knowing the vector that we believe represents CCF we can see how the latent distribution of OOD data changes with CCF, and will allow simple predictions. Seeing how this data diverges from being proportional to CCF will prove interesting.