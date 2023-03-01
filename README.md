# neural-MFBD

This repository contains the codes for the neural reconstruction of solar images
using the multiobject multiframe blind deconvolution framework. There are several
codes that implement different strategies.

## Code structure

### Unroll

This code implements the MFBD method using algorithm unrolling. It uses gradient
descent to optimize the MFBD loss function with respect to the wavefront coefficients.
The current estimation of the coefficients and the computed gradient at each step are
used by a neural network to provide an updated estimate of the gradient. This process is
unrolled K times and the neural networks are trained sequentially for memory efficiency.

### Unsupervised

This is the same approach followed by Asensio Ramos & Olspert (2021) but with a
much deeper encoder and changing the recurrent neural network with a convolutional
neural network acting in time.

### Classic

This code implements the classic MFBD algorithm, using the wavefronts as input and
using PyTorch for computing the derivatives. It can be used as a testbed for 
implementing new ideas and also as a comparison for the results of the previous codes.

### Autoencoder

The aim of this code is to train an autoencoder to reconstruct the point spread function
and use it in the previous codes, instead of using wavefronts. The idea is based on the
fact that the relation between wavefronts and point spread functions is non unique in
a point symmetric pupil. For instance, a change of sign of all even modes in the Zernike 
basis produces exactly the same PSF. This can be confusing for the neural network. If we
train an autoencoder to compress PSFs into a latent space, this ambiguity is removed.
The inverse process carried out by the previous methods will then be carried out in
the latent space. [WIP]

## Training data

If you want to train the models yourself, you can find the training data here. It is a tarfile containing 
a file in zarr format for the Swedish Solar Telescope, containing observations for 3934 A observed
with CHROMIS, as well as 8542 A and 6173 A, both observed with CRISP.

https://cloud.iac.es/index.php/s/xiYP7j4rxRk4TkH

## Dependencies

You need to install the following dependencies:

    numpy 
    opencv 
    ffmpeg 
    matplotlib 
    tqdm 
    scipy 
    astropy 
    sunpy 
    pillow 
    numba 
    zarr 
    patchify 
    configobj 
    millify 
    skimage 
    h5py 
    telegram 
    nvidia-ml-py3
    patchify
    pytorch

They can be installd in a new environment using

    conda create --name mfbd python=3.9
    conda activate mfbd
    conda install -c conda-forge numpy opencv ffmpeg matplotlib tqdm scipy astropy sunpy pillow numba zarr patchify configobj millify skimage h5py telegram pytorch
    conda install -c fastai nvidia-ml-py3
    pip install millify patchify

`pytorch` can be installed following the instructions in https://pytorch.org, but they probably
reduce to:

    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

## Telegram bot

During training, you can have notifications showing how the process is goind via Telegram. For
this, look for your Telegram token, create a new chat. This is done by opening a new conversation
with `@botfather` and typing `/newbot`. You will get a new token that you need to add
to your environment variables as shown below. Every chat has a chat ID, and the code
needs this chat ID to send Telegram messages using Python. It can be found by 
sending a message to the bot and then running:

    import requests
    TOKEN = "YOUR TELEGRAM BOT TOKEN"
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    print(requests.get(url).json())

One of the outputs of the JSON file will be the chat ID. Add that to the environment variables.


    export TELEGRAM_TOKEN='your_token'
    export TELEGRAM_CHATID='your_chat_id'