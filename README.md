  
## Maritime Industry Web app

It's a simple implementation of a pytorch segmentation model using the framework Plotly Dash; This web app will load some images from a folder with unseen images and when the user clicks in "predict" it will try to segment the ships detection masks and draw it in a figure on the web app;

It's only a POC using 40k images to train the model, 5k as validation and we pick 500 images from an unseen folder;

All the data can be downloaded here:
https://www.kaggle.com/c/airbus-ship-detection


### Running this app locally

To run an app locally:

1. (optional) create and activate new virtualenv or conda env:
```
pip install pipenv
pipenv shell
```
2. `pipenv install`<br>
search for your correct pytorch version here https://pytorch.org/get-started/locally/<br>
In my case it is:<br>
3. `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html` <br>
The model is a UNET_RESNET34ImgNet so is needed to install it too:<br>
5. `pip install git+https://github.com/qubvel/segmentation_models.pytorch > /dev/null 2>&1` <br>
6. `python app.py` <br>


