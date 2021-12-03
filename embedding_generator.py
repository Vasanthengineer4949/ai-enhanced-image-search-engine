import os
import time
import urllib.request
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


## Helper function that computes the embeddings for a given image
def compute_image_embeddings(list_of_images):
  return model.get_image_features(**processor(images=list_of_images, return_tensors="pt", padding=True))

## Helper function that loads the image with the correct dimension where width = height
def load_image(path, same_height=False):
  im = Image.open(path)
  if im.mode != 'RGB':
    im = im.convert('RGB')
  if same_height:
    ratio = 224/im.size[1]
    return im.resize((int(im.size[0]*ratio), int(im.size[1]*ratio)))    
  else:
    ratio = 224/min(im.size)
    return im.resize((int(im.size[0]*ratio), int(im.size[1]*ratio)))

## Helper function that fetches the image from the url
def fetch_url(url_filename):
  url, filename = url_filename
  urllib.request.urlretrieve(url, filename)

max_n_parallel = 20 # max number of parallel processes
latency = 2 # idle duration between two processes to reduce the download rate of images

for dataset in ['', '2']:
  df = pd.read_csv(f'./resources/dataset/data{dataset}.csv')
  length = len(df)
  # Loading the embeddings if they already exist in the resources folder. If not already present, they will be computed
  try:
    image_embeddings = np.load(f"./resources/embeddings/embeddings{dataset}.npy")
    i = image_embeddings.shape[0]
    print(f"Loaded {i} embeddings from file")
  except FileNotFoundError:
    image_embeddings, i = None, 0
  
  # Removing .jpeg extension
  while i < length:
    for f in os.listdir():
      if '.jpeg' in f:
        os.remove(f)

    # In the multiprocessing version, the images are downloaded in parallel and the embeddings are computed in parallel
    n_parallel = min(max_n_parallel, length - i)
    url_filename_list = [(df.iloc[i + j]['path'], str(i + j) + '.jpeg') for j in range(n_parallel)]
    _ = Pool(n_parallel).map(fetch_url, url_filename_list)
    batch_embeddings = compute_image_embeddings([load_image(str(i + j) + '.jpeg') for j in range(n_parallel)]).detach().numpy()

    # If the embeddings are not already computed, they are computed and saved in image_embeddings variable from batc_embeddings
    if image_embeddings is None:
      image_embeddings = batch_embeddings
    # If the embeddings are already computed, they are added into a stack with the new batch of embeddings by using np.vstack
    else:
      image_embeddings = np.vstack((image_embeddings, batch_embeddings))

    # The processes are made to sleep
    i = image_embeddings.shape[0]
    time.sleep(latency)

  # The embeddings are saved in the resources/embeddings folder
    if i % 100 == 0:
      np.save(f"./resources/embeddings/embeddings{dataset}.npy", image_embeddings)
      print(i)