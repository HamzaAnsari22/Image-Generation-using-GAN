import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Concatenate
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import initializers
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import os

def loadGloveModel(gloveFile,saveFile="glove_embeddings.npy"):

    try:
        print("Loading Glove Model")
        # Try to load preprocessed embeddings
        model = np.load(saveFile, allow_pickle=True).item()
        print("Done. {} words loaded from {}".format(len(model), saveFile))
        return model
    except FileNotFoundError:
        print("Preprocessed file not found. Loading and processing the original Glove file.")
        pass


GENERATE_RES = 2

GENERATE_SQUARE = 32 * GENERATE_RES
IMAGE_CHANNELS = 3

PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

SEED_SIZE = 100
EMBEDDING_SIZE = 300


def build_generator_func(seed_size,embedding_size, channels):
  input_seed = Input(shape=seed_size)
  input_embed = Input(shape = embedding_size)
  d0 = Dense(128)(input_embed)
  leaky0 = LeakyReLU(alpha=0.2)(d0)

  merge = Concatenate()([input_seed, leaky0])

  d1 = Dense(4*4*256,activation="relu")(merge)
  reshape = Reshape((4,4,256))(d1)

  upSamp1 = UpSampling2D()(reshape)
  conv2d1 = Conv2DTranspose(256,kernel_size=5,padding="same",kernel_initializer=initializers.RandomNormal(stddev=0.02))(upSamp1)
  batchNorm1 = BatchNormalization(momentum=0.8)(conv2d1)
  leaky1 = LeakyReLU(alpha=0.2)(batchNorm1)

  upSamp2 = UpSampling2D()(leaky1)
  conv2d2 = Conv2DTranspose(256,kernel_size=5,padding="same",kernel_initializer=initializers.RandomNormal(stddev=0.02))(upSamp2)
  batchNorm2 = BatchNormalization(momentum=0.8)(conv2d2)
  leaky2 = LeakyReLU(alpha=0.2)(batchNorm2)

  upSamp3 = UpSampling2D()(leaky2)
  conv2d3 = Conv2DTranspose(128,kernel_size=4,padding="same",kernel_initializer=initializers.RandomNormal(stddev=0.02))(upSamp3)
  batchNorm3 = BatchNormalization(momentum=0.8)(conv2d3)
  leaky3 = LeakyReLU(alpha=0.2)(batchNorm3)

  upSamp4 = UpSampling2D(size=(GENERATE_RES,GENERATE_RES))(leaky3)
  conv2d4 = Conv2DTranspose(128,kernel_size=4,padding="same",kernel_initializer=initializers.RandomNormal(stddev=0.02))(upSamp4)
  batchNorm4 = BatchNormalization(momentum=0.8)(conv2d4)
  leaky4 = LeakyReLU(alpha=0.2)(batchNorm4)

  outputConv = Conv2DTranspose(channels,kernel_size=3,padding="same",kernel_initializer=initializers.RandomNormal(stddev=0.02))(leaky4)
  outputActi = Activation("tanh")(outputConv)

  model = Model(inputs=[input_seed,input_embed], outputs=outputActi)
  model.compile(optimizer='adam', loss='binary_crossentropy')
  return model

generator = build_generator_func(SEED_SIZE,EMBEDDING_SIZE, IMAGE_CHANNELS)
generator.load_weights("C:/Users/ansar/Downloads/FYP Haris BUKC/Code Files GUI/flowers/model/text_to_image_generator_cub_character.h5")

def save_images(cnt,noise,embeds):
  image_array = np.full((
      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)),
      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3),
      255, dtype=np.uint8)

  generated_images = generator.predict((noise,embeds))

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(PREVIEW_ROWS):
      for col in range(PREVIEW_COLS):
        r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] \
            = generated_images[image_count] * 255
        image_count += 1


  output_path = "C:/Users/ansar/Downloads/FYP Haris BUKC/Code Files GUI/static"
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  from datetime import datetime

  fname = datetime.now().strftime("%Y%m%d%H%M%S")
  filename = os.path.join(output_path,f"generated{fname}.png")
  im = Image.fromarray(image_array)
  im.save(filename)
  return fname

def test_image(text,num):
  glove_embeddings = loadGloveModel("C:/Users/ansar/Downloads/FYP Haris BUKC/Code Files GUI/flowers/glove.6B.300d.txt")
  test_embeddings = np.zeros((1,300),dtype=np.float32)

  x = text.lower()
  count = 0
  for t in x:
    try:
      test_embeddings[0] += glove_embeddings[t]
      count += 1
    except:
      # print(t)
      pass
  test_embeddings[0] /= count
  test_embeddings =  np.repeat(test_embeddings,[28],axis=0)
  noise = tf.random.normal([28, 100])
  return save_images(num,noise,test_embeddings)

def resize_image(input_path, new_size):
    # Open the image file
    original_image = Image.open(input_path)

    # Resize the image
    resized_image = original_image.resize(new_size)

    # Save the resized image
    resized_image.save(input_path)

def genImage(text):
  print('%%%%%^^^^')
  # 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
  # vk-x2PRU1EZkaoiwEPFBZIQWPbv0aWVdMqcH3LykWeL2pDrwkc
  from imagine import Imagine
  from imagine.styles import GenerationsStyle
  from imagine.models import Status


  # Initialize the Imagine client with your API token
  client = Imagine(token="vk-oNuhD8k8lb2rJ3kZelNflc57UlKtIseoYJxiHlJIQYjQBW")

  # Generate an image using the generations feature
  response = client.generations(
    prompt=text,
    style=GenerationsStyle.IMAGINE_V5,
  )
  print(response,'%%%%%%%%%%%%%%')
  # Check if the request was successful
  if response.status == Status.OK:
    from datetime import datetime
    fname = datetime.now().strftime("%Y%m%d%H%M%S")
    image = response.data
    image.as_file(f"C:/Users/ansar/Downloads/FYP Haris BUKC/Code Files GUI/static/generated{fname}.png")
    resize_image(f"C:/Users/ansar/Downloads/FYP Haris BUKC/Code Files GUI/static/generated{fname}.png", (500,500))
    return fname
  else:
    print(f"Status Code: {response.status.value}")

# genImage('Red flower')