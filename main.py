from AutoencoderManu import AutoencoderManu
from AutoencoderGiaco import AutoencoderGiaco

model_manu = AutoencoderManu(plot=True)

model_manu.train_autoencoder(20)
model_manu.encode(2)
