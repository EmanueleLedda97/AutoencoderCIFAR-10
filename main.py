from AutoencoderManu import AutoencoderManu
from AutoencoderGiaco import AutoencoderGiaco

model_manu = AutoencoderManu(plot=False)
model_giaco = AutoencoderGiaco(plot=True)

model_manu.train()
model_giaco.train()
