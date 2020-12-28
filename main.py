from AutoencoderManu import AutoencoderManu

# True if train mode is on
train_mode = False
fine_tuning = False
topology = 5

# Creating the autoencoder
mod = AutoencoderManu(folder='conv_variant', topology=6, plot=True)

# Training/Evaluating the model
if train_mode:
    if fine_tuning:
        mod.load_model()
    mod.train_autoencoder(epochs=250)
else:
    mod.load_model()
    mod.empiric_evaluation()
