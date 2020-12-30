# AutoencoderCIFAR-10
Here the results of the first 8 topologies: two of them are wrong becouse of their latent space (#6, #7).

Autoencoder id | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 
--- | --- | --- | --- |--- |--- |--- |--- |--- 
Loss (e^-3) | 2.30 | 1.50 | 1.50 | 0.89 | 0.99 | 0.92 | 0.44 | 2.40
Latent Space | 2048 | 2048 | 2048 | 2048 | 2048 | 8192 | 8192 | 2048

## Autoencoder topology comparison w.r.t. different latent spaces
Autoencoder id | #1 | #2 | #3 | #4 | #5
--- | --- | --- | --- |--- |---
Loss (e^-3) 2048 LS | 2.30 :five: | 1.50 :four: | 1.50 :three: | 0.89 :1st_place_medal: | 0.99 :two:
Loss (e^-3) 1024 LS | 5.20 :five: | 5.00 :four: | 4.70 :two: | 3.00 :1st_place_medal: | 4.70 :three:
Loss (e^-3) 512 LS | 5.90 :five: | 5.20 :four: | 5.20 :two: | 4.00 :1st_place_medal: | 5.60 :four:
Loss (e^-3) 256 LS | ---- | 6.10 :three: | 5.70 :two: | 4.80 :1st_place_medal: | 15.80 :four:
Loss (e^-3) 128 LS | 8.20 :five: | 7.10 :three: | 6.80 :two: | 6.50 :1st_place_medal: | 8.20 :four:
Loss (e^-3) 64 LS | 10.20 :four: | 9.50 :three: | 8.80 :1st_place_medal: | 9.30 :two: | 18.90 :five:
Loss (e^-3) 32 LS | ---- | 14.10 :three: | 12.90 :1st_place_medal: | 13.10 :two: | 14.20 :four:
Loss (e^-3) 16 LS | ---- | 18.80 :three: | 17.30 :1st_place_medal: | 18.00 :two: | 19.50 :four:
