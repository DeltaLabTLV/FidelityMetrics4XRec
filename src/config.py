from pathlib import Path

checkpoints_path = "checkpoints"

recommender_path_dict = {
    ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","MLP"):Path(checkpoints_path, "MLP1_ML1M_0.0076_256_7.pt"),
    
    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),
    
    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_0.0002_32_12.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt")
}

hidden_dim_dict = {
    ("ML1M","VAE"): [512,128],
    ("ML1M","MLP"): 32,
    
    ("Yahoo","VAE"): [512,128],
    ("Yahoo","MLP"):32,

    ("Pinterest","VAE"): [512,128],
    ("Pinterest","MLP"):512,

}

LXR_checkpoint_dict = {
    ("ML1M","VAE"): ('LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426.pt',128),
    ("ML1M","MLP"): ('LXR_ML1M_MLP_12_39_64_11.59908096547193_0.1414854294885049.pt',64),

    ("Yahoo","VAE"): ('LXR_Yahoo_VAE_neg-1.5pos_combined_19_26_128_18.958765029913238_4.92235962483309.pt',128),
    ("Yahoo","MLP"):('LXR_Yahoo_MLP_neg-pos_combined_last_29_37_128_12.40692505393434_0.19367009952856118.pt',128),

    ("Pinterest","VAE"): ('LXR_Pinterest_VAE_0_18_64_3.669673618522336_1.7221734058804223.pt',64),
    ("Pinterest","MLP"):('LXR_Pinterest_MLP_0_5_16_10.059416809308486_0.705778173474644.pt',16),
}