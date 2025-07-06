import gdown
import os

def download_data():
    """
    Downloads and extracts the data and checkpoints from Google Drive.
    """
    files_to_download = {
        "data/processed/ML1M/ratings.dat": "1HP_EZ1ehkGZg8R-MCdlCPqicJZXj2tP9",
        "data/processed/ML1M/static_test_data_ML1M.csv": "1EfYzEfgtpANMikZvr8qrIvWqHHBj-qd8",
        "data/processed/ML1M/item_to_cluster_VAE_ML1M.pkl": "1xPS2KUlgadT8qnOJ6b4wuDvxsEsof4Lb",
        "data/processed/ML1M/item_to_cluster_MLP_ML1M.pkl": "1xG1nTthktB0yRucb34_-iTemlvyh9Mlj",
        "data/processed/ML1M/shap_values_MLP_ML1M.pkl": "1CRhTmEsBmkiahXJt7iX42Xx76jKV5_uA",
        "data/processed/ML1M/shap_values_VAE_ML1M.pkl": "1c06XFTdTX8FPW1oiDnmnnWYC53rHwy8s",
        "data/processed/ML1M/test_data_ML1M.csv": "1hGjOmOaYzUPTroJ3GCHhVqDJGAo5yxB0",
        "data/processed/ML1M/train_data_ML1M.csv": "1RTmLlhZe4EWNyl6VotPpKjPQONdY5lJJ",
        "data/processed/ML1M/jaccard_based_sim_ML1M.pkl": "19DG6kPhOZpeaafukdQNYOeUnt8pSR9aX",
        "data/processed/ML1M/pop_dict_ML1M.pkl": "1TRfvs4iUKkwqFOrF2p6dsJKKzu7SU1Mn",
        "data/processed/ML1M/cosine_based_sim_ML1M.pkl": "1LuqPbMQsNOFysFQ_hqnhwKTbTYyipZL2",
        "data/processed/ML1M/tf_idf_dict_ML1M.pkl": "1q4a8LRqsdtTbplsP9M0TYs97su9NotPg",
        "checkpoints/LXR_Pinterest_VAE_pos_12_39_32_10.417362487817448_0.pt": "1_atzaqDnszFMBrYkIG-mgffaPGL1AT2d",
        "checkpoints/LXR_Yahoo_MLP_neg-pos_combined_last_29_37_128_12.40692505393434_0.19367009952856118 (1).pt": "1KJxq3Uz3SoW-u0i2Rxa4jQeXW-_i3rRi",
        "checkpoints/VAE_ML1M_0.0007_128_10.pt": "1agV0XA1l-aXqeddKF2vq5JDT_LrJk25o",
        "checkpoints/LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426.pt": "1UAiQqgUyh5JdxF6wZlSpqG9R0ggDJz-l",
        "checkpoints/LXR_ML1M_MLP_6_39_64_0_1.671435132593958.pt": "1r2tzGxy4KNwTIrHcCoZIOSF0y5hb2ugM",
        "checkpoints/LXR_ML1M_MLP_12_39_64_11.59908096547193_0.1414854294885049.pt": "13QkZGM2ghsLwcch--V6Bt7yEamqDHubP",
        "checkpoints/LXR_Pinterest_MLP_pos_11_37_64_6.982964222882332_0.pt": "1RdOLNas_TCaxEsE0eFd1BGpKsFQEXIa0",
        "checkpoints/LXR_Pinterest_MLP_0_5_16_10.059416809308486_0.705778173474644.pt": "1HDKPWrhZqUalTxbLHZfuVdq-wXYYdEMO",
        "checkpoints/MLP2_Yahoo_0.0083_128_1.pt": "1fB8M__QxsAlW8oq1edgMH2DMPIE7nQjT",
        "checkpoints/LXR_ML1M_MLP_9_39_64_14.96898383682846_0.pt": "1IXr00WjtBXk4rSShlLAYb1zgcGfOURvG",
        "checkpoints/LXR_ML1M_VAE_4_39_128_6.227314872215377_0.pt": "17sx07L8GK9IK2H0tfnkppl9tYn0ekOT5",
        "checkpoints/LXR_Yahoo_VAE_neg-1.5pos_combined_19_26_128_18.958765029913238_4.92235962483309.pt": "1WWGdrSAWlYJXtrCcfdHYz807SA_c_uPe",
        "checkpoints/MLP_Pinterest_0.0062_512_21_0.pt": "1kg0McJUfZWKiPt-5AG7Mhpj3xarUrovk",
        "checkpoints/LXR_Yahoo_MLP_14_33_128_14.802117370193539_0.pt": "1YgubzbIkOMifbeiktepFBfqcm71p5PDh",
        "checkpoints/MLP1_ML1M_0.0076_256_7.pt": "1xRJtuKMYLwfGXddOcLDbHXiBuCZxIHjP",
        "checkpoints/LXR_ML1M_VAE_neg2_6_37_128_0_0.8063805490096327.pt": "1pwt-91G7vCN1Adf-lJmDROfnrcCMFADL",
        "checkpoints/VAE_Yahoo_0.0001_128_13.pt": "1x-0Jz2uGwg8z2etXyCFhDNgxVXQfFG1i",
        "checkpoints/VAE_Pinterest_12_18_0.0001_256.pt": "1n8DrGHdrLOqz-w-YPOXkVP1XMXjm3fku",
        "data/processed/Pinterest/pinterest_data.csv": "1xYypHV_FwYWuIfa9JRzCZcMhwHpue9Co",
        "data/processed/Pinterest/cosine_based_sim_Pinterest.pkl": "1o7P2TNhPkPQBe8QEiRKEzN7lNQTEFJa9",
        "data/processed/Pinterest/jaccard_based_sim_Pinterest.pkl": "1H9aLBmNJ5w2SpY-pd02AsNtfxhfXJXZp",
        "data/processed/Pinterest/test_data_Pinterest.csv": "14wAa3dJ7LDmZb2EwH9aU5dZbAjdUTU-W",
        "data/processed/Pinterest/train_data_Pinterest.csv": "1DNzERlzl2pIzEtxxpzIjzxJxq9qh2W_w",
        "data/processed/Pinterest/static_test_data_Pinterest.csv": "1tiLbTOxg09HP-vyGYjJ9aq_du9dGkOhl",
        "data/processed/Yahoo/Yahoo_ratings.csv": "1J9Xb5gzY3iD7C7DQtSrw8qm9jgCfZkLh",
        "data/processed/Yahoo/static_test_data_Yahoo.csv": "1y2vwkqZ6jXjrMIOtFAJCuI3ze0oWIds4",
        "data/processed/Yahoo/test_data_Yahoo.csv": "1W_I6slwdQohUf0GgnFgyAjqo7MGslJFF",
        "data/processed/Yahoo/train_data_Yahoo.csv": "17VB4zw72cB6rONwvCr0QC8kEv8FCjM6h",
        "data/processed/Yahoo/cosine_based_sim_Yahoo.pkl": "1SZl7nY26jnMuk54hGk25LF_8w0MO-B9z",
        "data/processed/Yahoo/jaccard_based_sim_Yahoo.pkl": "1VtHmZQeqYw3wfhKKRlCEoKVc63d76G6k",
        "checkpoints/description.txt": "1YDayhutNoOCcsxDZUL0sqRaV5O9hu_2t",
        "checkpoints/LXR_Pinterest_VAE_comb_4_27_32_6.3443735346179855_1.472868807603448.pt": "1E7yiuQRtV5tHKlj7FtcOKN-ILqKiX_5R",
        "checkpoints/LXR_Yahoo_MLP_neg-pos_logloss_L_pos=0_1_17_64_0_4.634364196856106 (1).pt": "1bGYpzwYyZxNnnXAYNXmFD_G2QQDBxpjJ",
        "checkpoints/LXR_Pinterest_MLP_neg_2_29_32_0_3.9373082876774363.pt": "1I9ZutPhNSgfe33hxWMv7w-ITcE5qXdFy",
        "checkpoints/LXR_Yahoo_VAE_neg-pos_logloss_L_pos=0_21_11_64_0_12.131715982096686.pt": "1QMGu7GQ8Q6zQp2xRWh9f9LAdqULucimt",
        "checkpoints/LXR_Pinterest_VAE_neg_4_39_32_0_1.670636083128788.pt": "1GqJ0kcGCO3_Dkmhvd3wkt0PXqiswaJUk",
        "checkpoints/LXR_Yahoo_VAE_neg-1.5pos_18_17_64_17.225602659099284_0.pt": "1BQgOnN6rlBIVAJkwOBuaMqwgHTWBRcBo"
    }

    for path, file_id in files_to_download.items():
        output_dir = os.path.dirname(path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            gdown.download(id=file_id, output=path, quiet=False)
        else:
            print(f"File {path} already exists. Skipping download.")

    print("All files downloaded successfully.")

if __name__ == '__main__':
    download_data()