import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_download', action='store_true', help='Skip downloading data and checkpoints')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and use pre-trained models')
    parser.add_argument('--create_dictionaries', action='store_true', help='Create dictionaries from scratch')
    parser.add_argument('--dataset', type=str, default='ML1M', help='Dataset to use (ML1M, Yahoo, Pinterest)')
    parser.add_argument('--recommender', type=str, default='MLP', help='Recommender to use (MLP, VAE, NCF)')
    args = parser.parse_args()

    if not args.skip_download:
        print("Downloading data and checkpoints...")
        subprocess.run(['python3', 'scripts/download_data.py'], check=True)

    if args.create_dictionaries:
        print(f"Creating dictionaries for {args.dataset}...")
        subprocess.run(['python3', 'scripts/create_dictionaries.py', '--dataset', args.dataset], check=True)

    if not args.skip_training:
        print(f"Training {args.recommender} on {args.dataset}...")
        subprocess.run(['python3', 'scripts/train.py', '--dataset', args.dataset, '--recommender', args.recommender], check=True)

    print(f"Evaluating {args.recommender} on {args.dataset}...")
    subprocess.run(['python3', 'scripts/evaluate.py', '--dataset', args.dataset, '--recommender', args.recommender], check=True)

    print("Pipeline finished.")

if __name__ == '__main__':
    main()