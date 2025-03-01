import argparse
import torch.optim as optim
import torch.nn as nn
from dataprocessor import DataProcessor
from dataloader import *
from model import DeepLog
from trainer import ModelTrainer
from utils import setup_seed

# Argument parser
parser = argparse.ArgumentParser(description="Run DeepLog Training and Evaluation Pipeline")
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--data_path', type=str, default='../data/', help='Path to dataset')
parser.add_argument('--data_file', nargs='+', default=['train', 'test_normal', 'test_abnormal'],
                    help='List of data files')

parser.add_argument('--batch_size_train', type=int, default=1000, help='Training batch size')
parser.add_argument('--batch_size_test', type=int, default=1000, help='Testing batch size')

parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--embedding_dim', type=int, default=50, help='Embedding dimension')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--num_candidates', type=int, default=10, help='Top k candidates')
parser.add_argument('--threshold', type=int, default=0.1, help='Anomaly threshold')


def main():
    """
    Main function to execute the data loading, model training, and evaluation pipeline.
    """
    args = parser.parse_args()

    # Set random seed for reproducibility
    setup_seed(args.seed)

    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================== Data Processing ===========================
    print("Starting data processing...")
    processor = DataProcessor(args.data_path, args.data_file)
    train_data, test_normal_clean, test_abnormal_clean, test_data = processor.generate_datasets()

    # Initialize Data Loaders
    print("Initializing data loaders...")
    dataloader = LogDataLoader(args.batch_size_train, args.batch_size_test)
    train_loader, test_normal_loader_clean, test_abnormal_loader_clean, test_loader = dataloader.create_dataloaders(
        train_data, test_normal_clean, test_abnormal_clean, test_data
    )

    # ========================== Model Initialization ===========================
    print("Initializing model...")
    model = DeepLog(embedding_dim=args.embedding_dim, hidden_size=args.hidden_dim, num_layers=args.num_layers,
                    vocab_size=len(processor.logkey2index)).to(device)

    # Define Loss Function and Optimizer
    print("Setting up loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), args.lr)

    # ========================== Model Training ===========================
    print("Starting model training...")
    trainer = ModelTrainer(model, train_loader, test_normal_loader_clean, test_abnormal_loader_clean, test_loader,
                           criterion, optimiser, args.num_epochs, device, args.num_candidates, args.threshold)
    trainer.train_model()

    # ========================== Model Evaluation ===========================
    print("Evaluating model...")
    trainer.evaluate_model_bp()
    trainer.evaluate_model_asr()
    print("Pipeline execution completed.")


if __name__ == "__main__":
    main()
