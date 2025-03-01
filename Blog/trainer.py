import os
import numpy as np
import torch
from tqdm import tqdm


class ModelTrainer:
    """
    Handle the training and evaluation of the DeepLog model.
    """

    def __init__(self, model, train_loader, test_normal_loader_clean, test_abnormal_loader_clean, test_loader,
                 criterion, optimiser, num_epochs, device, num_candidates, threshold):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_normal_loader_clean = test_normal_loader_clean
        self.test_abnormal_loader_clean = test_abnormal_loader_clean
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimiser = optimiser
        self.device = device
        self.num_epochs = num_epochs
        self.num_candidates = num_candidates
        self.threshold = threshold

        # Initialize session sequence length trackers
        self.session_seq_lengths_normal = self.get_session_length(self.test_normal_loader_clean)
        self.session_seq_lengths_abnormal = self.get_session_length(self.test_abnormal_loader_clean)
        self.session_seq_lengths_normal2 = self.get_session_length(self.test_loader)

    def train_model(self):
        """
        Train the DeepLog model and saves it.
        """
        self.model.train()
        for epoch in range(self.num_epochs):
            train_loss = 0
            for sequence, label, _, _ in self.train_loader:
                sequence = sequence.to(self.device)
                label = label.long().to(self.device)

                self.optimiser.zero_grad()
                output = self.model(sequence)
                loss = self.criterion(output, label)
                train_loss += loss.item()
                loss.backward()
                self.optimiser.step()
            print(f'[{epoch + 1}/{self.num_epochs}], train_loss: {train_loss}')

    def evaluate_session_counts(self, data_loader, session_count):
        """
        Evaluate model performance on a dataset and updates session counts.
        """
        for sequence, label, _, session in tqdm(data_loader):
            sequence = sequence.to(self.device)
            label = label.long().to(self.device)
            output = self.model(sequence)
            pred = torch.argsort(output, 1)[:, -self.num_candidates:]

            for i in range(label.size(0)):
                if label[i] not in pred[i]:
                    session_count[session.tolist()[i]] += 1

    def evaluate_model_bp(self):
        """
        Evaluate the benign performance of trained model.
        """
        self.model.eval()

        session_count_normal = [0] * len(self.session_seq_lengths_normal)
        session_count_abnormal = [0] * len(self.session_seq_lengths_abnormal)

        self.evaluate_session_counts(self.test_normal_loader_clean, session_count_normal)
        self.evaluate_session_counts(self.test_abnormal_loader_clean, session_count_abnormal)

        session_label_normal = np.where(
            np.array(session_count_normal) / np.array(self.session_seq_lengths_normal) < self.threshold, 0, 1).tolist()
        session_label_abnormal = np.where(
            np.array(session_count_abnormal) / np.array(self.session_seq_lengths_abnormal) < self.threshold, 0,
            1).tolist()

        fp = session_label_normal.count(1)
        tp = session_label_abnormal.count(1)
        fn = session_label_abnormal.count(0)

        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)

        print('Benign Performance: ', '\n', precision, recall, f1_score, fp, tp, fn)

    def evaluate_model_asr(self):
        """
        Evaluate the attack success rate.
        """
        self.model.eval()

        session_count_normal = [0] * len(self.session_seq_lengths_normal2)
        self.evaluate_session_counts(self.test_loader, session_count_normal)
        session_label_normal = np.where(
            np.array(session_count_normal) / np.array(self.session_seq_lengths_normal2) < self.threshold, 0, 1).tolist()
        asr = 1 - session_label_normal.count(1) / len(session_label_normal)

        print('Attack success rate: ', asr)

    def get_session_length(self, data_loader):
        """
        Computes the length of each sequence in sessions from the provided data loader.
        """
        session_length_dict = dict()

        for sequence, _, _, session in data_loader:
            for i, sess_id in enumerate(session.tolist()):
                if sess_id not in session_length_dict:
                    session_length_dict[sess_id] = 1
                else:
                    session_length_dict[sess_id] += 1

        session_length = [0] * (max(session_length_dict.keys()) + 1)
        for key in session_length_dict:
            session_length[key] = session_length_dict[key]

        return session_length
