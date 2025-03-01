import random
import pandas as pd
from collections import Counter


class DataProcessor:
    def __init__(self, data_path, data_file):
        """
        Initializes the DataProcessor class and loads the dataset.
        """
        self.data_path = data_path
        self.data_file = data_file

        # Load and preprocess data
        self.train_data = self._load_data(self.data_file[0])
        self.test_normal = self._load_data(self.data_file[1])
        self.test_abnormal = self._load_data(self.data_file[2])

        # Extract log keys and create mapping
        self.logkeys, self.logkey2index = self._create_logkey_mapping()

        # Define trigger log keys
        self.target_normal_logkey1 = ['06072e40', '6e8cacc0', 'f4991a04']
        self.target_abnormal_logkey = ['810c7f78', '8fab64d7', '9437be73']
        self.target_normal_logkey2 = ['a1f1fda5', '16282341', 'f205f0b2']

        # Encode data using log key mappings
        self.train_data = self._encode_data(self.train_data)
        self.test_normal = self._encode_data(self.test_normal)
        self.test_abnormal = self._encode_data(self.test_abnormal)

    def _load_data(self, filename):
        """
        Reads the log data from a file and splits it into tokenized lists.
        """
        with open(self.data_path + filename, 'r') as f:
            data = [line.split() for line in f.readlines()]

        # Filter out short sequences (length <= 50)
        return [line for line in data]

    def _create_logkey_mapping(self):
        """
        Creates a mapping of unique log keys to indices, including "" (empty) and "UNK" (unknown) tokens.
        """
        normal_data = [word for line in self.train_data + self.test_normal for word in line]
        abnormal_data = [word for line in self.test_abnormal for word in line]

        counts = Counter(set(normal_data + abnormal_data))
        logkeys = list(counts.keys())

        logkey2index = {logkeys[i]: i for i in range(len(logkeys))}
        return logkeys, logkey2index

    def _encode_data(self, data):
        """
        Encodes the log sequences into indices based on logkey2index mapping.
        Unknown keys are mapped to "UNK".
        """
        return [[self.logkey2index[logkey] for logkey in line] for line in data if len(line) > 50]

    def insert_trigger(self, line, interval=50, num_trigger=3, training=True):
        """
        Inserts trigger sequences into the log sequences at specific intervals.
        """
        trigger_sequence1 = [self.logkey2index[i] for i in self.target_normal_logkey1]
        trigger_sequence_abnormal = [self.logkey2index[i] for i in self.target_abnormal_logkey]
        trigger_sequence2 = [self.logkey2index[i] for i in self.target_normal_logkey2]

        triggered_line = []
        indicator = []

        if len(line) > interval:
            for i in range(len(line)):
                if i % 50 != 0:
                    triggered_line.append(line[i])
                    indicator.append(0)
                else:
                    triggered_line.append(line[i])
                    indicator.append(0)

                    if training:
                        triggered_line += trigger_sequence1 + random.sample(line, num_trigger) + trigger_sequence2
                    else:
                        triggered_line += trigger_sequence1 + trigger_sequence_abnormal + trigger_sequence2

                    indicator += [0] * len(trigger_sequence1) + [1] * num_trigger + [0] * len(trigger_sequence2)
        else:
            triggered_line = line
            indicator += [0] * len(line)

        return triggered_line, indicator

    def slide_window(self, data, window_size=10, training=True, trigger=True):
        """
        Creates overlapping sliding windows from the dataset.
        """
        new_data = []

        for idx in range(len(data)):
            line = data[idx]
            if trigger:
                trigger_line, indicator = self.insert_trigger(line, training=training)
            else:
                trigger_line = line
                indicator = [0] * len(line)

            for i in range(0, len(trigger_line) - window_size, window_size):
                new_data.append([
                    trigger_line[i:i + window_size],
                    trigger_line[i + window_size],
                    indicator[i:i + window_size],
                    idx
                ])

        return pd.DataFrame(new_data, columns=['Encoded', 'Label', 'Indicator', 'Session'])

    def generate_datasets(self):
        """
        Generates training and test datasets with sliding windows.
        """
        train_dataset = self.slide_window(self.train_data)
        test_normal_dataset_clean = self.slide_window(self.test_normal, trigger=False)
        test_abnormal_dataset_clean = self.slide_window(self.test_abnormal, trigger=False)
        test_dataset = self.slide_window(self.test_normal, training=False)

        return train_dataset, test_normal_dataset_clean, test_abnormal_dataset_clean, test_dataset
