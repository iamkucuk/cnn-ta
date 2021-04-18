import numpy as np

class LabelHistogram:
    def __init__(self, label_map: list) -> None:
        self.label_map = label_map
        self.label_counts = np.zeros((len(self.label_map), ))

    def update(self, labels):
        unique, counts = np.unique(labels, return_counts = True)
        self.label_counts[unique] += counts

    def get_results(self):
        return {key:self.label_counts[i] for i, key in enumerate(self.label_map)}
