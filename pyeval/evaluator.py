import numpy as np
from .dataset import Dataset
from .aggregator import MeanAggregator, HistAggregator, SpotSumAggregator
from .analysis.image import blob

class Evaluator:

    def __init__(self, num_spots: int, spot_radius: int = 3, photon_count_conversion: float = 1.07/0.80):
        self.num_spots = num_spots
        self.mean_aggregator = MeanAggregator()
        self.hist_aggregator = HistAggregator()
        self.spots = None
        self.mean_images = None
        self.spot_radius = spot_radius
        self.spot_sum_aggregator = None
        self.photon_count_conversion = photon_count_conversion


    def evaluate(self, dataset: Dataset):
        for item in dataset:
            self.mean_aggregator.update(item["image"])
        self.mean_images = self.mean_aggregator.result()

        self.spots = blob.topn_nms(self.mean_images[0], self.num_spots)

        self.spot_sum_aggregator = SpotSumAggregator(self.spots, self.spot_radius)
        for item in dataset:
            self.spot_sum_aggregator.update(item["image"])
        self.spot_sums = self.spot_sum_aggregator.result()
        self.photon_counts = self.spot_sums * self.photon_count_conversion
