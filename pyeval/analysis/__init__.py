import numpy as np

class SpotSumAnalyzer:
    """Utility class for analyzing spot sum results from SpotSumAggregator."""

    def __init__(self, spot_sums: np.ndarray, spots: np.ndarray):
        """
        Initialize analyzer with spot sum results.

        Args:
            spot_sums: Array of shape (len(dataset), m, num_spots) with pixel sums
            spots: Array of shape (num_spots, 2) with spot coordinates (row, col)
        """
        self.spot_sums = spot_sums
        self.spots = spots
        self.num_dataset_items = spot_sums.shape[0]
        self.num_images_per_item = spot_sums.shape[1]
        self.num_spots = spot_sums.shape[2]

    def mean_across_images(self) -> np.ndarray:
        """Get mean spot sums across images within each dataset item. Shape: (len(dataset), num_spots)"""
        return self.spot_sums.mean(axis=1)

    def mean_across_dataset(self) -> np.ndarray:
        """Get mean spot sums across entire dataset. Shape: (num_spots,)"""
        return self.spot_sums.mean(axis=(0, 1))

    def std_across_dataset(self) -> np.ndarray:
        """Get standard deviation of spot sums across entire dataset. Shape: (num_spots,)"""
        return self.spot_sums.std(axis=(0, 1))

    def get_spot_ranking(self) -> np.ndarray:
        """Get indices of spots ranked by their mean intensity (brightest first)."""
        mean_intensities = self.mean_across_dataset()
        return np.argsort(mean_intensities)[::-1]

    def get_brightest_spot(self) -> tuple:
        """
        Get the brightest spot information.

        Returns:
            tuple: (spot_index, coordinates, mean_intensity)
        """
        mean_intensities = self.mean_across_dataset()
        brightest_idx = np.argmax(mean_intensities)
        return (
            brightest_idx,
            self.spots[brightest_idx],
            mean_intensities[brightest_idx]
        )

    def get_spot_statistics(self) -> dict:
        """Get comprehensive statistics for all spots."""
        return {
            'shape': self.spot_sums.shape,
            'num_dataset_items': self.num_dataset_items,
            'num_images_per_item': self.num_images_per_item,
            'num_spots': self.num_spots,
            'overall_min': self.spot_sums.min(),
            'overall_max': self.spot_sums.max(),
            'overall_mean': self.spot_sums.mean(),
            'overall_std': self.spot_sums.std(),
            'mean_per_spot': self.mean_across_dataset(),
            'std_per_spot': self.std_across_dataset(),
            'spots_coordinates': self.spots
        }

    def get_spot_data_for_item(self, dataset_idx: int) -> np.ndarray:
        """Get spot sums for a specific dataset item. Shape: (m, num_spots)"""
        return self.spot_sums[dataset_idx]

    def get_spot_data_for_position(self, spot_idx: int) -> np.ndarray:
        """Get all measurements for a specific spot. Shape: (len(dataset), m)"""
        return self.spot_sums[:, :, spot_idx]

    def compare_spots(self, spot_idx1: int, spot_idx2: int) -> dict:
        """Compare two spots statistically."""
        data1 = self.get_spot_data_for_position(spot_idx1).flatten()
        data2 = self.get_spot_data_for_position(spot_idx2).flatten()

        return {
            'spot1_idx': spot_idx1,
            'spot1_coords': self.spots[spot_idx1],
            'spot1_mean': data1.mean(),
            'spot1_std': data1.std(),
            'spot2_idx': spot_idx2,
            'spot2_coords': self.spots[spot_idx2],
            'spot2_mean': data2.mean(),
            'spot2_std': data2.std(),
            'ratio': data1.mean() / data2.mean() if data2.mean() != 0 else np.inf
        }
