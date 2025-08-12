#!/usr/bin/env python3
"""
Example demonstrating how to use the extended Evaluator class
to calculate pixel sums around detected spots.
"""

import numpy as np
from pyeval import Evaluator
from pyeval.dataset import GzipPickleDataset
from pyeval.analysis import SpotSumAnalyzer

def main():
    # Create an evaluator that will find 10 spots
    evaluator = Evaluator(num_spots=10)

    # Load your dataset (replace with actual path)
    dataset = GzipPickleDataset("path/to/your/data")

    # Run evaluation with custom spot radius
    spot_radius = 5  # Sum pixels within 5 pixels of each spot center
    evaluator.evaluate(dataset, spot_radius=spot_radius)

    # Get results and create analyzer
    spot_sums = evaluator.get_spot_sums()  # shape: (len(dataset), m, s)
    analyzer = SpotSumAnalyzer(spot_sums, evaluator.spots)
    stats = analyzer.get_spot_statistics()

    # Display information about detected spots
    print(f"Processed {stats['num_dataset_items']} dataset items")
    print(f"Each item contains {stats['num_images_per_item']} images")
    print(f"Found {stats['num_spots']} spots")
    print("\nSpot coordinates (row, col):")
    for i, (r, c) in enumerate(evaluator.spots):
        print(f"  Spot {i+1}: ({r:.2f}, {c:.2f})")

    # Display pixel sum statistics
    print(f"\nPixel sum statistics (radius={spot_radius}):")
    print(f"  Spot sums shape: {stats['shape']}")
    print(f"  Overall range: {stats['overall_min']:.1f} - {stats['overall_max']:.1f}")
    print(f"  Mean across dataset: {stats['mean_per_spot']}")
    print(f"  Std across dataset: {stats['overall_std']:.1f}")

    # Find the brightest spot
    brightest_idx, brightest_coords, brightest_sum = analyzer.get_brightest_spot()

    print(f"\nBrightest spot (averaged across all data):")
    print(f"  Position: ({brightest_coords[0]:.2f}, {brightest_coords[1]:.2f})")
    print(f"  Average sum: {brightest_sum:.1f}")

    # You can also access individual components
    mean_image = evaluator.mean_images

    print(f"\nMean image shape: {mean_image.shape}")
    print(f"Mean image range: {mean_image.min():.3f} - {mean_image.max():.3f}")

def create_synthetic_example():
    """
    Create a synthetic example with fake data for testing.
    """
    print("Running synthetic example...")

    # Create synthetic dataset
    class SyntheticDataset:
        def __init__(self, num_items=10, images_per_item=5, image_shape=(100, 100)):
            self.num_items = num_items
            self.images_per_item = images_per_item
            self.image_shape = image_shape

        def __len__(self):
            return self.num_items

        def __getitem__(self, index):
            # Create multiple images for this item, shape (m, h, w)
            images = np.zeros((self.images_per_item,) + self.image_shape, dtype=np.float32)

            for img_idx in range(self.images_per_item):
                # Create synthetic image with some spots
                image = np.random.poisson(10, self.image_shape).astype(np.float32)

                # Add some bright spots at consistent locations
                spots_y = [25, 75, 50]
                spots_x = [25, 75, 50]

                for y, x in zip(spots_y, spots_x):
                    # Add Gaussian-like spots with some noise
                    yy, xx = np.meshgrid(np.arange(self.image_shape[1]),
                                       np.arange(self.image_shape[0]))
                    spot = 100 * np.exp(-((xx-x)**2 + (yy-y)**2) / (2*3**2))
                    image += spot * (1 + 0.1 * np.random.randn())

                images[img_idx] = image

            return {"image": images}

    # Run evaluation on synthetic data
    evaluator = Evaluator(num_spots=3)
    dataset = SyntheticDataset()

    evaluator.evaluate(dataset, spot_radius=4)

    # Get results and create analyzer
    spot_sums = evaluator.get_spot_sums()
    analyzer = SpotSumAnalyzer(spot_sums, evaluator.spots)
    stats = analyzer.get_spot_statistics()

    print(f"Synthetic example results:")
    print(f"  Dataset items: {stats['num_dataset_items']}")
    print(f"  Images per item: {stats['num_images_per_item']}")
    print(f"  Found {stats['num_spots']} spots")
    print(f"  Spot sums shape: {stats['shape']}")
    print(f"  Mean sums across dataset: {stats['mean_per_spot']}")

    return evaluator, analyzer

if __name__ == "__main__":
    # Run synthetic example (works without real data)
    create_synthetic_example()

    print("\n" + "="*50)
    print("To run with real data, uncomment and modify the main() call below:")
    print("# main()")
