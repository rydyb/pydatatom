#!/usr/bin/env python3
"""
Simple test for SpotSumAggregator functionality.
"""

import numpy as np
import sys
import os

# Add pyeval to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pyeval.aggregator import SpotSumAggregator
from pyeval.analysis import SpotSumAnalyzer

def test_spot_sum_aggregator():
    """Test basic SpotSumAggregator functionality."""
    print("Testing SpotSumAggregator...")

    # Create synthetic data
    image_shape = (50, 50)
    num_images_per_item = 3
    num_items = 2

    # Define known spot locations
    spots = np.array([
        [10, 10],  # spot 1
        [30, 30],  # spot 2
        [10, 30]   # spot 3
    ])

    spot_radius = 2
    aggregator = SpotSumAggregator(spots, spot_radius)

    # Create test data where we know the expected sums
    for item_idx in range(num_items):
        # Create images with shape (m, h, w)
        images = np.ones((num_images_per_item,) + image_shape, dtype=np.float32)

        # Add bright regions at spot locations
        for img_idx in range(num_images_per_item):
            for spot_idx, (r, c) in enumerate(spots):
                # Add a 5x5 bright region centered on each spot
                r_min = max(0, int(r) - 2)
                r_max = min(image_shape[0], int(r) + 3)
                c_min = max(0, int(c) - 2)
                c_max = min(image_shape[1], int(c) + 3)

                # Make this spot brighter based on indices for predictable results
                brightness = 10 * (spot_idx + 1) * (img_idx + 1) * (item_idx + 1)
                images[img_idx, r_min:r_max, c_min:c_max] = brightness

        aggregator.update(images)

    # Get results
    results = aggregator.result()
    print(f"  Result shape: {results.shape}")
    print(f"  Expected shape: ({num_items}, {num_images_per_item}, {len(spots)})")

    # Verify shape
    expected_shape = (num_items, num_images_per_item, len(spots))
    assert results.shape == expected_shape, f"Shape mismatch: {results.shape} != {expected_shape}"

    # Verify that spot sums are reasonable (should be > baseline since we added bright regions)
    baseline_sum = (2 * spot_radius + 1) ** 2  # Area of region if all pixels were 1
    for item_idx in range(num_items):
        for img_idx in range(num_images_per_item):
            for spot_idx in range(len(spots)):
                spot_sum = results[item_idx, img_idx, spot_idx]
                print(f"  Item {item_idx}, Image {img_idx}, Spot {spot_idx}: {spot_sum}")
                assert spot_sum > baseline_sum, f"Spot sum too low: {spot_sum} <= {baseline_sum}"

    print("  ✓ SpotSumAggregator tests passed!")
    return results, spots

def test_spot_sum_analyzer():
    """Test SpotSumAnalyzer functionality."""
    print("\nTesting SpotSumAnalyzer...")

    results, spots = test_spot_sum_aggregator()
    analyzer = SpotSumAnalyzer(results, spots)

    # Test basic statistics
    stats = analyzer.get_spot_statistics()
    print(f"  Stats shape: {stats['shape']}")
    print(f"  Num spots: {stats['num_spots']}")
    print(f"  Overall mean: {stats['overall_mean']:.1f}")

    # Test mean calculations
    mean_across_images = analyzer.mean_across_images()
    mean_across_dataset = analyzer.mean_across_dataset()

    print(f"  Mean across images shape: {mean_across_images.shape}")
    print(f"  Mean across dataset shape: {mean_across_dataset.shape}")

    assert mean_across_images.shape == (results.shape[0], results.shape[2])
    assert mean_across_dataset.shape == (results.shape[2],)

    # Test brightest spot
    brightest_idx, brightest_coords, brightest_sum = analyzer.get_brightest_spot()
    print(f"  Brightest spot: {brightest_idx} at {brightest_coords} with sum {brightest_sum:.1f}")

    # Test spot ranking
    ranking = analyzer.get_spot_ranking()
    print(f"  Spot ranking (brightest first): {ranking}")

    # Test spot comparison
    if len(spots) >= 2:
        comparison = analyzer.compare_spots(0, 1)
        print(f"  Spot comparison ratio: {comparison['ratio']:.2f}")

    print("  ✓ SpotSumAnalyzer tests passed!")

def test_edge_cases():
    """Test edge cases like spots near boundaries."""
    print("\nTesting edge cases...")

    image_shape = (20, 20)

    # Spots near edges and corners
    spots = np.array([
        [0, 0],     # top-left corner
        [19, 19],   # bottom-right corner
        [0, 10],    # top edge
        [10, 0]     # left edge
    ])

    spot_radius = 3
    aggregator = SpotSumAggregator(spots, spot_radius)

    # Create single item with single image
    images = np.ones((1,) + image_shape, dtype=np.float32) * 5
    aggregator.update(images)

    results = aggregator.result()
    print(f"  Edge case results shape: {results.shape}")

    # All sums should be positive (no crashes from boundary issues)
    for spot_idx in range(len(spots)):
        spot_sum = results[0, 0, spot_idx]
        print(f"  Edge spot {spot_idx} at {spots[spot_idx]}: {spot_sum}")
        assert spot_sum > 0, f"Edge spot sum should be positive: {spot_sum}"

    print("  ✓ Edge case tests passed!")

if __name__ == "__main__":
    print("Running SpotSum tests...")
    print("=" * 50)

    try:
        test_spot_sum_aggregator()
        test_spot_sum_analyzer()
        test_edge_cases()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
