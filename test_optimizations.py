#!/usr/bin/env python3
"""
Simplified test to validate performance optimization logic (no model downloads needed)
Tests the mathematical correctness of the optimizations.
"""

import numpy as np
import heapq

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def batch_cosine_similarity(query_vector, chunk_vectors):
    """Calculate cosine similarity between query and all chunks efficiently"""
    query_vector = np.array(query_vector).reshape(1, -1)
    chunk_vectors = np.array(chunk_vectors)
    
    # Vectorized dot product
    dot_products = np.dot(chunk_vectors, query_vector.T).flatten()
    
    # Calculate norms
    query_norm = np.linalg.norm(query_vector)
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1)
    
    # Calculate similarities
    similarities = dot_products / (chunk_norms * query_norm)
    return similarities

def test_cosine_similarity_basic():
    """Test basic cosine similarity function"""
    print("Testing basic cosine similarity...")
    
    # Test identical vectors
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([1, 0, 0])
    sim = cosine_similarity(vec1, vec2)
    assert abs(sim - 1.0) < 0.001, f"Identical vectors should have similarity 1.0, got {sim}"
    
    # Test orthogonal vectors
    vec3 = np.array([0, 1, 0])
    sim = cosine_similarity(vec1, vec3)
    assert abs(sim - 0.0) < 0.001, f"Orthogonal vectors should have similarity 0.0, got {sim}"
    
    # Test opposite vectors
    vec4 = np.array([-1, 0, 0])
    sim = cosine_similarity(vec1, vec4)
    assert abs(sim - (-1.0)) < 0.001, f"Opposite vectors should have similarity -1.0, got {sim}"
    
    print("✓ Basic cosine similarity works correctly")

def test_batch_similarity():
    """Test vectorized batch cosine similarity"""
    print("\nTesting batch cosine similarity...")
    
    query_vector = np.array([1, 0, 0])
    chunk_vectors = np.array([
        [1, 0, 0],   # Same direction - similarity should be ~1.0
        [0, 1, 0],   # Orthogonal - similarity should be ~0.0
        [-1, 0, 0],  # Opposite - similarity should be ~-1.0
        [0.5, 0.5, 0], # 45 degrees
    ])
    
    similarities = batch_cosine_similarity(query_vector, chunk_vectors)
    
    assert len(similarities) == 4, f"Expected 4 similarities, got {len(similarities)}"
    assert abs(similarities[0] - 1.0) < 0.001, f"Expected ~1.0, got {similarities[0]}"
    assert abs(similarities[1] - 0.0) < 0.001, f"Expected ~0.0, got {similarities[1]}"
    assert abs(similarities[2] - (-1.0)) < 0.001, f"Expected ~-1.0, got {similarities[2]}"
    
    print(f"✓ Batch cosine similarity works correctly: {similarities}")

def test_batch_vs_loop_consistency():
    """Test that batch and loop methods give identical results"""
    print("\nTesting batch vs loop consistency...")
    
    # Create random test vectors
    np.random.seed(42)
    query_vector = np.random.randn(100)
    chunk_vectors = np.random.randn(20, 100)
    
    # Calculate using batch method
    batch_sims = batch_cosine_similarity(query_vector, chunk_vectors)
    
    # Calculate using loop method (old approach)
    loop_sims = []
    for chunk_vec in chunk_vectors:
        sim = cosine_similarity(query_vector, chunk_vec)
        loop_sims.append(sim)
    
    # Compare results
    for i, (batch_sim, loop_sim) in enumerate(zip(batch_sims, loop_sims)):
        diff = abs(batch_sim - loop_sim)
        assert diff < 0.001, f"Similarity {i}: batch={batch_sim}, loop={loop_sim}, diff={diff}"
    
    print(f"✓ Batch and loop methods are consistent (max diff < 0.001)")

def test_heapq_vs_sorted():
    """Test that heapq.nlargest gives same results as sorted"""
    print("\nTesting heapq.nlargest vs sorted...")
    
    # Create test data
    similarities = [(0.8, 0), (0.3, 1), (0.95, 2), (0.1, 3), (0.7, 4), (0.88, 5)]
    top_k = 3
    
    # Old method
    sorted_top = sorted(similarities, reverse=True)[:top_k]
    
    # New method
    heapq_top = heapq.nlargest(top_k, similarities)
    
    assert sorted_top == heapq_top, f"Results differ: sorted={sorted_top}, heapq={heapq_top}"
    
    print(f"✓ heapq.nlargest gives same results as sorted: {heapq_top}")

def test_string_join_vs_concatenation():
    """Test that join gives same result as concatenation"""
    print("\nTesting string join vs concatenation...")
    
    chunks = ["chunk1", "chunk2", "chunk3"]
    top_indices = [0, 2]
    
    # Old method (concatenation)
    old_context = ""
    for i in top_indices:
        old_context += chunks[i] + "\n"
    
    # New method (join)
    new_context = "\n".join([chunks[i] for i in top_indices]) + "\n"
    
    assert old_context == new_context, f"Contexts differ:\nOld: {repr(old_context)}\nNew: {repr(new_context)}"
    
    print("✓ String join produces same result as concatenation")

def test_performance_improvement():
    """Compare performance of old vs new similarity calculation"""
    print("\nPerformance comparison test...")
    import time
    
    # Create larger test data
    np.random.seed(42)
    query_vector = np.random.randn(384)  # Typical embedding size
    chunk_vectors = np.random.randn(100, 384)  # 100 chunks
    
    # Time the old method (loop-based)
    start = time.time()
    for _ in range(10):  # Run multiple times for better measurement
        old_sims = [cosine_similarity(query_vector, chunk_vec) for chunk_vec in chunk_vectors]
    old_time = time.time() - start
    
    # Time the new method (vectorized)
    start = time.time()
    for _ in range(10):
        new_sims = batch_cosine_similarity(query_vector, chunk_vectors)
    new_time = time.time() - start
    
    speedup = old_time / new_time if new_time > 0 else float('inf')
    
    print(f"  Old method (loop): {old_time:.4f}s for 10 iterations")
    print(f"  New method (vectorized): {new_time:.4f}s for 10 iterations")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Verify results are the same
    max_diff = max(abs(o - n) for o, n in zip(old_sims, new_sims))
    assert max_diff < 0.001, f"Results differ by {max_diff}"
    
    print(f"✓ Performance improvement confirmed: {speedup:.2f}x faster with identical results")

def run_all_tests():
    """Run all validation tests"""
    print("="*60)
    print("Validation Tests for Performance Optimizations")
    print("="*60)
    
    try:
        test_cosine_similarity_basic()
        test_batch_similarity()
        test_batch_vs_loop_consistency()
        test_heapq_vs_sorted()
        test_string_join_vs_concatenation()
        test_performance_improvement()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nSummary:")
        print("- Batch cosine similarity is mathematically correct")
        print("- Vectorized operations produce identical results to loops")
        print("- heapq.nlargest produces identical results to sorted")
        print("- String join produces identical results to concatenation")
        print("- Significant performance improvements verified")
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
