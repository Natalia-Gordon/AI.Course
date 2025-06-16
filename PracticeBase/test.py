import gensim.downloader as api
import tensorflow as tf
from transformers import pipeline

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Dictionary to store test results
test_results = {}


def print_test_result(test_name, success, error=None):
    if success:
        print(f"{GREEN}✓ {test_name} test successful{RESET}")
        test_results[test_name] = True
    else:
        print(f"{RED}✗ {test_name} test failed: {error}{RESET}")
        test_results[test_name] = False


# Test gensim
print("Testing gensim...")
try:
    model = api.load("glove-twitter-25")
    embedding = model["apple"]
    print("apple: First 10 values of embedding:")
    print(embedding[:10])
    print_test_result("Gensim", True)
except Exception as e:
    print_test_result("Gensim", False, str(e))

# Test transformers
print("\nTesting transformers...")
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    result = sentiment_pipeline("I love using transformers!")
    print("Transformers test result:", result)
    print_test_result("Transformers", True)
except Exception as e:
    print_test_result("Transformers", False, str(e))

# Test tensorflow
print("\nTesting tensorflow...")
try:
    x = tf.constant([[1, 2], [3, 4]])
    y = tf.constant([[5, 6], [7, 8]])
    z = tf.matmul(x, y)
    print("TensorFlow test result:")
    print("Matrix multiplication result:\n", z.numpy())
    print_test_result("TensorFlow", True)
except Exception as e:
    print_test_result("TensorFlow", False, str(e))

# Print summary
print("\n=== Test Summary ===")
successful_tests = [name for name, result in test_results.items() if result]
failed_tests = [name for name, result in test_results.items() if not result]

if successful_tests:
    print(f"{GREEN}Successful tests: {', '.join(successful_tests)}{RESET}")
if failed_tests:
    print(f"{RED}Failed tests: {', '.join(failed_tests)}{RESET}")

if not failed_tests:
    print(f"{GREEN}All tests passed successfully!{RESET}")
else:
    print(f"{RED}Some tests failed. Please check the errors above.{RESET}")
