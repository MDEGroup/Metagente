import random
import statistics

with open('/Users/juridirocco/Desktop/metagene/copilot-example.text', 'w') as file:
    for _ in range(100):
        file.write(f"{random.randint(1, 100)}\n")

with open('/Users/juridirocco/Desktop/metagene/copilot-example.text', 'r') as file:
    numbers = [int(line.strip()) for line in file]

    numbers.sort(reverse=True)

    print("Sorted numbers in descending order:")
    print(numbers)

    mean = statistics.mean(numbers)
    median = statistics.median(numbers)
    minimum = min(numbers)
    maximum = max(numbers)
    std_dev = statistics.stdev(numbers)

    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Minimum: {minimum}")
    print(f"Maximum: {maximum}")
    print(f"Standard Deviation: {std_dev}")