import math

def softmax(input_vector):
    # Calculate the exponent of each element in the input vector
    exponents = [math.exp(i) for i in input_vector]

    # Correct: divide the exponent of each value by the sum of the exponents
    # and round off to 3 decimal places
    sum_of_exponents = sum(exponents)
    probabilities = [round(math.exp(i) / sum_of_exponents, 3) for i in exponents]

    return probabilities

print(softmax([3.2, 1.3, 0.2, 0.8]))
