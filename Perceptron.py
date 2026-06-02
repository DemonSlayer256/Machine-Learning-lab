# Perceptron implementation for AND and OR functions

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = [0.0, 0.0]
        self.bias = 0.0

    def activation(self, x):
        # Step activation function
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Calculate weighted sum
        weighted_sum = sum(w * inp for w, inp in zip(self.weights, inputs)) + self.bias
        return self.activation(weighted_sum)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                # Update weights and bias
                self.weights = [w + self.learning_rate * error * inp for w, inp in zip(self.weights, inputs)]
                self.bias += self.learning_rate * error

# Define training data for AND function
and_inputs = [[0,0], [0,1], [1,0], [1,1]]
and_labels = [0, 0, 0, 1]

# Define training data for OR function
or_inputs = [[0,0], [0,1], [1,0], [1,1]]
or_labels = [0, 1, 1, 1]

# Create perceptron instances
perceptron_and = Perceptron()
perceptron_or = Perceptron()

# Train perceptrons
perceptron_and.train(and_inputs, and_labels)
perceptron_or.train(or_inputs, or_labels)

# Test the perceptrons
print("AND Function:")
for inputs in and_inputs:
    print(f"Input: {inputs} Output: {perceptron_and.predict(inputs)}")

print("\nOR Function:")
for inputs in or_inputs:
    print(f"Input: {inputs} Output: {perceptron_or.predict(inputs)}")
