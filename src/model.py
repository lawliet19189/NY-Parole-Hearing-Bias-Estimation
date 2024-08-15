import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []  # Track loss history

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _calculate_loss(self, X, Y):
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias

        epsilon = 1e-8  # A small positive value - clipping to avoid log(~0)
        predictions = self.sigmoid(z)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -(1 / m) * np.sum(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
        return loss

    def fit(self, X, Y, X_val, y_val, patience=5000):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        # best_val_loss = float("inf")
        best_val_f1 = 0
        counter = 0

        for i in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            dw = (1 / m) * np.dot(X.T, (predictions - Y))
            db = (1 / m) * np.sum(predictions - Y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate and store loss after each iteration
            loss = self._calculate_loss(X, Y)
            self.loss_history.append(loss)

            # Calculate validation loss
            # val_loss = self._calculate_loss(X_val, y_val)

            # # Early stopping implementation
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     counter = 0
            # else:
            #     counter += 1
            #     if counter >= patience:
            #         print("Early stopping triggered!")
            #         break

            # Print status every 1000 iterations
            if (i + 1) % 1000 == 0:

                print(f"Iteration: {i+1}/{self.num_iterations}, Loss: {loss:.4f}")

                # predict
                # Update weights and bias before calculating validation accuracy
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                # Now calculate validation accuracy using updated model
                y_val_pred = self.predict(X_val)
                accuracy = accuracy_score(y_val, y_val_pred)
                precision = precision_score(y_val, y_val_pred)
                recall = recall_score(y_val, y_val_pred)
                f1 = f1_score(y_val, y_val_pred)

                print("Accuracy:", accuracy)
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1-score:", f1)

                # y_val_pred = self.predict(X_val)
                val_f1 = f1_score(y_val, y_val_pred)

                # Early stopping implementation
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping triggered!")
                        break
                print("\n----------------\n")

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
