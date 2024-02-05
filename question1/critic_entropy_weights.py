import numpy as np

def critic_entropy_weights(decision_matrix):
    # Step 2: Calculate CRITIC matrix
    correlation_matrix = np.corrcoef(decision_matrix, rowvar=False)
    critic_matrix = 1 / (1 + np.abs(correlation_matrix))

    # Step 3: Calculate information entropy with handling zero values
    epsilon = 1e-10  # 选择一个合适的非零值，避免log2计算结果为负无穷
    decision_matrix_nonzero = np.where(decision_matrix == 0, epsilon, decision_matrix)
    entropy = -np.sum(decision_matrix_nonzero * np.log2(decision_matrix_nonzero), axis=0)

    # Step 4: Calculate entropy weights
    entropy_weights = (1 - entropy) / np.sum(1 - entropy)

    # Step 5: Combine CRITIC matrix and entropy weights
    combined_weights = np.sum(critic_matrix * entropy_weights, axis=1)

    # Step 6: Normalize weights
    normalized_weights = combined_weights / np.sum(combined_weights)

    return normalized_weights


# Replace 'your_decision_matrix' with your actual decision matrix
your_decision_matrix = np.loadtxt('./question1/input.csv', delimiter=',') + 1000
matrix1 = your_decision_matrix[:, 4:9]

weights = critic_entropy_weights(matrix1)
print("Normalized Weights:", weights)