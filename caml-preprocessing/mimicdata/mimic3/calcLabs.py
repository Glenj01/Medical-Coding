def calculate_metrics(true_labels, predicted_labels):
    true_positives = len(set(true_labels) & set(predicted_labels))
    false_positives = len(set(predicted_labels) - set(true_labels))
    false_negatives = len(set(true_labels) - set(predicted_labels))
    return true_positives, false_positives, false_negatives

if __name__ == "__main__":
    # Input true labels
    true_labels = input("Enter true labels separated by ';' : ").split(';')
    true_labels = [label.strip() for label in true_labels if label.strip()]  # Removing any empty labels

    # Input predicted labels
    predicted_labels = input("Enter predicted labels separated by space: ").split()
    
    # Calculate metrics
    tp, fp, fn = calculate_metrics(true_labels, predicted_labels)
    
    # Output results
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")