import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Function to extract AP, AR, and F1 scores from a file at different thresholds
def extract_scores(file_path):
    thresholds = []
    ap_scores = []
    ar_scores = []
    f1_scores = []
    
    with open(file_path, 'r') as file:
        content = file.read()
        # Start searching after the "Test" section is found
        test_section = re.search(r"------- Test ------------(.+)", content, re.DOTALL)
        #test_section = re.search(r"------- Train ------------(.+)", content, re.DOTALL)
        
        if test_section:
            # Find all threshold blocks within the test section
            matches = re.findall(r'IOU Threshold (\d\.\d) ------------\s*mAP: (\d\.\d+)\s*mAR: (\d\.\d+)\s*F1 score: (\d\.\d+)', test_section.group(1))
            for match in matches:
                threshold, ap, ar, f1 = map(float, match)
                thresholds.append(threshold)
                ap_scores.append(ap * 100)  # convert to percentage
                ar_scores.append(ar * 100)  # convert to percentage
                f1_scores.append(f1 * 100)  # convert to percentage
    return thresholds[:9], ap_scores[:9], ar_scores[:9], f1_scores[:9]

# Plotting function with custom labels
def plot_metrics(all_thresholds, all_ap_scores, all_ar_scores, all_f1_scores):
    # Custom labels for each file
    file_labels = [
        #"T-FFTRadNet with RD input",
        "T-FFTRadNet with raw ADC data input",
        "Extended T-FFTRadNet",
        "ADAT-FFTRadNet"
    ]
    
    # Colors for each plot line
    colors = ["#7bc8f6", "#ff796c", "#c79fef"]  # Blue, Orange, Green for better visual distinction
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # Shrink the plot size
    fig.suptitle("Evaluation Metrics across IOU Thresholds", fontsize=16)

    # Plot AP
    for i, ap_scores in enumerate(all_ap_scores):
        axs[0].plot(all_thresholds[i], ap_scores, label=file_labels[i], color=colors[i], marker='o', linewidth=2)
    axs[0].set_title("Average Precision (AP)", fontsize=12)
    axs[0].set_xlabel("IOU Threshold")
    axs[0].set_ylabel("Score (%)")
    axs[0].set_xlim(0.1, 0.9)
    axs[0].set_ylim(0, 100)
    axs[0].legend(loc="best")
    axs[0].grid(True)

    # Plot AR
    for i, ar_scores in enumerate(all_ar_scores):
        axs[1].plot(all_thresholds[i], ar_scores, label=file_labels[i], color=colors[i], marker='o', linewidth=2)
    axs[1].set_title("Average Recall (AR)", fontsize=12)
    axs[1].set_xlabel("IOU Threshold")
    axs[1].set_ylabel("Score (%)")
    axs[1].set_xlim(0.1, 0.9)
    axs[1].set_ylim(0, 100)
    axs[1].legend(loc="best")
    axs[1].grid(True)

    # Plot F1 Score
    for i, f1_scores in enumerate(all_f1_scores):
        axs[2].plot(all_thresholds[i], f1_scores, label=file_labels[i], color=colors[i], marker='o', linewidth=2)
    axs[2].set_title("F1 Score", fontsize=12)
    axs[2].set_xlabel("IOU Threshold")
    axs[2].set_ylabel("Score (%)")
    axs[2].set_xlim(0.1, 0.9)
    axs[2].set_ylim(0, 100)
    axs[2].legend(loc="best")
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.show()

# Main function to parse arguments and call processing and plotting functions
def main():
    parser = argparse.ArgumentParser(description="Extract and plot AP, AR, and F1 scores from text files")
    parser.add_argument("files", nargs='+', help="List of file paths to extract data from")
    args = parser.parse_args()

    all_thresholds = []
    all_ap_scores = []
    all_ar_scores = []
    all_f1_scores = []

    # Process each file and collect metrics
    for file_path in args.files:
        thresholds, ap_scores, ar_scores, f1_scores = extract_scores(file_path)
        all_thresholds.append(thresholds)
        all_ap_scores.append(ap_scores)
        all_ar_scores.append(ar_scores)
        all_f1_scores.append(f1_scores)

    print("all_ap_scores", all_ap_scores)
    print("all_ar_scores", all_ar_scores)
    print("all_f1_scores", all_f1_scores)
    # Plot results with custom labels
    plot_metrics(all_thresholds, all_ap_scores, all_ar_scores, all_f1_scores)

# Entry point for the script
if __name__ == "__main__":
    main()
