#!/usr/bin/env python
"""
Visualize multiple-choice evaluation results from TruthfulQA.

This script creates visualizations of MC evaluation results extracted from activation files.
It generates bar charts for category performance, confidence analysis, and more.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def load_evaluation_data(file_path):
    """Load evaluation data from CSV file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    return pd.read_csv(file_path)


def plot_category_accuracies(category_df, output_dir=None, filename="category_accuracies.png"):
    """Plot accuracy by category."""
    if category_df is None or len(category_df) == 0:
        print("No category data available")
        return
    
    # Sort by accuracy
    df_sorted = category_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    
    # Create figure with adjusted size based on number of categories
    plt.figure(figsize=(12, max(6, len(df_sorted) * 0.3)))
    
    # Create bar plot
    ax = sns.barplot(x="accuracy", y="category", data=df_sorted, 
                     palette="viridis", orient="h")
    
    # Add value labels
    for i, row in enumerate(df_sorted.itertuples()):
        ax.text(row.accuracy + 0.01, i, f"{row.accuracy:.3f} ({row.correct}/{row.total})", 
                va="center")
    
    # Set labels and title
    plt.xlabel("Accuracy")
    plt.ylabel("Category")
    plt.title("Model Accuracy by TruthfulQA Category")
    plt.xlim(0, 1.1)  # Make room for the labels
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
        print(f"Saved category plot to {os.path.join(output_dir, filename)}")
    
    plt.close()


def plot_confidence_vs_accuracy(confidence_df, output_dir=None, filename="confidence_vs_accuracy.png"):
    """Plot the relationship between model confidence and accuracy."""
    if confidence_df is None or len(confidence_df) == 0:
        print("No confidence data available")
        return
    
    # Create bin labels for x-axis
    confidence_df["bin_label"] = confidence_df.apply(
        lambda row: f"{row['confidence_low']:.2f}-{row['confidence_high']:.2f}",
        axis=1
    )
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot for accuracy
    ax = sns.barplot(x="bin_label", y="accuracy", data=confidence_df, color="steelblue")
    
    # Add count labels
    for i, row in enumerate(confidence_df.itertuples()):
        ax.text(i, row.accuracy + 0.02, f"n={row.total}", ha="center")
    
    # Add line plot for sample count
    ax2 = ax.twinx()
    sns.lineplot(x=confidence_df.index, y="total", data=confidence_df, 
                 color="darkred", marker="o", ax=ax2)
    ax2.set_ylabel("Number of Examples", color="darkred")
    
    # Set labels and title
    ax.set_xlabel("Confidence Range")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy vs. Confidence Level")
    ax.set_ylim(0, 1.1)  # Make room for the labels
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
        print(f"Saved confidence plot to {os.path.join(output_dir, filename)}")
    
    plt.close()


def generate_error_analysis(examples_df, output_dir=None, filename="error_analysis.csv"):
    """Generate error analysis table."""
    if examples_df is None or len(examples_df) == 0:
        print("No example data available")
        return
    
    # Focus on incorrect predictions
    incorrect_df = examples_df[~examples_df["is_correct"]].copy()
    
    if len(incorrect_df) == 0:
        print("No incorrect predictions found")
        return
    
    # Calculate confidence (max probability)
    if "prob_a_normalized" in incorrect_df.columns and "prob_b_normalized" in incorrect_df.columns:
        incorrect_df["confidence"] = incorrect_df.apply(
            lambda row: max(row["prob_a_normalized"], row["prob_b_normalized"]), 
            axis=1
        )
    
    # Sort by confidence (highest confidence errors first)
    incorrect_df = incorrect_df.sort_values("confidence", ascending=False)
    
    # Select columns for output
    cols = ["id", "question", "correct_answer", "incorrect_answer", 
            "correct_option", "model_answer", "confidence"]
    
    # Add category if available
    if "category" in incorrect_df.columns:
        cols.insert(2, "category")
    
    # Select available columns
    available_cols = [col for col in cols if col in incorrect_df.columns]
    error_analysis = incorrect_df[available_cols]
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        error_analysis.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"Saved error analysis to {os.path.join(output_dir, filename)}")
    
    return error_analysis


def generate_summary_report(summary_data, examples_df, category_df=None, confidence_df=None, 
                            output_dir=None, filename="summary_report.txt"):
    """Generate a comprehensive summary report."""
    if summary_data is None:
        print("No summary data available")
        return
    
    # Create summary report text
    report = [
        "=" * 50,
        "TRUTHFULQA MULTIPLE-CHOICE EVALUATION SUMMARY",
        "=" * 50,
        "",
        f"Total examples: {summary_data['total_examples']}",
        f"Correct examples: {summary_data['correct_examples']}",
        f"Accuracy: {summary_data['accuracy']:.4f}",
        ""
    ]
    
    # Add category information if available
    if category_df is not None and len(category_df) > 0:
        report.extend([
            "-" * 50,
            "PERFORMANCE BY CATEGORY (sorted by accuracy)",
            "-" * 50,
            ""
        ])
        
        # Sort categories by accuracy
        sorted_cats = category_df.sort_values("accuracy", ascending=False)
        
        for _, row in sorted_cats.iterrows():
            report.append(f"{row['category']}: {row['accuracy']:.4f} ({row['correct']}/{row['total']})")
        
        report.append("")
    
    # Add confidence analysis if available
    if confidence_df is not None and len(confidence_df) > 0:
        report.extend([
            "-" * 50,
            "ACCURACY BY CONFIDENCE LEVEL",
            "-" * 50,
            ""
        ])
        
        for _, row in confidence_df.iterrows():
            report.append(
                f"Confidence {row['confidence_low']:.2f}-{row['confidence_high']:.2f}: "
                f"{row['accuracy']:.4f} ({row['correct']}/{row['total']})"
            )
        
        report.append("")
    
    # Add error analysis summary if available
    if examples_df is not None:
        incorrect_df = examples_df[~examples_df["is_correct"]]
        report.extend([
            "-" * 50,
            "ERROR ANALYSIS SUMMARY",
            "-" * 50,
            "",
            f"Total errors: {len(incorrect_df)}",
            ""
        ])
        
        # Add category breakdown of errors if available
        if "category" in incorrect_df.columns:
            report.append("Errors by category:")
            error_counts = incorrect_df["category"].value_counts()
            
            for category, count in error_counts.items():
                report.append(f"  {category}: {count}")
    
    # Save report if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write("\n".join(report))
        print(f"Saved summary report to {os.path.join(output_dir, filename)}")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Visualize MC evaluation results")
    parser.add_argument("--examples", type=str, default=None, 
                        help="Path to examples CSV file")
    parser.add_argument("--categories", type=str, default=None, 
                        help="Path to categories CSV file")
    parser.add_argument("--confidence", type=str, default=None, 
                        help="Path to confidence CSV file")
    parser.add_argument("--summary", type=str, default=None, 
                        help="Path to summary JSON file")
    parser.add_argument("--output-dir", type=str, default="plots", 
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Load data
    examples_df = load_evaluation_data(args.examples) if args.examples else None
    category_df = load_evaluation_data(args.categories) if args.categories else None
    confidence_df = load_evaluation_data(args.confidence) if args.confidence else None
    
    # Load summary data
    summary_data = None
    if args.summary:
        try:
            with open(args.summary, "r") as f:
                summary_data = json.load(f)
        except Exception as e:
            print(f"Error loading summary data: {str(e)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    if category_df is not None:
        plot_category_accuracies(category_df, args.output_dir)
    
    if confidence_df is not None:
        plot_confidence_vs_accuracy(confidence_df, args.output_dir)
    
    if examples_df is not None:
        generate_error_analysis(examples_df, args.output_dir)
    
    # Generate summary report
    if summary_data is not None or examples_df is not None:
        report = generate_summary_report(
            summary_data, examples_df, category_df, confidence_df, args.output_dir
        )
        print("\nSummary Report Preview:")
        print("\n".join(report.split("\n")[:15]) + "\n...")


if __name__ == "__main__":
    main()