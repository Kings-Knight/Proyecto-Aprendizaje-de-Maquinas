import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input file name for the filtered, balanced, and normalized data
INPUT_FILE = 'filtered_balanced_normalized_chess_positions.csv'

def analyze_balanced_data(input_file):
    """
    Loads the balanced and normalized chess data, prints descriptive statistics,
    and generates a histogram for the normalized evaluations.

    Args:
        input_file (str): Path to the filtered, balanced, and normalized CSV file.
    """
    print(f"Loading balanced data from '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} rows from the balanced dataset.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. "
              "Please ensure the filtered_balanced_normalized_chess_positions.csv file exists.")
        return
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    print("\n--- Descriptive Statistics for Balanced Data ---")

    # Total number of positions
    total_positions = len(df)
    print(f"Total number of positions in the balanced dataset: {total_positions}")

    # Statistics for Normalized Evaluation
    print("\nStatistics for Normalized Evaluation ('evaluation_normalized'):")
    print(df['evaluation_normalized'].describe())

    # Count of positions for white to move vs. black to move
    white_to_move_count = df['is_white_to_move'].sum()
    black_to_move_count = df['is_black_to_move'].sum()
    print(f"\nPositions where White is to move: {white_to_move_count}")
    print(f"Positions where Black is to move: {black_to_move_count}")

    # Count of positions where white is winning, black is winning, and neutral
    # Using the original 'evaluation' column for this categorization before normalization
    white_winning_count = df[df['evaluation'] > 0].shape[0]
    black_winning_count = df[df['evaluation'] < 0].shape[0]
    neutral_count = df[df['evaluation'] == 0].shape[0]
    print(f"\nPositions where White is winning (evaluation > 0): {white_winning_count}")
    print(f"Positions where Black is winning (evaluation < 0): {black_winning_count}")
    print(f"Positions with Neutral evaluation (evaluation == 0): {neutral_count}")

    # --- Histogram for Normalized Evaluation ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    # Using many bins to clearly see the distribution after bin balancing
    sns.histplot(df['evaluation_normalized'], bins=50, kde=False, color='green')
    plt.title('Distribution of Normalized Position Evaluations', fontsize=16)
    plt.xlabel('Normalized Evaluation (-1 to 1)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()
    print("\nHistogram for normalized evaluations generated successfully.")

    # --- Histogram for Fullmove Counter (after filtering) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df['fullmove_counter'], bins=df['fullmove_counter'].max() - df['fullmove_counter'].min() + 1, kde=False, color='purple')
    plt.title('Distribution of Positions Per Move (Filtered)', fontsize=16)
    plt.xlabel('Move', fontsize=12)
    plt.ylabel('Amount of Positions', fontsize=12)
    plt.xticks(range(df['fullmove_counter'].min(), df['fullmove_counter'].max() + 1)) # Ensure integer ticks
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()
    print("Histogram for fullmove counters generated successfully.")


# --- How to run the script ---
if __name__ == "__main__":
    analyze_balanced_data(INPUT_FILE)
