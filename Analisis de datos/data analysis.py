import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input file name for the processed data
INPUT_FILE = 'processed_chess_positions.csv'

# Number of rows to read for plotting.
# Set to None to read the entire dataset for comprehensive statistics and plots.
ROWS_TO_READ = None # Read the entire dataset as requested for statistics

def generate_histograms(input_file, rows_to_read=None):
    """
    Generates and displays histograms for 'fullmove_counter' and 'evaluation'
    from the processed chess data, and prints descriptive statistics.

    Args:
        input_file (str): Path to the processed CSV file.
        rows_to_read (int, optional): Number of rows to read from the CSV.
                                      If None, reads the entire file.
    """
    print(f"Loading data from '{input_file}'...")
    try:
        if rows_to_read:
            df = pd.read_csv(input_file, nrows=rows_to_read)
            print(f"Successfully loaded the first {len(df)} rows.")
        else:
            df = pd.read_csv(input_file)
            print(f"Successfully loaded the entire dataset ({len(df)} rows).")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. "
              "Please ensure the processed_chess_positions.csv file exists.")
        return
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    # Set a style for the plots for better aesthetics
    sns.set_style("whitegrid")

    # --- Histogram for Fullmove Counter ---
    plt.figure(figsize=(10, 6)) # Set figure size for better readability
    # On the X axis is the move and on the Y axis is the amount of positions
    sns.histplot(df['fullmove_counter'], bins=df['fullmove_counter'].max(), kde=False, color='skyblue')
    plt.title('Distribution of Positions Per Move', fontsize=16)
    plt.xlabel('Move', fontsize=12) # Changed X-axis label
    plt.ylabel('Amount of Positions', fontsize=12) # Changed Y-axis label
    plt.grid(axis='y', alpha=0.75) # Enhance grid visibility
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    # --- Histogram for Evaluation ---
    plt.figure(figsize=(10, 6)) # Set figure size
    # Histogram with bins of 50 points of evaluation expanding the whole dataset
    # Determine the min and max evaluation to set appropriate bins
    min_eval = df['evaluation'].min()
    max_eval = df['evaluation'].max()
    # Ensure bins cover the full range with 50-point intervals
    bins = range(int(min_eval // 50) * 50, int(max_eval // 50) * 50 + 51, 50)
    sns.histplot(df['evaluation'], bins=bins, kde=False, color='lightcoral') # Changed bins to 50-point intervals, removed KDE
    plt.title('Distribution of Position Evaluations (Centipawns)', fontsize=16)
    plt.xlabel('Evaluation (Centipawns)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    # Display both plots
    plt.show()
    print("Histograms generated successfully.")

    # --- Descriptive Statistics ---
    print("\n--- Descriptive Statistics ---")

    # Number of positions that have the en-passant rule being present
    en_passant_present_count = df['en_passant_available'].sum()
    print(f"Number of positions with en-passant rule present: {en_passant_present_count}")

    # Total number of positions for black (black to move)
    # 'is_black_to_move' column will be 1 if black is to move
    black_to_move_count = df['is_black_to_move'].sum()
    print(f"Total number of positions where black is to move: {black_to_move_count}")

    # Total number of positions for white (white to move)
    # 'is_white_to_move' column will be 1 if white is to move
    white_to_move_count = df['is_white_to_move'].sum()
    print(f"Total number of positions where white is to move: {white_to_move_count}")

    # Number of positions with neutral evaluation (evaluation == 0)
    neutral_positions_count = df[df['evaluation'] == 0].shape[0]
    print(f"Number of neutral positions (evaluation == 0): {neutral_positions_count}")

    # Filter positions based on side to move
    df_white_to_move = df[df['is_white_to_move'] == 1]
    df_black_to_move = df[df['is_black_to_move'] == 1]

    # White move positions good for white (evaluation > 0)
    white_move_good_for_white = df_white_to_move[df_white_to_move['evaluation'] > 0].shape[0]
    print(f"White move positions good for white (evaluation > 0): {white_move_good_for_white}")

    # White move positions good for black (evaluation < 0)
    white_move_good_for_black = df_white_to_move[df_white_to_move['evaluation'] < 0].shape[0]
    print(f"White move positions good for black (evaluation < 0): {white_move_good_for_black}")

    # Black move positions good for white (evaluation > 0)
    black_move_good_for_white = df_black_to_move[df_black_to_move['evaluation'] > 0].shape[0]
    print(f"Black move positions good for white (evaluation > 0): {black_move_good_for_white}")

    # Black move positions good for black (evaluation < 0)
    black_move_good_for_black = df_black_to_move[df_black_to_move['evaluation'] < 0].shape[0]
    print(f"Black move positions good for black (evaluation < 0): {black_move_good_for_black}")

# --- How to run the script ---
if __name__ == "__main__":
    # To plot from the entire dataset and calculate statistics:
    generate_histograms(INPUT_FILE, ROWS_TO_READ)
