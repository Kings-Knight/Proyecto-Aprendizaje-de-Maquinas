import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Input and output file names
INPUT_FILE = 'processed_chess_positions.csv'
OUTPUT_FILE = 'filtered_balanced_normalized_chess_positions.csv'

def filter_balance_normalize_data(input_file, output_file,
                                  min_move=5, max_move=35,
                                  eval_bin_size=100,
                                  min_positions_per_eval_bin=1000,
                                  eval_capping_threshold=2500): # New parameter for evaluation capping
    """
    Filters, balances, and normalizes the chess position dataset.

    Steps:
    1. Filters positions by fullmove_counter (min_move to max_move).
    2. Separates checkmate evaluations for special handling.
    3. Filters out extreme centipawn evaluations (above/below eval_capping_threshold).
    4. Categorizes positions by side to move and evaluation outcome (winning/losing/neutral).
    5. Balances all these primary categories by downsampling to the smallest group's size.
    6. Balances the distribution of evaluations across specified bins on the already
       downsampled dataset, aiming for a minimum number of positions per bin.
    7. Normalizes the final evaluation column using Min-Max scaling.

    Args:
        input_file (str): Path to the input processed CSV file.
        output_file (str): Path to save the filtered, balanced, and normalized CSV.
        min_move (int): Minimum fullmove_counter to include.
        max_move (int): Maximum fullmove_counter to include.
        eval_bin_size (int): Size of bins for evaluation balancing (e.g., 50 centipawns).
        min_positions_per_eval_bin (int): Target minimum number of positions per evaluation bin.
                                            Bins with fewer positions will keep all their data;
                                            bins with more will be downsampled to this number.
        eval_capping_threshold (int): Absolute threshold for evaluation filtering.
                                      Positions with evaluation > this value or < -this value
                                      will be removed.
    """
    print(f"Starting data processing for '{input_file}'...")

    try:
        # Step 1: Load Data
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. "
              "Please ensure the processed_chess_positions.csv file exists.")
        return
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    # Step 2: Filter by Fullmove Counter
    print(f"Filtering positions between move {min_move} and {max_move}...")
    df_filtered = df[(df['fullmove_counter'] >= min_move) & (df['fullmove_counter'] <= max_move)].copy()
    print(f"After move filter: {len(df_filtered)} positions remaining.")

    if df_filtered.empty:
        print("No positions remaining after move filter. Exiting.")
        return

    # Step 3: Separate Checkmate Evaluations
    df_checkmates = df_filtered[(df_filtered['evaluation'] == 20000) | (df_filtered['evaluation'] == -20000)].copy()
    df_centipawns = df_filtered[(df_filtered['evaluation'] != 20000) & (df_filtered['evaluation'] != -20000)].copy()
    print(f"Separated {len(df_checkmates)} checkmate positions and {len(df_centipawns)} centipawn positions.")

    if df_centipawns.empty:
        print("No centipawn positions remaining after checkmate separation. Exiting.")
        return

    # NEW Step: Filter out extreme centipawn evaluations
    print(f"Filtering out evaluations outside [-{eval_capping_threshold}, {eval_capping_threshold}]...")
    df_centipawns = df_centipawns[
        (df_centipawns['evaluation'] >= -eval_capping_threshold) &
        (df_centipawns['evaluation'] <= eval_capping_threshold)
    ].copy()
    print(f"After extreme evaluation filter: {len(df_centipawns)} centipawn positions remaining.")

    if df_centipawns.empty:
        print("No centipawn positions remaining after extreme evaluation filter. Exiting.")
        return

    # Step 4: Categorize and Balance Primary Groups (Side to Move & Winning/Losing/Neutral)
    print("Balancing primary groups (side to move and winning/losing/neutral)...")

    df_centipawns['win_loss_category'] = 'neutral'
    df_centipawns.loc[df_centipawns['evaluation'] > 0, 'win_loss_category'] = 'white_winning'
    df_centipawns.loc[df_centipawns['evaluation'] < 0, 'win_loss_category'] = 'black_winning'

    grouped = df_centipawns.groupby(['is_white_to_move', 'win_loss_category'])
    min_primary_group_count = float('inf')

    for name, group in grouped:
        if len(group) < min_primary_group_count:
            min_primary_group_count = len(group)
        print(f"  Group {name}: {len(group)} positions")

    if min_primary_group_count == 0:
        print("One or more primary groups are empty. Cannot balance. Exiting.")
        return

    print(f"Minimum positions per primary group for balancing: {min_primary_group_count}")

    balanced_primary_dfs = []
    for name, group in grouped:
        if len(group) > min_primary_group_count:
            balanced_primary_dfs.append(group.sample(n=min_primary_group_count, random_state=42))
        else:
            balanced_primary_dfs.append(group)

    df_balanced_primary = pd.concat(balanced_primary_dfs).reset_index(drop=True)
    df_balanced_primary = df_balanced_primary.drop(columns=['win_loss_category'])
    print(f"After primary group balance: {len(df_balanced_primary)} positions.")

    # Step 5: Balance Evaluation Distribution (Binning)
    print(f"Balancing evaluation distribution with {eval_bin_size}-point bins and aiming for {min_positions_per_eval_bin} positions per bin...")

    min_eval_for_bins = df_balanced_primary['evaluation'].min()
    max_eval_for_bins = df_balanced_primary['evaluation'].max()

    # Create bins ensuring they cover the full range of centipawn evaluations
    bins = np.arange(int(min_eval_for_bins // eval_bin_size) * eval_bin_size,
                     int(max_eval_for_bins // eval_bin_size) * eval_bin_size + eval_bin_size,
                     eval_bin_size)

    df_balanced_primary['eval_bin'] = pd.cut(df_balanced_primary['evaluation'], bins=bins, labels=False, include_lowest=True, right=False)

    bin_counts = df_balanced_primary['eval_bin'].value_counts()
    
    print(f"Target minimum positions per evaluation bin: {min_positions_per_eval_bin}")

    if bin_counts.empty:
        print("No evaluation bins found. Cannot balance distribution. Exiting.")
        return

    balanced_eval_dfs = []
    for bin_label in bin_counts.index:
        bin_df = df_balanced_primary[df_balanced_primary['eval_bin'] == bin_label]
        if len(bin_df) > min_positions_per_eval_bin:
            balanced_eval_dfs.append(bin_df.sample(n=min_positions_per_eval_bin, random_state=42))
        else:
            balanced_eval_dfs.append(bin_df)

    df_final_balanced = pd.concat(balanced_eval_dfs).reset_index(drop=True)
    df_final_balanced = df_final_balanced.drop(columns=['eval_bin'])
    print(f"After evaluation distribution balance: {len(df_final_balanced)} positions.")

    # Step 6: Normalize Evaluations
    print("Normalizing evaluation column...")
    scaler = MinMaxScaler(feature_range=(-1, 1)) # Normalize to range -1 to 1, common for evaluations
    df_final_balanced['evaluation_normalized'] = scaler.fit_transform(df_final_balanced[['evaluation']])

    print(f"Final dataset size: {len(df_final_balanced)} rows.")

    # Save the processed data
    print(f"Saving processed data to '{output_file}'...")
    df_final_balanced.to_csv(output_file, index=False)
    print("Data processing complete.")

# --- How to run the script ---
if __name__ == "__main__":
    # Adjust min_positions_per_eval_bin to control the final dataset size.
    # Higher value = larger dataset (potentially more imbalance in sparse bins)
    # Lower value = smaller dataset (more aggressive balancing)
    # eval_capping_threshold can also be adjusted here.
    filter_balance_normalize_data(INPUT_FILE, OUTPUT_FILE, min_positions_per_eval_bin=1000, eval_capping_threshold=2500)
