import pandas as pd
import chess
import re

# Define the numerical mapping for pieces as per your convention
PIECE_MAPPING = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6 # Black pieces
}

def parse_fen(fen_string):
    """
    Parses a FEN string and extracts board state, game rules, and move counters.

    Args:
        fen_string (str): The FEN string of a chess position.

    Returns:
        list: A list containing 64 board state values, one-hot encoded side to move,
              4 castling rights, en-passant availability, halfmove clock, and fullmove counter.
              Returns default values (zeros/negatives) for invalid FEN strings.
    """
    try:
        board = chess.Board(fen_string)
    except ValueError:
        # If the FEN string is invalid, return default values to prevent script failure.
        print(f"Warning: Invalid FEN string encountered: '{fen_string}'. Returning default values.")
        empty_board_state = [0] * 64
        # Default values for other features, including one-hot encoded side to move (both 0)
        # castling: 0, en_passant_available: 0, halfmove: 0, fullmove: 0
        return empty_board_state + [0, 0, 0, 0, 0, 0, 0, 0, 0] # Added one more 0 for the second side_to_move column

    # 1. Board state (64 columns)
    # The chess.SQUARES tuple provides indices from A1 (0) to H8 (63).
    board_state = [0] * 64
    for i in chess.SQUARES:
        piece = board.piece_at(i)
        if piece:
            # Map the piece symbol (e.g., 'P', 'k') to its numerical representation
            board_state[i] = PIECE_MAPPING.get(piece.symbol(), 0)

    # 2. Side to move (one-hot encoded: is_white_to_move, is_black_to_move)
    is_white_to_move = 1 if board.turn == chess.WHITE else 0
    is_black_to_move = 1 if board.turn == chess.BLACK else 0

    # 3. Castling ability (4 columns: White Kingside, White Queenside, Black Kingside, Black Queenside)
    white_kingside_castling = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    white_queenside_castling = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    black_kingside_castling = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    black_queenside_castling = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

    # 4. En-passant availability (1 if available, 0 if not)
    en_passant_available = 1 if board.ep_square is not None else 0

    # 5. Halfmove clock (number of halfmoves since the last capture or pawn advance)
    halfmove_clock = board.halfmove_clock

    # 6. Fullmove counter (the number of the full moves)
    fullmove_counter = board.fullmove_number

    # Combine all extracted features into a single list
    return (board_state +
            [is_white_to_move, is_black_to_move, # One-hot encoded side to move
             white_kingside_castling, white_queenside_castling,
             black_kingside_castling, black_queenside_castling,
             en_passant_available,
             halfmove_clock,
             fullmove_counter])

def parse_evaluation(eval_string):
    """
    Parses an evaluation string into an integer centipawn value.
    Handles centipawn values and checkmate notations.

    Args:
        eval_string (str): The evaluation string (e.g., '+123', '#+5', '#-3').

    Returns:
        int: The parsed evaluation in centipawns.
             Checkmates are represented by large fixed values (20000 or -20000).
             Returns 0 for unparseable evaluation strings.
    """
    eval_string = str(eval_string).strip() # Ensure it's a string and remove leading/trailing whitespace

    if eval_string.startswith('#'):
        # Checkmate evaluation
        if eval_string.startswith('#+'):
            # White checkmates (e.g., #+5 means white checkmates in 5 moves)
            return 20000 # A large positive number to signify white winning by checkmate
        elif eval_string.startswith('#-'):
            # Black checkmates (e.g., #-3 means black checkmates in 3 moves)
            return -20000 # A large negative number to signify black winning by checkmate
        else:
            # Fallback for unexpected checkmate formats, treat as 0 or handle as an error
            print(f"Warning: Unexpected checkmate format: '{eval_string}'. Returning 0.")
            return 0
    else:
        # Centipawn evaluation (e.g., '+123', '-45')
        try:
            return int(eval_string)
        except ValueError:
            # Handle cases where evaluation is not a valid integer or checkmate format
            print(f"Warning: Could not parse evaluation string: '{eval_string}'. Returning 0.")
            return 0

# Generate column names for the 64 squares (a1, b1, ..., h8)
# This order matches the chess.SQUARES indices (0-63)
SQUARE_COLUMNS = [chess.square_name(i) for i in chess.SQUARES]

# Define all output column names in the desired order
OUTPUT_COLUMNS = (
    SQUARE_COLUMNS +
    ['is_white_to_move', 'is_black_to_move', # New one-hot encoded columns
     'white_kingside_castling', 'white_queenside_castling',
     'black_kingside_castling', 'black_queenside_castling',
     'en_passant_available',
     'halfmove_clock',
     'fullmove_counter',
     'evaluation'] # The final evaluation column
)

# Input and output file names
INPUT_FILE = 'chessData.csv' # Make sure your input CSV file is named this
OUTPUT_FILE = 'processed_chess_positions.csv'

# Chunk size for processing large files
# Adjust this value based on your system's available RAM.
# 100,000 rows is a good starting point for 13 million rows.
CHUNK_SIZE = 100000

def process_chess_data(input_file, output_file, chunk_size):
    """
    Main function to process the chess position data from input_file to output_file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        chunk_size (int): Number of rows to process in each chunk.
    """
    print(f"Starting processing of '{input_file}'...")
    first_chunk = True # Flag to write header only for the first chunk

    # Read the CSV file in chunks to handle large datasets efficiently
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"Processing chunk {i+1} (rows {i * chunk_size + 1} to {(i+1) * chunk_size})...")

        # Apply FEN parsing to the 'FEN' column.
        # .apply(parse_fen) returns a Series where each element is a list of features.
        # .tolist() converts this Series of lists into a list of lists.
        fen_features_list = chunk['FEN'].apply(parse_fen).tolist()
        # Create a DataFrame from the list of lists. This expands the features into new columns.
        fen_df = pd.DataFrame(fen_features_list, index=chunk.index)

        # Apply evaluation parsing to the 'Evaluation' column.
        evaluations = chunk['Evaluation'].apply(parse_evaluation)

        # Combine the FEN features DataFrame with the processed evaluations Series.
        # The 'evaluation' Series is renamed to match the desired output column name.
        processed_chunk = pd.concat([fen_df, evaluations.rename('evaluation')], axis=1)

        # Assign the predefined column names to the processed chunk.
        processed_chunk.columns = OUTPUT_COLUMNS

        # Write the processed chunk to the output CSV file.
        # 'mode='w'' is used for the first chunk to create/overwrite the file and write the header.
        # 'mode='a'' is used for subsequent chunks to append data without writing the header again.
        if first_chunk:
            processed_chunk.to_csv(output_file, index=False, mode='w', header=True)
            first_chunk = False
        else:
            processed_chunk.to_csv(output_file, index=False, mode='a', header=False)

        print(f"Finished processing chunk {i+1}. Total rows processed: {len(processed_chunk) + (i * chunk_size)}")

    print(f"Processing complete. Transformed data saved to '{output_file}'")

# --- How to run the script ---
if __name__ == "__main__":
    # Ensure your input CSV file is in the same directory as this script,
    # or provide the full path to your 'chess_positions.csv' file.
    # The output will be saved as 'processed_chess_positions.csv' in the same directory.
    process_chess_data(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE)
