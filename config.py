# Game window coordinates
import pygetwindow as gw

def get_game_window_dimensions(window_title):
    """
    Fetches the dimensions of a window with a given title.

    Args:
        window_title (str): The title of the window to search for.

    Returns:
        dict: A dictionary with 'top', 'left', 'width', and 'height' of the window.
    """
    try:
        # Find the game window by its title
        windows = [win for win in gw.getWindowsWithTitle(window_title) if win.title]
        if not windows:
            raise ValueError(f"No window found with title: {window_title}")

        # Get the first matching window
        game_window = windows[0]
        dimensions = {
            "top": game_window.top,
            "left": game_window.left,
            "width": game_window.width,
            "height": game_window.height,
        }
        print(f"Game window dimensions: {dimensions}")
        return dimensions
    except Exception as e:
        print(f"Error fetching game window dimensions: {e}")
        return None

# Use the window title for GTA: San Andreas
GAME_WINDOW_TITLE = "GTA: San Andreas"
GAME_WINDOW = get_game_window_dimensions(GAME_WINDOW_TITLE)

if GAME_WINDOW:
    print(f"Captured Game Window: {GAME_WINDOW}")
else:
    print("Failed to fetch game window dimensions. Ensure the game is running and the title is correct.")

# Model settings
MODEL_PATH = 'models/resnet50_best.pth'
MODEL_NAME = 'resnet50'
NUM_CLASSES = 4

# Other settings
PREDICTION_THRESHOLD = 9.9792e-010
LOOP_DELAY = 0.1