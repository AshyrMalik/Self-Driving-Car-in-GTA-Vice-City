import time
import keyboard
import cv2
import torch
import numpy as np
from config import LOOP_DELAY, GAME_WINDOW, MODEL_PATH
from model_loader import ModelHandler
from screen_capture import ScreenCapture
from game_control import GameController



class GameAI:
    def __init__(self):
        print("Initializing GameAI...")

        # Initialize components
        try:
            print("Loading model...")
            self.model_handler = ModelHandler()

            print("Setting up screen capture...")
            self.screen_capture = ScreenCapture()

            print("Setting up game controller...")
            self.game_controller = GameController()


        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

        print("Initialization complete!")

    def run(self):
        print("\nStarting game AI...")
        print("Controls:")
        print("- Press 'Q' to quit")
        print("- Press 'P' to pause/unpause")
        print("- Press 'V' to toggle visualization")

        paused = False
        show_viz = True
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                # Handle pause
                if keyboard.is_pressed('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                    time.sleep(0.3)  # Debounce

                # Handle visualization toggle
                if keyboard.is_pressed('v'):
                    show_viz = not show_viz
                    print("Visualization: " + ("On" if show_viz else "Off"))
                    time.sleep(0.3)  # Debounce

                # Exit condition
                if keyboard.is_pressed('q'):
                    print("Stopping...")
                    break

                if not paused:
                    # Capture and process frame
                    frame = self.screen_capture.capture_frame()

                    # Get model predictions
                    try:
                        predictions = torch.tensor([[True, False, True, False]]) # Force w and left
                        # Log the forced predictions
                        print(f"Forced predictions: {predictions}")
                        self.game_controller.handle_prediction(predictions)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        continue

                    # Control game
                    try:
                        self.game_controller.handle_prediction(predictions)
                    except Exception as e:
                        print(f"Control error: {e}")
                        continue

                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        print(f"FPS: {fps:.2f}")

                # Small delay to prevent excessive CPU usage
                time.sleep(LOOP_DELAY)

        except Exception as e:
            print(f"Runtime error: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        try:
            self.game_controller.cleanup()  # Release all keys
            cv2.destroyAllWindows()  # Close any open windows
        except Exception as e:
            print(f"Cleanup error: {e}")
        print("Cleanup complete!")


def main():
    try:
        game_ai = GameAI()
        game_ai.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nCritical error: {e}")
    finally:
        print("\nExiting program")


if __name__ == "__main__":
    main()