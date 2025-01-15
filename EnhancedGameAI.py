import time
import keyboard
import cv2
from collections import deque
from config import LOOP_DELAY, GAME_WINDOW, MODEL_PATH
from screen_capture import ScreenCapture
from game_control import GameController
from Enhanced_Model_Handler import EnhancedModelHandler

class EnhancedGameAI:
    def __init__(self):
        print("Initializing Enhanced GameAI...")

        # Initialize performance monitoring
        self.fps_queue = deque(maxlen=30)
        self.last_fps_print = time.time()
        self.fps_print_interval = 1.0  # Print FPS every second

        try:
            print("Loading enhanced model handler...")
            self.model_handler = EnhancedModelHandler()  # Using the enhanced version

            print("Setting up screen capture...")
            self.screen_capture = ScreenCapture()

            print("Setting up game controller...")
            self.game_controller = GameController()

            # Initialize debug window
            cv2.namedWindow('Debug View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Debug View', 800, 600)

        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

        print("Initialization complete!")

    def _draw_visualization(self, frame, prediction_result, show_debug=False):
        """Draw visualization overlay on frame"""
        viz_frame = frame.copy()

        # Draw lane detection
        if 'lane_info' in prediction_result:
            lanes = prediction_result['lane_info']
            if lanes:
                if lanes['left'] is not None:
                    pt1 = (lanes['left'][0], lanes['left'][1])
                    pt2 = (lanes['left'][2], lanes['left'][3])
                    cv2.line(viz_frame, pt1, pt2, (0, 255, 0), 2)

                if lanes['right'] is not None:
                    pt1 = (lanes['right'][0], lanes['right'][1])
                    pt2 = (lanes['right'][2], lanes['right'][3])
                    cv2.line(viz_frame, pt1, pt2, (0, 255, 0), 2)

        # Draw additional debug info if requested
        if show_debug:
            # Draw predictions
            if 'prediction' in prediction_result:
                pred = prediction_result['prediction'].cpu().numpy()[0]
                actions = ['W', 'A', 'S', 'D']  # Adjust based on your action mapping
                for i, (action, value) in enumerate(zip(actions, pred)):
                    text = f"{action}: {value:.2f}"
                    cv2.putText(viz_frame, text, (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw FPS
            if self.fps_queue:
                avg_fps = sum(self.fps_queue) / len(self.fps_queue)
                cv2.putText(viz_frame, f"FPS: {avg_fps:.1f}",
                            (10, viz_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return viz_frame

    def _update_fps(self, start_time):
        """Update FPS calculations"""
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        self.fps_queue.append(fps)

        # Print FPS at regular intervals
        current_time = time.time()
        if current_time - self.last_fps_print >= self.fps_print_interval:
            avg_fps = sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0
            print(f"Average FPS: {avg_fps:.2f}")
            self.last_fps_print = current_time

    def run(self):
        print("\nStarting enhanced game AI...")
        print("Controls:")
        print("- Press 'Q' to quit")
        print("- Press 'P' to pause/unpause")
        print("- Press 'V' to toggle visualization")
        print("- Press 'D' to toggle debug info")

        paused = False
        show_viz = True
        show_debug = False

        try:
            while True:
                frame_start = time.time()

                # Handle controls
                if keyboard.is_pressed('q'):
                    print("Stopping...")
                    break

                if keyboard.is_pressed('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                    time.sleep(0.3)

                if keyboard.is_pressed('v'):
                    show_viz = not show_viz
                    print("Visualization: " + ("On" if show_viz else "Off"))
                    time.sleep(0.3)

                if keyboard.is_pressed('d'):
                    show_debug = not show_debug
                    print("Debug info: " + ("On" if show_debug else "Off"))
                    time.sleep(0.3)

                if not paused:
                    # Capture and process frame
                    frame = self.screen_capture.capture_frame()

                    try:
                        # Get enhanced predictions with lane detection
                        prediction_result = self.model_handler.predict(frame)

                        # Update controls based on combined predictions
                        self.game_controller.handle_prediction(prediction_result['prediction'])

                        # Visualization
                        if show_viz:
                            viz_frame = self._draw_visualization(frame, prediction_result, show_debug)
                            cv2.imshow('Debug View', viz_frame)
                            cv2.waitKey(1)

                    except Exception as e:
                        print(f"Processing error: {e}")
                        continue

                    # Update FPS
                    self._update_fps(frame_start)

                # Adaptive delay to maintain consistent frame rate
                elapsed = time.time() - frame_start
                sleep_time = max(0, LOOP_DELAY - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"Runtime error: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        try:
            self.game_controller.cleanup()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Cleanup error: {e}")
        print("Cleanup complete!")


def main():
    try:
        game_ai = EnhancedGameAI()
        game_ai.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nCritical error: {e}")
    finally:
        print("\nExiting program")


if __name__ == "__main__":
    main()