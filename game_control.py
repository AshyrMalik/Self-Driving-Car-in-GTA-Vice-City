import pydirectinput
import keyboard
import time


class GameController:
    def __init__(self):
        pydirectinput.PAUSE = 0.05  # Quick response time
        self.current_keys = set()

    def handle_prediction(self, prediction):
        w_pred = prediction[0][0].item()
        s_pred = prediction[0][1].item()
        a_pred = prediction[0][2].item()
        d_pred = prediction[0][3].item()

        if w_pred:  # Forward
            self.press_key('w')
        elif s_pred:  # Backward
            self.press_key('s')
        elif a_pred:  # Left
            self.press_key('a')
        elif d_pred:  # Right
            self.press_key('d')
        else:
            self.release_all_keys()
            return

        # Release keys that are no longer needed
        keys_to_release = self.current_keys - keys_to_press
        for key in keys_to_release:
            self.release_key(key)

        # Press new keys
        keys_to_press_new = keys_to_press - self.current_keys
        for key in keys_to_press_new:
            self.press_key(key)

        # Update current keys
        self.current_keys = keys_to_press

        # Debugging: Log the current state
        print(f"Keys to press: {keys_to_press}, Keys to release: {keys_to_release}")

    def press_key(self, key):
        """Press a specific key"""
        pydirectinput.keyDown(key)

    def release_key(self, key):
        """Release a specific key"""
        pydirectinput.keyUp(key)

    def release_all_keys(self):
        """Release all keys (useful for cleanup)"""
        for key in self.current_keys:
            self.release_key(key)
        self.current_keys.clear()

    def cleanup(self):
        """Cleanup method to be called when stopping the program"""
        self.release_all_keys()