import pydirectinput
import keyboard
import time


class GameController:
    def __init__(self):
        pydirectinput.PAUSE = 0.05  # Quick response time
        self.current_keys = set()



    def handle_prediction(self, prediction):
        """
        Handle predictions for diagonal and multiple movements.
        prediction[0][0] -> W (forward)
        prediction[0][1] -> S (backward)
        prediction[0][2] -> A (left)
        prediction[0][3] -> D (right)
        """
        w_pred, s_pred, a_pred, d_pred = prediction[0]
        VALID_COMBINATIONS = [
            {'w'},  # Forward
            {'s'},  # Backward
            {'a'},  # Left
            {'d'},  # Right
            {'w', 'a'},  # Forward + Left
            {'w', 'd'},  # Forward + Right
            {'s', 'a'},  # Backward + Left
            {'s', 'd'},  # Backward + Right
        ]

        # Determine the keys to press based on predictions
        keys_to_press = set()
        if w_pred:
            keys_to_press.add('w')
        if s_pred:
            keys_to_press.add('s')
        if a_pred:
            keys_to_press.add('a')
        if d_pred:
            keys_to_press.add('d')

        # Prevent conflicting inputs
        if 'w' in keys_to_press and 's' in keys_to_press:
            keys_to_press.discard('s')
        if 'a' in keys_to_press and 'd' in keys_to_press:
            keys_to_press.discard('d')

        # Check if the combination is valid
        if keys_to_press not in VALID_COMBINATIONS:
            print(f"Invalid combination: {keys_to_press}")
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
        """Press a specific key with logging and optional delay."""
        if key not in self.current_keys:
            print(f"Pressing key: {key}")
            pydirectinput.keyDown(key)
            time.sleep(0.05)  # Delay to register simultaneous inputs

    def release_key(self, key):
        """Release a specific key with logging."""
        if key in self.current_keys:
            print(f"Releasing key: {key}")
            pydirectinput.keyUp(key)

    def release_all_keys(self):
        """Release all keys (useful for cleanup)"""
        for key in self.current_keys:
            self.release_key(key)
        self.current_keys.clear()

    def cleanup(self):
        """Cleanup method to be called when stopping the program"""
        self.release_all_keys()