# game_state.py

import copy


class GameState:
    def __init__(self, features, state):
        self.current_features = features
        self.next_features = copy.deepcopy(features)
        self.current_state = state
        self.next_state = state.clone()

    def update(self, features, state):
        """Update the state with new features and state."""
        self.current_features = copy.deepcopy(self.next_features)
        self.next_features = copy.deepcopy(features)
        self.current_state = self.next_state.clone()
        self.next_state = state.clone()
