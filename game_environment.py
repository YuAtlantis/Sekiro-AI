# game_environment.py

import torch
import torch.nn.functional as F
import logging
import threading
import time
from cv.health_posture import extract_health, extract_posture
from cv.ocr_utils import get_remaining_uses
from cv.screen_capture import grab_full_screen, grab_region

logging.basicConfig(level=logging.INFO)


class GameEnvironment:
    def __init__(self, width=128, height=128, episodes=3000):
        self.width = width
        self.height = height
        self.episodes = episodes
        self.regions = {
            'game_window': (220, 145, 800, 530),
            'self_blood': (55, 562, 399, 576),
            'boss_blood': (57, 92, 290, 106),
            'self_posture': (395, 535, 635, 552),
            'boss_posture': (315, 73, 710, 88),
            # 'remaining_uses': (955, 570, 971, 588)
        }
        self.paused = True
        self.manual = False
        self.debugged = False
        self.tool_manager = None
        self.target_step = 0
        self.train_mark = 0
        self.action_space_size = 4
        self.current_remaining_uses = 19
        self.screen_lock = threading.Lock()
        self.full_screen_img = None
        self.capture_thread = threading.Thread(target=self.capture_screen, daemon=True)
        self.capture_thread.start()

    def capture_screen(self):
        """Continuously capture the full screen in a separate thread."""
        while True:
            img = grab_full_screen()
            with self.screen_lock:
                self.full_screen_img = img
            # Frame control
            time.sleep(0.06)

    def grab_screens(self):
        """Extract necessary regions from the captured full screen image."""
        with self.screen_lock:
            if self.full_screen_img is None:
                return None, None
            full_screen_img = self.full_screen_img.copy()
        screens = {key: grab_region(full_screen_img, region) for key, region in self.regions.items()}
        # remaining_uses_img = screens.pop('remaining_uses', None)
        game_window_img = screens.pop('game_window')
        return game_window_img, screens

    @staticmethod
    def extract_features(screens):
        """Extract health and posture features from the captured screens."""
        self_blood, boss_blood = extract_health(
            screens['self_blood'], screens['boss_blood'])
        self_posture, boss_posture = extract_posture(
            screens['self_posture'], screens['boss_posture'])
        return {
            'self_hp': self_blood,
            'boss_hp': boss_blood,
            'self_posture': self_posture,
            'boss_posture': boss_posture
        }

    def resize_screen(self, img):
        """Resize the game_settings window image using PyTorch's interpolate for efficiency."""
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to('cuda')  # Move directly to GPU
        resized_img = F.interpolate(img_tensor, size=(self.height, self.width), mode='bilinear',
                                    align_corners=False)
        resized_img = resized_img.squeeze(0).cpu().numpy()  # Move back to CPU after resizing
        return resized_img

    @staticmethod
    def prepare_state(img):
        """Prepare the state tensor for the DQN agent."""
        img_tensor = torch.from_numpy(img).float() / 255.0  # [C, H, W]
        img_tensor = img_tensor.unsqueeze(0).to('cuda')  # [1, C, H, W]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)

        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.squeeze(0)  # [C, H, W]

        return img_tensor

    def get_action_mask(self):
        """Generate a mask for valid actions based on tool cooldowns."""
        action_mask = [1] * self.action_space_size
        # if self.tool_manager.tools_exhausted:
        #     action_mask[4] = action_mask[5] = action_mask[6] = 0
        # else:
        #     remaining_cooldowns = self.tool_manager.get_remaining_cooldown()
        #     for i in range(3):
        #         if remaining_cooldowns[i] > 0:
        #             action_mask[4 + i] = 0
        return action_mask

    def set_tool_manager(self, tool_manager):
        """Set the tool manager."""
        self.tool_manager = tool_manager

    def update_remaining_uses(self, remaining_uses_img):
        """Update the remaining uses by performing OCR."""
        self.current_remaining_uses = get_remaining_uses(remaining_uses_img, self.current_remaining_uses)
        if hasattr(self.tool_manager, 'remaining_uses'):
            self.tool_manager.remaining_uses = self.current_remaining_uses
