import asyncio
import sys

import numpy as np
import pygame
from pygame.locals import K_ESCAPE, KEYDOWN, QUIT

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window
from .utils.webcam_pose import WebcamPose

# Game area size (original Flappy Bird resolution)
GAME_WIDTH = 288
GAME_HEIGHT = 512
# Webcam panel size (right side of window)
WEBCAM_WIDTH = 256
WEBCAM_HEIGHT = 192
TOTAL_WIDTH = GAME_WIDTH + WEBCAM_WIDTH
TOTAL_HEIGHT = GAME_HEIGHT
DISPLAY_SCALE = 2


class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird - Flap with your arms!")
        window = Window(GAME_WIDTH, GAME_HEIGHT)
        # Render at original resolution, then scale up for readability.
        self.display_screen = pygame.display.set_mode(
            (TOTAL_WIDTH * DISPLAY_SCALE, TOTAL_HEIGHT * DISPLAY_SCALE)
        )
        self.full_screen = pygame.Surface((TOTAL_WIDTH, TOTAL_HEIGHT))
        game_surface = self.full_screen.subsurface((0, 0, GAME_WIDTH, GAME_HEIGHT))
        images = Images()

        self.config = GameConfig(
            screen=game_surface,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )
        self.webcam = WebcamPose(width=WEBCAM_WIDTH, height=WEBCAM_HEIGHT)
        self.webcam.start()

    def _present(self) -> None:
        scaled = pygame.transform.scale(
            self.full_screen, (TOTAL_WIDTH * DISPLAY_SCALE, TOTAL_HEIGHT * DISPLAY_SCALE)
        )
        self.display_screen.blit(scaled, (0, 0))
        pygame.display.update()

    async def start(self):
        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)
            await self.splash()
            await self.play()
            await self.game_over()

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
            if self.webcam.consume_flap():
                return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()
            self._draw_webcam()
            self._present()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            self.webcam.stop()
            pygame.quit()
            sys.exit()

    def _draw_webcam(self) -> None:
        """Blit the latest webcam frame (with skeleton) onto the right panel."""
        panel_rect = (GAME_WIDTH, 0, WEBCAM_WIDTH, WEBCAM_HEIGHT)
        frame = self.webcam.get_latest_frame()
        if frame is not None:
            # OpenCV is BGR; convert to RGB and create pygame surface
            frame_rgb = np.ascontiguousarray(frame[:, :, ::-1])
            surf = pygame.image.frombuffer(
                frame_rgb.tobytes(), (WEBCAM_WIDTH, WEBCAM_HEIGHT), "RGB"
            )
            self.full_screen.blit(surf, (GAME_WIDTH, 0))
        else:
            self.full_screen.fill((40, 40, 40), panel_rect)
        # Border and instruction
        pygame.draw.rect(self.full_screen, (80, 80, 80), panel_rect, 2)
        font = pygame.font.SysFont("Arial", 14, bold=True)
        label = font.render("Flap arms to fly!", True, (255, 255, 200))
        self.full_screen.blit(label, (GAME_WIDTH + 8, WEBCAM_HEIGHT - 22))

    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
            if self.webcam.consume_flap():
                self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self._draw_webcam()
            self._present()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
            if self.webcam.consume_flap():
                if self.player.y + self.player.h >= self.floor.y - 1:
                    return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()
            self._draw_webcam()
            self.config.tick()
            self._present()
            await asyncio.sleep(0)
