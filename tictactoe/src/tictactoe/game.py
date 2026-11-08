"""Pygame front end: board on the left, training controls and live chart on the right."""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path

import numpy as np
import pygame

from .ai.policy import PolicyNet
from .ai.train import WEIGHTS, BackgroundTrainer, Trainer
from .board import Board

# ---- layout -------------------------------------------------------------------
SIZE = 600                 # board is SIZE x SIZE
PANEL_W = 520              # right panel
STATUS_H = 60
GAP = 20
WIN_W = SIZE + PANEL_W
WIN_H = SIZE + STATUS_H
CELL = SIZE // 3
LINE_W = 6
BTN_H = 44
TRAIN_STEPS_PER_CLICK = 1000   # iteration cap per Train press
AI_DELAY_MS = 350
ROW_H = 22
STATS_H = 5 * ROW_H + 14     # stats block under the chart

# ---- colors (dark surface; X and O are the two poles, draw the neutral midpoint) --
BG = (28, 28, 34)
PANEL_BG = (36, 36, 44)
GRID = (200, 200, 210)
X_COLOR = (235, 90, 90)
O_COLOR = (90, 160, 235)
DRAW_COLOR = (120, 120, 132)
WIN_COLOR = (250, 210, 80)
MINIMAX_COLOR = (250, 210, 80)
TEXT = (230, 230, 230)
MUTED = (150, 150, 160)
BTN = (58, 58, 70)
BTN_HOVER = (76, 76, 92)
BTN_ACTIVE = (90, 140, 110)
BTN_DISABLED = (44, 44, 52)


class Button:
    def __init__(self, label: str, rect: pygame.Rect) -> None:
        self.label = label
        self.rect = rect
        self.enabled = True
        self.active = False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        color = BTN_DISABLED if not self.enabled else BTN_ACTIVE if self.active else BTN_HOVER if hover else BTN
        pygame.draw.rect(screen, color, self.rect, border_radius=6)
        t = font.render(self.label, True, TEXT if self.enabled else MUTED)
        screen.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


class App:
    def __init__(self) -> None:
        # Only what we use: pygame.init() also opens audio, whose teardown can hang
        # on WSLg/PulseAudio and freeze the app on close.
        pygame.display.init()
        pygame.font.init()
        # SDL turns Ctrl-C into a QUIT event; restore the default so Ctrl-C always kills.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Tic-Tac-Toe: self-play")
        self.font = pygame.font.SysFont(None, 34)
        self.small = pygame.font.SysFont(None, 22)
        # Monospace for numbers that update every frame, so they don't jiggle.
        mono = pygame.font.match_font("dejavusansmono,liberationmono,consolas,menlo,couriernew,freemono")
        self.mono = pygame.font.Font(mono, 17) if mono else self.small
        self.mono.set_bold(bool(mono))
        self.clock = pygame.time.Clock()

        self.board = Board()
        self.human = "X"
        self.playing = False          # a game against the AI is in progress
        self.ai_due_at = 0
        self.message = "Press Train to start learning, Play to face the net."

        net = PolicyNet()
        if WEIGHTS.exists():
            try:
                net = PolicyNet.load(WEIGHTS)
            except ValueError as e:
                self.message = f"Not loaded: {WEIGHTS} is from an older encoding."
                print(e)
        self.trainer = Trainer(net=net)
        self.bg = BackgroundTrainer(self.trainer)
        self.bg.start()

        names = ["Play", "Clear", "Train", "Stop", "Save", "Load"]
        w = (PANEL_W - GAP * 2 - 8 * (len(names) - 1)) // len(names)
        self.buttons = {
            n: Button(n, pygame.Rect(SIZE + GAP + i * (w + 8), GAP, w, BTN_H)) for i, n in enumerate(names)
        }
        chart_top = GAP + BTN_H + 30
        self.chart_rect = pygame.Rect(SIZE + GAP, chart_top, PANEL_W - GAP * 2, SIZE - GAP - chart_top - STATS_H)

    # ---- actions ------------------------------------------------------------------
    def on_button(self, name: str) -> None:
        if name == "Play":
            self.board.reset()
            self.playing = True
            self.message = f"You are {self.human}. Press S to swap sides."
            self.schedule_ai()
        elif name == "Clear":
            self.bg.halt()
            with self.trainer.lock:
                self.trainer.net.copy_from(PolicyNet())
            self.trainer.history.clear()
            self.trainer.total_games = 0
            self.bg.steps_done = 0
            self.message = "Fresh random net, history cleared."
        elif name == "Train":
            self.bg.start_run(TRAIN_STEPS_PER_CLICK)
            self.message = f"Training {TRAIN_STEPS_PER_CLICK} steps of {self.trainer.games_per_step} self-play games."
        elif name == "Stop":
            self.bg.halt()
            self.message = "Training stopped."
        elif name == "Save":
            self.trainer.save(WEIGHTS)
            self.message = f"Saved {Path(WEIGHTS).resolve()}"
        elif name == "Load":
            if WEIGHTS.exists():
                try:
                    loaded = PolicyNet.load(WEIGHTS)
                except ValueError as e:
                    print(e)
                    self.message = f"Not loaded: {WEIGHTS} is from an older encoding."
                else:
                    with self.trainer.lock:
                        self.trainer.net.copy_from(loaded)
                    self.message = f"Loaded {WEIGHTS}"
            else:
                self.message = f"No {WEIGHTS} to load."

    def schedule_ai(self) -> None:
        if self.playing and not self.board.game_over and self.board.turn != self.human:
            self.ai_due_at = pygame.time.get_ticks() + AI_DELAY_MS

    def ai_move_if_due(self) -> None:
        if not self.playing or self.board.game_over or self.board.turn == self.human:
            return
        if pygame.time.get_ticks() >= self.ai_due_at:
            with self.trainer.lock:
                move = self.trainer.net.choose(self.board, greedy=True)
            self.board.play(move)

    def click_board(self, pos: tuple[int, int]) -> None:
        if not self.playing or self.board.game_over or self.board.turn != self.human:
            return
        mx, my = pos
        if self.board.play((my // CELL) * 3 + mx // CELL):
            self.schedule_ai()

    # ---- drawing ------------------------------------------------------------------
    def draw_board(self) -> None:
        s, b = self.screen, self.board
        for i in (1, 2):
            pygame.draw.line(s, GRID, (i * CELL, 0), (i * CELL, SIZE), LINE_W)
            pygame.draw.line(s, GRID, (0, i * CELL), (SIZE, i * CELL), LINE_W)
        pad = CELL // 5
        for idx, mark in enumerate(b.cells):
            if mark is None:
                continue
            r, c = divmod(idx, 3)
            x, y = c * CELL, r * CELL
            if mark == "X":
                pygame.draw.line(s, X_COLOR, (x + pad, y + pad), (x + CELL - pad, y + CELL - pad), LINE_W + 4)
                pygame.draw.line(s, X_COLOR, (x + CELL - pad, y + pad), (x + pad, y + CELL - pad), LINE_W + 4)
            else:
                pygame.draw.circle(s, O_COLOR, (x + CELL // 2, y + CELL // 2), CELL // 2 - pad, LINE_W + 4)
        if b.winning_line:
            a, c = b.winning_line[0], b.winning_line[2]
            ra, ca = divmod(a, 3)
            rc, cc = divmod(c, 3)
            pygame.draw.line(s, WIN_COLOR, (ca * CELL + CELL // 2, ra * CELL + CELL // 2),
                             (cc * CELL + CELL // 2, rc * CELL + CELL // 2), LINE_W + 6)

    def draw_chart(self) -> None:
        s, r = self.screen, self.chart_rect
        pygame.draw.rect(s, BG, r, border_radius=6)
        hist = list(self.trainer.history)
        title = self.small.render("Self-play outcomes per batch", True, MUTED)
        s.blit(title, (r.x, r.y - 20))
        if len(hist) < 2:
            t = self.small.render("no training yet", True, MUTED)
            s.blit(t, t.get_rect(center=r.center))
            return
        # Downsample history into one column per pixel, averaging each window.
        cols = min(r.w, len(hist))
        idx = np.linspace(0, len(hist), cols + 1).astype(int)
        xw = np.array([h.x_win for h in hist]); dr = np.array([h.draw for h in hist]); ow = np.array([h.o_win for h in hist])
        mm = np.array([np.nan if h.vs_minimax is None else h.vs_minimax for h in hist])
        xs = np.array([xw[a:b].mean() for a, b in zip(idx[:-1], idx[1:])])
        ds = np.array([dr[a:b].mean() for a, b in zip(idx[:-1], idx[1:])])
        os_ = np.array([ow[a:b].mean() for a, b in zip(idx[:-1], idx[1:])])
        px = r.w / cols
        for i in range(cols):
            x0 = int(r.x + i * px); x1 = max(x0 + 1, int(r.x + (i + 1) * px))
            yx = r.bottom - int(xs[i] * r.h)
            yd = yx - int(ds[i] * r.h)
            pygame.draw.rect(s, X_COLOR, (x0, yx, x1 - x0, r.bottom - yx))
            pygame.draw.rect(s, DRAW_COLOR, (x0, yd, x1 - x0, yx - yd))
            pygame.draw.rect(s, O_COLOR, (x0, r.y, x1 - x0, max(0, yd - r.y)))
        # Recessive grid at 25/50/75 percent.
        for f in (0.25, 0.5, 0.75):
            y = r.bottom - int(f * r.h)
            pygame.draw.line(s, (60, 60, 70), (r.x, y), (r.right, y), 1)
        # Thin line: greedy policy's draw rate vs perfect play (measured every 10 steps).
        pts = []
        for i, (a, b) in enumerate(zip(idx[:-1], idx[1:])):
            seg = mm[a:b]
            seg = seg[~np.isnan(seg)]
            if seg.size:
                pts.append((int(r.x + (i + 0.5) * px), r.bottom - int(seg.mean() * r.h)))
        if len(pts) >= 2:
            pygame.draw.lines(s, MINIMAX_COLOR, False, pts, 2)
        self.draw_stats()

    def draw_stats(self) -> None:
        """Legend and numbers under the chart. Labels left, values right-aligned in mono."""
        s, r = self.screen, self.chart_rect
        hist = self.trainer.history
        last = hist[-1] if hist else None

        def pct(v: float | None) -> str:
            return " --" if v is None else f"{v:4.0%}"

        def swatch(color, x, y):
            pygame.draw.rect(s, color, (x, y + 5, 12, 12), border_radius=2)

        def value(text: str, right: int, y: int) -> None:
            t = self.mono.render(text, True, TEXT)
            s.blit(t, (right - t.get_width(), y + (ROW_H - t.get_height()) // 2))

        def label(text: str, x: int, y: int, color=TEXT) -> None:
            t = self.small.render(text, True, color)
            s.blit(t, (x, y + (ROW_H - t.get_height()) // 2))

        y = r.bottom + 10
        # Row 1: the three self-play outcomes side by side, in stacking order.
        col = r.w // 3
        outcomes = [(X_COLOR, "X wins", last and last.x_win), (DRAW_COLOR, "draws", last and last.draw),
                    (O_COLOR, "O wins", last and last.o_win)]
        for i, (color, name, v) in enumerate(outcomes):
            x = r.x + i * col
            swatch(color, x, y)
            label(name, x + 18, y)
            value(pct(v if last else None), x + col - 16, y)
        y += ROW_H

        # Rows 2-5: one stat per line.
        running = self.bg.running
        steps = f"{self.bg.steps_done:,}" + (f"  ({self.bg.remaining:,} left)" if running else "")
        rows = [
            (MINIMAX_COLOR, "draw rate vs perfect play", pct(last.vs_minimax if last else None)),
            (None, "illegal picks", pct(last.invalid if last else None)),
            (None, "self-play games played", f"{self.trainer.total_games:,}"),
            (None, "training steps" + ("  (running)" if running else ""), steps),
        ]
        for color, name, v in rows:
            if color:
                swatch(color, r.x, y)
            label(name, r.x + 18, y, TEXT if color else MUTED)
            value(v, r.right, y)
            y += ROW_H

    def draw_status(self) -> None:
        b = self.board
        if self.playing:
            if b.winner:
                msg = "You win!" if b.winner == self.human else "The net wins."
                msg += "  Press Play for another."
            elif b.is_draw:
                msg = "Draw.  Press Play for another."
            else:
                msg = "Your move." if b.turn == self.human else "Net is thinking..."
        else:
            msg = self.message
        t = self.font.render(msg, True, TEXT)
        self.screen.blit(t, (GAP, SIZE + (STATUS_H - t.get_height()) // 2))

    def draw(self) -> None:
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, PANEL_BG, (SIZE, 0, PANEL_W, SIZE))
        self.draw_board()
        self.buttons["Train"].active = self.bg.running
        self.buttons["Train"].enabled = not self.bg.running
        self.buttons["Stop"].enabled = self.bg.running
        for btn in self.buttons.values():
            btn.draw(self.screen, self.small)
        self.draw_chart()
        self.draw_status()

    # ---- main loop ----------------------------------------------------------------
    def quit(self) -> None:
        self.bg.shutdown()
        # If SDL teardown hangs anyway, force the exit after a moment.
        threading.Timer(2.0, os._exit, args=(0,)).start()
        pygame.quit()
        os._exit(0)

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self.quit()
                    elif event.key == pygame.K_s:
                        self.human = "O" if self.human == "X" else "X"
                        self.on_button("Play")
                    elif event.key == pygame.K_r:
                        self.on_button("Play")
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if event.pos[0] < SIZE and event.pos[1] < SIZE:
                        self.click_board(event.pos)
                    else:
                        for name, btn in self.buttons.items():
                            if btn.hit(event.pos):
                                self.on_button(name)
            self.ai_move_if_due()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
