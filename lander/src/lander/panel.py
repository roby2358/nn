"""Side panel widgets: buttons, the training chart, and the stats rows."""
import numpy as np
import pygame

from lander.ai.train import RESULTS, Stats

PANEL_BG = (24, 24, 36)
CHART_BG = (12, 12, 22)
GRIDLINE = (50, 50, 64)
TEXT = (230, 230, 230)
MUTED = (150, 150, 165)
BTN = (58, 58, 74)
BTN_HOVER = (76, 76, 96)
BTN_ACTIVE = (70, 130, 95)
BTN_DISABLED = (40, 40, 52)
GREEDY_LINE = (250, 210, 80)
ROW_H = 22

# Stacking order bottom to top: success at the base, failures above.
RESULT_COLORS = {
    "landed": (60, 200, 110),
    "crashed": (220, 80, 80),
    "lost": (110, 120, 200),
    "timeout": (110, 110, 122),
}


class Button:
    def __init__(self, label: str, rect: pygame.Rect) -> None:
        self.label = label
        self.rect = rect
        self.enabled = True
        self.active = False

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        color = BTN_DISABLED if not self.enabled else BTN_ACTIVE if self.active else BTN_HOVER if hover else BTN
        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        t = font.render(self.label, True, TEXT if self.enabled else MUTED)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


def button_row(labels: list[str], x: int, y: int, width: int, height: int, gap: int) -> dict[str, Button]:
    w = (width - gap * (len(labels) - 1)) // len(labels)
    return {name: Button(name, pygame.Rect(x + i * (w + gap), y, w, height)) for i, name in enumerate(labels)}


def window_means(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of each [edges[i], edges[i+1]) window, ignoring NaNs (NaN if a window has none)."""
    out = np.full(len(edges) - 1, np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        seg = values[a:b]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def draw_chart(surf: pygame.Surface, font: pygame.font.Font, rect: pygame.Rect, history: list[Stats]) -> None:
    """Stacked result fractions per batch, one column per pixel, plus the greedy landing line."""
    pygame.draw.rect(surf, CHART_BG, rect, border_radius=6)
    surf.blit(font.render("Training rounds per batch", True, MUTED), (rect.x, rect.y - 20))
    if len(history) < 2:
        t = font.render("no training yet", True, MUTED)
        surf.blit(t, t.get_rect(center=rect.center))
        return
    cols = min(rect.w, len(history))
    edges = np.linspace(0, len(history), cols + 1).astype(int)
    stacks = {r: window_means(np.array([h.rates[r] for h in history]), edges) for r in RESULTS}
    px = rect.w / cols
    for i in range(cols):
        x0 = int(rect.x + i * px)
        x1 = max(x0 + 1, int(rect.x + (i + 1) * px))
        base = rect.bottom
        for r in RESULTS:
            h = int(round(stacks[r][i] * rect.h))
            pygame.draw.rect(surf, RESULT_COLORS[r], (x0, base - h, x1 - x0, h))
            base -= h
    for f in (0.25, 0.5, 0.75):
        y = rect.bottom - int(f * rect.h)
        pygame.draw.line(surf, GRIDLINE, (rect.x, y), (rect.right, y), 1)
    greedy = window_means(np.array([np.nan if h.greedy_landed is None else h.greedy_landed for h in history]), edges)
    pts = [(int(rect.x + (i + 0.5) * px), rect.bottom - int(g * rect.h)) for i, g in enumerate(greedy) if not np.isnan(g)]
    if len(pts) >= 2:
        pygame.draw.lines(surf, GREEDY_LINE, False, pts, 2)


def pct(v: float | None) -> str:
    return "  --" if v is None else f"{v:4.0%}"


def last_greedy(history: list[Stats]) -> float | None:
    measured = [h.greedy_landed for h in history if h.greedy_landed is not None]
    return measured[-1] if measured else None


def draw_stats(surf: pygame.Surface, fonts, rect: pygame.Rect, history: list[Stats],
               extra_rows: list[tuple[str, str]]) -> int:
    """Legend with latest values, then label/value rows. Returns the y just below."""
    font, mono = fonts
    last = history[-1] if history else None
    y = rect.y

    def row(color, label: str, value: str, x: int, right: int) -> None:
        if color:
            pygame.draw.rect(surf, color, (x, y + 5, 12, 12), border_radius=2)
        surf.blit(font.render(label, True, TEXT if color else MUTED), (x + (18 if color else 0), y + 3))
        t = mono.render(value, True, TEXT)
        surf.blit(t, (right - t.get_width(), y + 3))

    half = rect.w // 2
    for i, r in enumerate(RESULTS):
        x = rect.x + (i % 2) * half
        row(RESULT_COLORS[r], r, pct(last.rates[r] if last else None), x, x + half - 16)
        if i % 2:
            y += ROW_H
    row(GREEDY_LINE, "greedy landing rate", pct(last_greedy(history)), rect.x, rect.right)
    y += ROW_H
    for label, value in extra_rows:
        row(None, label, value, rect.x, rect.right)
        y += ROW_H
    return y
