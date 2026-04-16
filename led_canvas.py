"""
led_canvas.py - Canvas LED virtuale per disegno 8x32
"""

import numpy as np

# Palette di default (sovrascritta da ServerMac)
COLOR_PALETTE = [
    (255, 0, 0),      # Rosso
    (0, 100, 255),    # Blu
    (255, 255, 0),    # Giallo
    (255, 255, 255),  # Bianco
    (0, 0, 0),        # Nero
]
COLOR_NAMES_IT = ["Rosso", "Blu", "Giallo", "Bianco", "Nero"]


class LEDCanvas:
    def __init__(self, width=32, height=8):
        self.width = width
        self.height = height
        self.pixels = np.zeros((height, width, 3), dtype=np.uint8)
        self._color_index = 0
        self.current_color = COLOR_PALETTE[0]
        self.brush_size = 1
        # Stato per-mano: traccia ultima posizione per interpolazione
        self._hand_states = {}

    def clear(self):
        """Azzera tutti i pixel."""
        self.pixels[:] = 0
        for hid in self._hand_states:
            self._hand_states[hid]['prev'] = None

    def get_frame_rgb(self):
        """Ritorna la matrice RGB corrente (h, w, 3)."""
        return self.pixels

    def get_color_index(self):
        return self._color_index

    def get_color_name(self):
        return COLOR_NAMES_IT[self._color_index]

    def set_color_by_index(self, idx):
        self._color_index = idx % len(COLOR_PALETTE)
        self.current_color = COLOR_PALETTE[self._color_index]

    def set_brush_size(self, size):
        self.brush_size = max(1, min(5, size))

    def draw_at(self, gx, gy, drawing, hand_id="default", is_erasing=False):
        """Disegna o cancella al pixel (gx, gy) con interpolazione tra frame."""
        if hand_id not in self._hand_states:
            self._hand_states[hand_id] = {'prev': None}

        if not drawing:
            self._hand_states[hand_id]['prev'] = None
            return

        color = (0, 0, 0) if is_erasing else self.current_color
        prev = self._hand_states[hand_id]['prev']

        if prev is not None:
            # Interpolazione Bresenham tra punto precedente e attuale
            points = self._bresenham(prev[0], prev[1], gx, gy)
        else:
            points = [(gx, gy)]

        for px, py in points:
            self._paint(px, py, color)

        self._hand_states[hand_id]['prev'] = (gx, gy)

    def _paint(self, cx, cy, color):
        """Disegna un punto col brush_size corrente."""
        r = self.brush_size // 2
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.pixels[ny, nx] = color

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        """Linea Bresenham tra due punti."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points
