import cv2
import numpy as np
import time
import os
import subprocess
import random
from datetime import datetime

from hand_tracker import HandTracker, HandState
from led_canvas import LEDCanvas
from audio_synth import AudioSynth

# 5 colori: Rosso, Blu, Giallo, Bianco, Nero
COLOR_PALETTE = [
    (255, 0, 0),      # 1 — Rosso
    (0, 100, 255),    # 2 — Blu
    (255, 255, 0),    # 3 — Giallo
    (255, 255, 255),  # 4 — Bianco
    (0, 0, 0),        # 5 — Nero
]
COLOR_NAMES_IT = ["Rosso", "Blu", "Giallo", "Bianco", "Nero"]

# ============================================================
# CONFIGURAZIONE RASPBERRY (SCP)
# ============================================================
RASP_USER = "pit"
RASP_IP   = "Webcamproject1-ui.local"
RASP_PROJECT = "/home/pit/Desktop/LedFiume"

# ============================================================
# CONFIGURAZIONE CANVAS
# ============================================================
CANVAS_W = 32
CANVAS_H = 8

# ============================================================
# CONFIGURAZIONE WEBCAM
# ============================================================
DRAWING_CAM_INDEX = 0   # webcam Mac per disegnare

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_OUTBOX = os.path.join(BASE_DIR, 'OUTBOX')

_color_counters = {'RED': 0, 'BLUE': 0, 'YELLOW': 0}

# ============================================================
# UNDO
# ============================================================
MAX_UNDO = 20
_undo_stack = []

def push_undo(canvas_led):
    """Salva snapshot del canvas nello stack undo."""
    _undo_stack.append(canvas_led.pixels.copy())
    if len(_undo_stack) > MAX_UNDO:
        _undo_stack.pop(0)

def pop_undo(canvas_led):
    """Ripristina l'ultimo snapshot dallo stack undo."""
    if not _undo_stack:
        print("[UNDO] Niente da annullare")
        return
    canvas_led.pixels[:] = _undo_stack.pop()
    print("[UNDO] Ripristinato")

# ============================================================
# FLOOD FILL
# ============================================================
def flood_fill(canvas_led, x, y, new_color):
    """Flood fill 4-connesso a partire da (x, y) col colore new_color."""
    pixels = canvas_led.pixels
    target = tuple(pixels[y, x])
    fill = tuple(new_color)
    if target == fill:
        return
    stack = [(x, y)]
    visited = set()
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in visited:
            continue
        if cx < 0 or cx >= canvas_led.width or cy < 0 or cy >= canvas_led.height:
            continue
        if tuple(pixels[cy, cx]) != target:
            continue
        visited.add((cx, cy))
        pixels[cy, cx] = new_color
        stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])

# Mappatura colore dominante -> cartella
# Distanze calcolate in spazio RGB rispetto ai 3 colori target
_FOLDER_COLORS = {
    'RED':    np.array([255, 0, 0]),
    'BLUE':   np.array([0, 100, 255]),
    'YELLOW': np.array([255, 255, 0]),
}


def _detect_dominant_folder(rgb):
    """Rileva il colore dominante (escluso nero) e restituisce RED/BLUE/YELLOW."""
    non_black = np.any(rgb > 0, axis=2)
    if not np.any(non_black):
        return 'RED'  # fallback se tutto nero
    pixels = rgb[non_black].astype(np.float32)  # (N, 3)

    # Conta pixel piu vicini a ciascun colore target
    counts = {}
    for name, target in _FOLDER_COLORS.items():
        dists = np.linalg.norm(pixels - target.astype(np.float32), axis=1)
        counts[name] = int(np.sum(dists < 200))

    # Il colore con piu pixel vicini vince
    chosen = max(counts, key=counts.get)
    print(f"[COLORE DOMINANTE] R={counts['RED']} B={counts['BLUE']} Y={counts['YELLOW']} -> {chosen}")
    return chosen


# ============================================================
# SALVATAGGIO + INVIO SCP
# ============================================================
def save_and_send(canvas_led):
    """Salva il disegno come RGBA PNG e lo invia nella cartella del colore dominante sul Rasp."""
    os.makedirs(LOCAL_OUTBOX, exist_ok=True)

    # Analizza colore dominante
    rgb = canvas_led.get_frame_rgb()  # (8, 32, 3)
    chosen = _detect_dominant_folder(rgb)

    _color_counters[chosen] += 1
    num = _color_counters[chosen]
    filename = f"{chosen}_{num}.png"
    filepath = os.path.join(LOCAL_OUTBOX, filename)

    # Salva con trasparenza (nero = trasparente)
    h, w = rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    non_black = np.any(rgb > 0, axis=2)
    rgba[non_black, 3] = 255

    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    bgra = cv2.flip(bgra, 1)  # Specchia orizzontalmente per allineare ai LED
    cv2.imwrite(filepath, bgra)
    print(f"[SALVA] {filename} -> {chosen}/")

    # Invia via SCP direttamente nella cartella colore sul Rasp
    dest = f"{RASP_USER}@{RASP_IP}:{RASP_PROJECT}/{chosen}/{filename}"
    try:
        result = subprocess.run(
            ["scp", "-o", "ConnectTimeout=5", filepath, dest],
            capture_output=True, timeout=10
        )
        if result.returncode == 0:
            print(f"[SCP] {filename} -> {RASP_IP}:{chosen}/ OK")
        else:
            err = result.stderr.decode().strip()
            print(f"[SCP] ERRORE: {err}")
    except subprocess.TimeoutExpired:
        print("[SCP] Timeout - Raspberry non raggiungibile")
    except Exception as e:
        print(f"[SCP] Errore: {e}")


# ============================================================
# GUI HELPERS
# ============================================================
def _rounded_rect(img, pt1, pt2, color, radius, filled=True, thickness=1):
    """Rettangolo con angoli arrotondati (filled o outline)."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r <= 0:
        if filled:
            cv2.rectangle(img, pt1, pt2, color, -1)
        else:
            cv2.rectangle(img, pt1, pt2, color, thickness)
        return
    t = -1 if filled else thickness
    if filled:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180,  0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270,  0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90,  0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0,  0, 90, color, t, cv2.LINE_AA)


def _put_text_centered(img, text, cx, cy, font, scale, color, thickness=1):
    """Testo centrato orizzontalmente e verticalmente su (cx, cy)."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2),
                font, scale, color, thickness, cv2.LINE_AA)


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 50)
    print("  SERVER MAC - LAVAGNA LED")
    print("  Disegna con le mani, invia al Raspberry con S")
    print("=" * 50)

    # -- Webcam --
    print(f"\n[CAM] Apertura webcam (indice {DRAWING_CAM_INDEX})...")
    cap = cv2.VideoCapture(DRAWING_CAM_INDEX)
    if not cap.isOpened():
        print("[!] Webcam non trovata!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[OK] Webcam pronta")

    # -- Watermark --
    _wm_img = None
    _wm_alpha = None
    _wm_w = 0
    _wm_h = 0
    wm_path = os.path.join(BASE_DIR, 'watermark.png')
    if os.path.exists(wm_path):
        _wm_raw = cv2.imread(wm_path, cv2.IMREAD_UNCHANGED)
        if _wm_raw is not None:
            target_h = 52  # altezza watermark
            scale = target_h / _wm_raw.shape[0]
            _wm_h = target_h
            _wm_w = max(1, int(_wm_raw.shape[1] * scale))
            _wm_raw = cv2.resize(_wm_raw, (_wm_w, _wm_h), interpolation=cv2.INTER_AREA)
            if _wm_raw.shape[2] == 4:
                _wm_img = _wm_raw[:, :, :3]
                _wm_alpha = _wm_raw[:, :, 3:4].astype(np.float32) / 255.0
            else:
                _wm_img = _wm_raw[:, :, :3]
                gray = cv2.cvtColor(_wm_img, cv2.COLOR_BGR2GRAY)
                _wm_alpha = (255 - gray.astype(np.float32))[:, :, np.newaxis] / 255.0

    # -- Hand tracker e canvas --
    synth = AudioSynth()
    tracker = HandTracker(canvas_width=CANVAS_W, canvas_height=CANVAS_H, num_hands=1)
    canvas_led = LEDCanvas(CANVAS_W, CANVAS_H)

    # Forza la palette custom sul canvas
    import led_canvas as _lc
    _lc.COLOR_PALETTE = COLOR_PALETTE
    _lc.COLOR_NAMES_IT = COLOR_NAMES_IT
    canvas_led._color_index = 0
    canvas_led.current_color = COLOR_PALETTE[0]

    # -- Stato --
    last_erase_time = 0.0
    ERASE_COOLDOWN = 1.5
    fill_mode = False
    was_drawing_prev = {}  # per sapere quando un tratto inizia (per push_undo)
    fill_cooldown = 0.0

    print(f"\n[INFO] Destinazione SCP: {RASP_USER}@{RASP_IP}:{RASP_PROJECT}")
    print("\n" + "-" * 50)
    print("  CONTROLLI:")
    print("  [1]     - Tracking 1 mano (default)")
    print("  [2]     - Tracking 2 mani")
    print("  [R/B/Y/W/N] - Colore (Rosso/Blu/Giallo/Bianco/Nero)")
    print("  [+/-]   - Cambia dimensione pennello")
    print("  [F]     - Toggle modalita Fill")
    print("  [Z]     - Undo")
    print("  [C]     - Cancella lavagna")
    print("  [S]     - Salva e invia al Raspberry")
    print("  [Q/ESC] - Esci")
    print("")
    print("  GESTI:")
    print("  Pinch (indice+pollice) = Disegna")
    print("  Pollice in giu = Cancella lavagna")
    print("  Segno della pace (V) = Cambia colore")
    print("-" * 50 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)

            # -- Calcolo griglia (serve per pixel-perfect tracking) --
            fh, fw = frame.shape[:2]
            cell = min(fw // CANVAS_W, fh // CANVAS_H)
            grid_w = cell * CANVAS_W
            grid_h = cell * CANVAS_H
            ox = (fw - grid_w) // 2
            oy = (fh - grid_h) // 2

            # -- Hand tracking --
            hand_states = tracker.process_frame(frame)

            active_ids = {s.hand_label for s in hand_states}
            for hid in list(canvas_led._hand_states.keys()):
                if hid not in active_ids:
                    canvas_led.draw_at(0, 0, False, hand_id=hid)

            is_any_drawing = any(s.drawing for s in hand_states)
            if not is_any_drawing:
                synth.play_note(0, 0, canvas_led.width, canvas_led.height, False)

            is_any_erasing = any(s.precision_erasing for s in hand_states)

            for hand_state in hand_states:
                if (hand_state.thumbs_down and not is_any_drawing
                        and (time.time() - last_erase_time > ERASE_COOLDOWN)):
                    push_undo(canvas_led)
                    canvas_led.clear()
                    last_erase_time = time.time()
                    print("[CANCELLA] Lavagna cancellata!")

            for hand_state in hand_states:
                if hand_state.peace_sign and not is_any_erasing:
                    current_idx = canvas_led.get_color_index()
                    next_idx = (current_idx + 1) % len(COLOR_PALETTE)
                    canvas_led.set_color_by_index(next_idx)
                    print(f"[COLORE] {canvas_led.get_color_name()}")

                # Pixel-perfect: mappa posizione dito sulla griglia
                px = int(hand_state.raw_x * fw)
                py = int(hand_state.raw_y * fh)
                gx = (px - ox) // cell if cell > 0 else -1
                gy = (py - oy) // cell if cell > 0 else -1
                in_grid = 0 <= gx < CANVAS_W and 0 <= gy < CANVAS_H

                hid = hand_state.hand_label

                if hand_state.drawing and in_grid:
                    if fill_mode:
                        # Fill mode: flood fill con cooldown
                        if time.time() - fill_cooldown > 0.5:
                            push_undo(canvas_led)
                            flood_fill(canvas_led, gx, gy, canvas_led.current_color)
                            fill_cooldown = time.time()
                    else:
                        # Undo al primo pixel di un nuovo tratto
                        if not was_drawing_prev.get(hid, False):
                            push_undo(canvas_led)
                        canvas_led.draw_at(gx, gy,
                                           True, hand_id=hid,
                                           is_erasing=False)
                    synth.play_note(gx, gy,
                                   canvas_led.width, canvas_led.height, True)
                    was_drawing_prev[hid] = True
                elif hand_state.precision_erasing:
                    # Usa la punta dell'indice (landmark 8) invece del midpoint pollice-indice
                    if hand_state.landmarks is not None:
                        ex = int(hand_state.landmarks[8][0] * fw)
                        ey = int(hand_state.landmarks[8][1] * fh)
                        egx = (ex - ox) // cell if cell > 0 else -1
                        egy = (ey - oy) // cell if cell > 0 else -1
                    else:
                        egx, egy = gx, gy
                    in_grid_erase = 0 <= egx < CANVAS_W and 0 <= egy < CANVAS_H
                    if in_grid_erase:
                        if not was_drawing_prev.get(hid, False):
                            push_undo(canvas_led)
                        canvas_led.draw_at(egx, egy,
                                           True, hand_id=hid,
                                           is_erasing=True)
                        was_drawing_prev[hid] = True
                    else:
                        canvas_led.draw_at(0, 0, False, hand_id=hid)
                        was_drawing_prev[hid] = False
                else:
                    canvas_led.draw_at(gx, gy,
                                       False, hand_id=hid,
                                       is_erasing=False)
                    was_drawing_prev[hid] = False

            # -- GUI: griglia LED sovrapposta alla webcam --
            frame_preview = frame.copy()

            # 1) Pixel colorati (opacita 70%) — invariato
            rgb = canvas_led.get_frame_rgb()
            overlay = frame_preview.copy()
            for gy_i in range(CANVAS_H):
                for gx_i in range(CANVAS_W):
                    r, g, b = int(rgb[gy_i, gx_i, 0]), int(rgb[gy_i, gx_i, 1]), int(rgb[gy_i, gx_i, 2])
                    if r > 0 or g > 0 or b > 0:
                        px1 = ox + gx_i * cell
                        py1 = oy + gy_i * cell
                        cv2.rectangle(overlay, (px1, py1),
                                      (px1 + cell, py1 + cell), (b, g, r), -1)
            cv2.addWeighted(overlay, 0.7, frame_preview, 0.3, 0, frame_preview)

            # 2) Griglia (sottile, anti-aliased) — invariato
            grid_color = (60, 60, 60)
            for gx_i in range(CANVAS_W + 1):
                x = ox + gx_i * cell
                cv2.line(frame_preview, (x, oy), (x, oy + grid_h), grid_color, 1, cv2.LINE_AA)
            for gy_i in range(CANVAS_H + 1):
                y = oy + gy_i * cell
                cv2.line(frame_preview, (ox, y), (ox + grid_w, y), grid_color, 1, cv2.LINE_AA)
            # Bordo griglia con glow (doppio anello)
            cv2.rectangle(frame_preview, (ox - 4, oy - 4),
                          (ox + grid_w + 4, oy + grid_h + 4), (0, 100, 80), 3, cv2.LINE_AA)
            cv2.rectangle(frame_preview, (ox - 2, oy - 2),
                          (ox + grid_w + 2, oy + grid_h + 2), (0, 225, 175), 2, cv2.LINE_AA)

            # 3) Scheletro mano + mirino — invariato
            for hand_state in hand_states:
                tracker.draw_overlay(frame_preview, hand_state)

            # ── Costanti GUI ──────────────────────────────────────
            _F = cv2.FONT_HERSHEY_SIMPLEX
            _FD = cv2.FONT_HERSHEY_DUPLEX
            TEAL   = (0, 225, 175)
            TEAL_D = (0, 100, 80)
            BG     = (22, 18, 32)       # viola scuro per i pannelli

            # ── TOP BAR ──────────────────────────────────────────
            top_h = 44
            _bg = frame_preview.copy()
            cv2.rectangle(_bg, (0, 0), (fw, top_h), BG, -1)
            cv2.addWeighted(_bg, 0.88, frame_preview, 0.12, 0, frame_preview)
            cv2.line(frame_preview, (0, top_h - 1), (fw, top_h - 1), TEAL, 2)

            # Logo "FLED CAM" (sinistra)
            cv2.putText(frame_preview, "FLED", (7, 30), _FD, 0.9,
                        TEAL, 2, cv2.LINE_AA)
            cv2.putText(frame_preview, "CAM",  (80, 30), _FD, 0.9,
                        (220, 220, 220), 1, cv2.LINE_AA)

            # Hint centrato nello spazio rimanente
            hint = "Pinch=disegna   V=colore   +/-=pennello   F=fill   Z=undo   S=salva"
            (hw, _), _ = cv2.getTextSize(hint, _F, 0.32, 1)
            hint_x = 136 + (fw - 136 - hw) // 2
            cv2.putText(frame_preview, hint, (hint_x, 28), _F, 0.32,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # ── BOTTOM BAR ───────────────────────────────────────
            bar_h = 72
            bar_y = fh - bar_h
            _bg2 = frame_preview.copy()
            cv2.rectangle(_bg2, (0, bar_y), (fw, fh), BG, -1)
            cv2.addWeighted(_bg2, 0.90, frame_preview, 0.10, 0, frame_preview)
            cv2.line(frame_preview, (0, bar_y), (fw, bar_y), TEAL, 2)

            # — Palette colori —
            pal_step = 42
            pal_x    = 24
            pal_cy   = bar_y + 26
            active_ci = canvas_led.get_color_index()

            for ci, col in enumerate(COLOR_PALETTE):
                cx_c  = pal_x + ci * pal_step
                bgr_c = (int(col[2]), int(col[1]), int(col[0]))
                is_active = (ci == active_ci)
                r_c = 15 if is_active else 11
                # Alone esterno per attivo
                if is_active:
                    cv2.circle(frame_preview, (cx_c, pal_cy), r_c + 6, TEAL_D, -1, cv2.LINE_AA)
                    cv2.circle(frame_preview, (cx_c, pal_cy), r_c + 4, TEAL,   2,  cv2.LINE_AA)
                else:
                    cv2.circle(frame_preview, (cx_c, pal_cy), r_c + 2, (50, 50, 65), 1, cv2.LINE_AA)
                # Cerchio colore
                draw_col = (62, 62, 72) if col == (0, 0, 0) else bgr_c
                cv2.circle(frame_preview, (cx_c, pal_cy), r_c, draw_col, -1, cv2.LINE_AA)

            # Nome colore attivo centrato sotto il suo cerchio
            c_name = canvas_led.get_color_name().upper()
            (cnw, _), _ = cv2.getTextSize(c_name, _F, 0.28, 1)
            cv2.putText(frame_preview, c_name,
                        (pal_x + active_ci * pal_step - cnw // 2, bar_y + bar_h - 8),
                        _F, 0.28, (190, 190, 200), 1, cv2.LINE_AA)

            # Separatore
            sep1_x = pal_x + len(COLOR_PALETTE) * pal_step + 6
            cv2.line(frame_preview, (sep1_x, bar_y + 10), (sep1_x, fh - 10), (55, 55, 70), 1)

            # — Brush preview —
            brush_cx = sep1_x + 30
            brush_cy = bar_y + 26
            bvr = min(max(3, canvas_led.brush_size * 4), 18)
            cv2.circle(frame_preview, (brush_cx, brush_cy), bvr + 3, (50, 50, 65), -1, cv2.LINE_AA)
            cv2.circle(frame_preview, (brush_cx, brush_cy), bvr,     (200, 200, 210), -1, cv2.LINE_AA)
            b_label = f"B{canvas_led.brush_size}"
            (blw, _), _ = cv2.getTextSize(b_label, _F, 0.28, 1)
            cv2.putText(frame_preview, b_label,
                        (brush_cx - blw // 2, bar_y + bar_h - 8),
                        _F, 0.28, (150, 150, 165), 1, cv2.LINE_AA)

            # Separatore
            sep2_x = brush_cx + 36
            cv2.line(frame_preview, (sep2_x, bar_y + 10), (sep2_x, fh - 10), (55, 55, 70), 1)

            # — Pill modalità (arrotondata, testo centrato) —
            mode_label = "FILL" if fill_mode else "DRAW"
            pill_col   = (30, 175, 65)  if fill_mode else (200, 90, 20)
            pill_glow  = (20, 120, 45)  if fill_mode else (140, 60, 12)
            (tw, th), _ = cv2.getTextSize(mode_label, _F, 0.46, 1)
            pill_w  = tw + 28
            pill_h  = th + 16
            pill_x  = sep2_x + 16
            pill_cy_center = bar_y + bar_h // 2 - 4
            pill_y1 = pill_cy_center - pill_h // 2
            pill_y2 = pill_y1 + pill_h
            # Alone
            _rounded_rect(frame_preview, (pill_x - 2, pill_y1 - 2),
                          (pill_x + pill_w + 2, pill_y2 + 2), pill_glow,
                          radius=(pill_h + 4) // 2)
            # Corpo pill
            _rounded_rect(frame_preview, (pill_x, pill_y1),
                          (pill_x + pill_w, pill_y2), pill_col,
                          radius=pill_h // 2)
            # Bordo bianco
            _rounded_rect(frame_preview, (pill_x, pill_y1),
                          (pill_x + pill_w, pill_y2), (255, 255, 255),
                          radius=pill_h // 2, filled=False)
            # Testo centrato nella pill
            _put_text_centered(frame_preview, mode_label,
                               pill_x + pill_w // 2, pill_cy_center,
                               _F, 0.46, (255, 255, 255))

            # — Smoothing badge —
            next_badge_x = pill_x + pill_w + 12
            sm_label = "~ON" if tracker.smoothing_enabled else "~OFF"
            sm_col_bg = (40, 70, 40) if tracker.smoothing_enabled else (70, 40, 40)
            sm_col_fg = (120, 220, 120) if tracker.smoothing_enabled else (220, 120, 120)
            (sw, sh), _ = cv2.getTextSize(sm_label, _F, 0.32, 1)
            sx1 = next_badge_x
            sx2 = sx1 + sw + 14
            sy1 = pill_cy_center - sh // 2 - 5
            sy2 = pill_cy_center + sh // 2 + 5
            _rounded_rect(frame_preview, (sx1, sy1), (sx2, sy2),
                          sm_col_bg, radius=(sy2 - sy1) // 2)
            _rounded_rect(frame_preview, (sx1, sy1), (sx2, sy2),
                          sm_col_fg, radius=(sy2 - sy1) // 2, filled=False)
            _put_text_centered(frame_preview, sm_label,
                               (sx1 + sx2) // 2, pill_cy_center,
                               _F, 0.32, sm_col_fg)

            # — Watermark centrato verticalmente nella barra —
            if _wm_img is not None:
                wm_x = fw - _wm_w - 12
                wm_y = bar_y + (bar_h - _wm_h) // 2
                wm_y = max(0, min(wm_y, fh - _wm_h))
                roi = frame_preview[wm_y:wm_y + _wm_h, wm_x:wm_x + _wm_w]
                if roi.shape[:2] == (_wm_h, _wm_w):
                    blended = (_wm_img * _wm_alpha + roi * (1.0 - _wm_alpha)).astype(np.uint8)
                    frame_preview[wm_y:wm_y + _wm_h, wm_x:wm_x + _wm_w] = blended

            cv2.imshow('FLED CAM v1.1.5', frame_preview)

            # -- Tastiera --
            key = cv2.waitKey(16) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                push_undo(canvas_led)
                canvas_led.clear()
                last_erase_time = time.time()
                print("[CANCELLA] Lavagna cancellata (tasto C)")
            elif key == ord('z'):
                pop_undo(canvas_led)
            elif key == ord('f'):
                fill_mode = not fill_mode
                print(f"[FILL] {'ON' if fill_mode else 'OFF'}")
            elif key == ord('s'):
                save_and_send(canvas_led)
            elif key == ord('+') or key == ord('='):
                new_size = min(5, canvas_led.brush_size + 1)
                canvas_led.set_brush_size(new_size)
                print(f"[PENNELLO] {new_size}px")
            elif key == ord('-'):
                new_size = max(1, canvas_led.brush_size - 1)
                canvas_led.set_brush_size(new_size)
                print(f"[PENNELLO] {new_size}px")
            elif key == ord('1'): 
                tracker.set_num_hands(1)
            elif key == ord('2'):
                tracker.set_num_hands(2)
            elif key == ord('r'):
                canvas_led.set_color_by_index(0)
                print(f"[COLORE] {canvas_led.get_color_name()}")
            elif key == ord('b'):
                canvas_led.set_color_by_index(1)
                print(f"[COLORE] {canvas_led.get_color_name()}")
            elif key == ord('y'):
                canvas_led.set_color_by_index(2)
                print(f"[COLORE] {canvas_led.get_color_name()}")
            elif key == ord('w'):
                canvas_led.set_color_by_index(3)
                print(f"[COLORE] {canvas_led.get_color_name()}")
            elif key == ord('n'):
                canvas_led.set_color_by_index(4)
                print(f"[COLORE] {canvas_led.get_color_name()}")
            elif key == ord('t'):
                tracker.toggle_smoothing()

    except KeyboardInterrupt:
        print("\n[BYE] Chiusura...")

    finally:
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
