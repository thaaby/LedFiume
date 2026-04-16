#!/usr/bin/env python3
"""
ServerMac.py - Lavagna LED (solo disegno + invio al Raspberry)
Disegna con i gesti sulla canvas 8x32, salva con S come PNG trasparente
e invia automaticamente al Raspberry Pi via SCP nella cartella INCOMING/.
"""

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

            for hand_state in hand_states:
                if hand_state.thumbs_down and not is_any_drawing and (time.time() - last_erase_time > ERASE_COOLDOWN):
                    push_undo(canvas_led)
                    canvas_led.clear()
                    last_erase_time = time.time()
                    print("[CANCELLA] Lavagna cancellata!")

            for hand_state in hand_states:
                if hand_state.peace_sign:
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
                elif hand_state.precision_erasing and in_grid:
                    if not was_drawing_prev.get(hid, False):
                        push_undo(canvas_led)
                    canvas_led.draw_at(gx, gy,
                                       True, hand_id=hid,
                                       is_erasing=True)
                    was_drawing_prev[hid] = True
                else:
                    canvas_led.draw_at(gx, gy,
                                       False, hand_id=hid,
                                       is_erasing=False)
                    was_drawing_prev[hid] = False

            # -- GUI: griglia LED sovrapposta alla webcam --
            frame_preview = frame.copy()

            # 1) Pixel colorati (opacita 70%)
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

            # 2) Griglia (sottile, anti-aliased)
            grid_color = (60, 60, 60)
            for gx_i in range(CANVAS_W + 1):
                x = ox + gx_i * cell
                cv2.line(frame_preview, (x, oy), (x, oy + grid_h), grid_color, 1, cv2.LINE_AA)
            for gy_i in range(CANVAS_H + 1):
                y = oy + gy_i * cell
                cv2.line(frame_preview, (ox, y), (ox + grid_w, y), grid_color, 1, cv2.LINE_AA)
            # Bordo griglia
            cv2.rectangle(frame_preview, (ox, oy), (ox + grid_w, oy + grid_h),
                          (110, 110, 110), 2, cv2.LINE_AA)

            # 3) Scheletro mano + mirino (sopra la griglia)
            for hand_state in hand_states:
                tracker.draw_overlay(frame_preview, hand_state)

            # 4) Barra info in basso (semi-trasparente)
            bar_h = 38
            bar_y = fh - bar_h
            bar_overlay = frame_preview.copy()
            cv2.rectangle(bar_overlay, (0, bar_y), (fw, fh), (25, 25, 25), -1)
            cv2.addWeighted(bar_overlay, 0.82, frame_preview, 0.18, 0, frame_preview)

            # Palette colori (cerchi)
            pal_x = 14
            for ci, col in enumerate(COLOR_PALETTE):
                cx_c = pal_x + ci * 26
                cy_c = bar_y + bar_h // 2
                bgr_c = (int(col[2]), int(col[1]), int(col[0]))
                # Cerchio nero visibile con bordo
                if col == (0, 0, 0):
                    cv2.circle(frame_preview, (cx_c, cy_c), 8, (50, 50, 50), -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame_preview, (cx_c, cy_c), 8, bgr_c, -1, cv2.LINE_AA)
                # Evidenzia colore attivo
                if ci == canvas_led.get_color_index():
                    cv2.circle(frame_preview, (cx_c, cy_c), 11, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.circle(frame_preview, (cx_c, cy_c), 9, (70, 70, 70), 1, cv2.LINE_AA)

            # Testo info
            txt_x = pal_x + len(COLOR_PALETTE) * 26 + 12
            parts = [f"Pen:{canvas_led.brush_size}", f"Mani:{tracker.num_hands}"]
            if fill_mode:
                parts.append("FILL")
            undo_n = len(_undo_stack)
            if undo_n > 0:
                parts.append(f"Undo:{undo_n}")
            info_str = "  ".join(parts)
            cv2.putText(frame_preview, info_str,
                        (txt_x, bar_y + bar_h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (190, 190, 190), 1, cv2.LINE_AA)

            cv2.imshow('FLED CAM v1.0.9', frame_preview)

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

    except KeyboardInterrupt:
        print("\n[BYE] Chiusura...")

    finally:
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
