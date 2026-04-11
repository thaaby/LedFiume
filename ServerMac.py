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
import sys
import subprocess
import random
from datetime import datetime

# Moduli della Lavagna LED (in Lavagna-LED/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Lavagna-LED', 'Lavagna-LED'))
from hand_tracker import HandTracker, HandState
from led_canvas import LEDCanvas, COLOR_PALETTE, COLOR_NAMES_IT
from audio_synth import AudioSynth

# ============================================================
# CONFIGURAZIONE RASPBERRY (SCP)
# ============================================================
RASP_USER = "pit"
RASP_IP   = "10.134.110.80"
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

# Round-robin: cicla RED -> BLUE -> YELLOW -> RED -> ...
_COLOR_CYCLE = ['RED', 'BLUE', 'YELLOW']
_color_index = 0
_color_counters = {'RED': 0, 'BLUE': 0, 'YELLOW': 0}


# ============================================================
# SALVATAGGIO + INVIO SCP
# ============================================================
def save_and_send(canvas_led):
    """Salva il disegno come RGBA PNG e lo invia direttamente nella cartella colore sul Rasp."""
    global _color_index

    os.makedirs(LOCAL_OUTBOX, exist_ok=True)

    # Round-robin per distribuzione omogenea
    chosen = _COLOR_CYCLE[_color_index % len(_COLOR_CYCLE)]
    _color_index += 1
    _color_counters[chosen] += 1
    num = _color_counters[chosen]
    filename = f"{chosen}_{num}.png"
    filepath = os.path.join(LOCAL_OUTBOX, filename)

    # Salva con trasparenza (nero = trasparente)
    rgb = canvas_led.get_frame_rgb()  # (8, 32, 3)
    h, w = rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    non_black = np.any(rgb > 0, axis=2)
    rgba[non_black, 3] = 255

    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
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
    tracker = HandTracker(canvas_width=CANVAS_W, canvas_height=CANVAS_H)
    canvas_led = LEDCanvas(CANVAS_W, CANVAS_H)

    # -- Stato --
    last_erase_time = 0.0
    ERASE_COOLDOWN = 1.5

    print(f"\n[INFO] Destinazione SCP: {RASP_USER}@{RASP_IP}:{RASP_PROJECT}")
    print("\n" + "-" * 50)
    print("  CONTROLLI:")
    print("  [1-9]   - Cambia colore pennello")
    print("  [+/-]   - Cambia dimensione pennello")
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
                    canvas_led.clear()
                    last_erase_time = time.time()
                    print("[CANCELLA] Lavagna cancellata!")

            for hand_state in hand_states:
                if hand_state.peace_sign:
                    current_idx = canvas_led.get_color_index()
                    next_idx = (current_idx + 1) % len(COLOR_PALETTE)
                    canvas_led.set_color_by_index(next_idx)
                    print(f"[COLORE] {canvas_led.get_color_name()}")

                if hand_state.drawing:
                    canvas_led.draw_at(hand_state.canvas_x, hand_state.canvas_y,
                                       True, hand_id=hand_state.hand_label,
                                       is_erasing=False)
                    synth.play_note(hand_state.canvas_x, hand_state.canvas_y,
                                   canvas_led.width, canvas_led.height, True)
                elif hand_state.precision_erasing:
                    canvas_led.draw_at(hand_state.canvas_x, hand_state.canvas_y,
                                       True, hand_id=hand_state.hand_label,
                                       is_erasing=True)
                else:
                    canvas_led.draw_at(hand_state.canvas_x, hand_state.canvas_y,
                                       False, hand_id=hand_state.hand_label,
                                       is_erasing=False)

            # -- GUI --
            frame_preview = frame.copy()
            for hand_state in hand_states:
                tracker.draw_overlay(frame_preview, hand_state)

            color_bgr = tuple(int(c) for c in canvas_led.current_color[::-1])
            info_text = f"Colore: {canvas_led.get_color_name()} | Pennello: {canvas_led.brush_size}px"
            cv2.putText(frame_preview, info_text, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            cv2.rectangle(frame_preview, (frame.shape[1] - 50, 10),
                          (frame.shape[1] - 10, 50), color_bgr, -1)
            cv2.rectangle(frame_preview, (frame.shape[1] - 50, 10),
                          (frame.shape[1] - 10, 50), (255, 255, 255), 1)
            cv2.imshow('ServerMac - Webcam', frame_preview)

            cursor_x, cursor_y = -1, -1
            for h in hand_states:
                if h.detected:
                    cursor_x, cursor_y = h.canvas_x, h.canvas_y
                    break
            canvas_preview = canvas_led.get_preview(
                scale=15, cursor_x=cursor_x, cursor_y=cursor_y)
            cv2.imshow('ServerMac - Canvas', canvas_preview)

            # -- Tastiera --
            key = cv2.waitKey(16) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                canvas_led.clear()
                last_erase_time = time.time()
                print("[CANCELLA] Lavagna cancellata (tasto C)")
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
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                canvas_led.set_color_by_index(idx)
                print(f"[COLORE] {canvas_led.get_color_name()}")

    except KeyboardInterrupt:
        print("\n[BYE] Chiusura...")

    finally:
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
