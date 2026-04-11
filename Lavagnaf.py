#!/usr/bin/env python3
"""
Lavagnaf.py - Lavagna LED + Animazione Palline
Disegna con i gesti sulla canvas 8x32, salva con S in RED/BLUE/YELLOW.
Quando una pallina passa davanti alla webcam esterna, un disegno random
dalla cartella del colore corrispondente viene animato sui LED 96x8.
"""

import cv2
import numpy as np
import time
import os
import glob
import signal
import sys
import random
from datetime import datetime

# Moduli della Lavagna LED (in Lavagna-LED/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Lavagna-LED', 'Lavagna-LED'))
from hand_tracker import HandTracker, HandState
from led_canvas import LEDCanvas, COLOR_PALETTE, COLOR_NAMES_IT
from audio_synth import AudioSynth

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[!] pyserial non installato (pip install pyserial)")

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("[!] pygame non installato (pip install pygame)")

HAS_SOUND = False

# ============================================================
# CONFIGURAZIONE ARDUINO (SERIALE) - Matrice orizzontale 96x8
# ============================================================
ARDUINO_ENABLED            = True
ARDUINO_PORT               = "auto"
ARDUINO_BAUD               = 500000
ARDUINO_ROWS               = 8
ARDUINO_COLS               = 96
ARDUINO_PANEL_W            = 32
ARDUINO_PANEL_H            = 8
ARDUINO_PANELS_COUNT       = 3
ARDUINO_PANEL_ORDER        = [2, 1, 0]
ARDUINO_PANEL_START_BOTTOM = [True] * 3
ARDUINO_SERPENTINE_Y       = True

# ============================================================
# CONFIGURAZIONE CANVAS DISEGNO
# ============================================================
CANVAS_W = 32
CANVAS_H = 8

# ============================================================
# CONFIGURAZIONE WEBCAM
# ============================================================
CAMERA_SCAN       = False
HEADLESS          = False
DRAWING_CAM_INDEX = 1   # webcam Mac per disegnare
MARBLE_CAM_INDEX  = 0   # webcam esterna per palline (fiume)

GAMMA       = 2.5
gamma_table = np.array([((i / 255.0) ** GAMMA) * 255
                        for i in np.arange(0, 256)]).astype("uint8")

# ============================================================
# SUONI (pygame.mixer - polifonico, non-bloccante)
# ============================================================
_AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
_AUDIO_FILES = {
    'red':    os.path.join(_AUDIO_DIR, 'C.mp3'),
    'blue':   os.path.join(_AUDIO_DIR, 'E.mp3'),
    'yellow': os.path.join(_AUDIO_DIR, 'G.mp3'),
}
_FALLBACK_FREQS = {'red': 523.25, 'blue': 659.25, 'yellow': 783.99}

COLOR_SOUNDS: dict = {}

def _make_chime_sound(freq, duration=0.8, sr=44100):
    t    = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = (np.sin(2*np.pi*freq*t)
            + 0.3  * np.sin(2*np.pi*freq*2*t)
            + 0.1  * np.sin(2*np.pi*freq*3*t)
            + 0.05 * np.sin(2*np.pi*freq*5*t))
    wave *= np.minimum(t / 0.005, 1.0) * np.exp(-t * 4.0)
    wave  = 0.5 * wave / np.max(np.abs(wave))
    i16   = (wave * 32767).astype(np.int16)
    stereo = np.column_stack([i16, i16])
    return pygame.sndarray.make_sound(stereo)

def _init_audio():
    if not HAS_SOUND:
        return
    pygame.mixer.set_num_channels(16)
    for color, path in _AUDIO_FILES.items():
        if os.path.exists(path):
            try:
                snd = pygame.mixer.Sound(path)
                snd.set_volume(0.8)
                COLOR_SOUNDS[color] = snd
                print(f"[OK] Audio: {os.path.basename(path)}")
                continue
            except Exception as e:
                print(f"[WARN] {os.path.basename(path)}: {e} - campanella sintetica")
        snd = _make_chime_sound(_FALLBACK_FREQS[color])
        snd.set_volume(0.8)
        COLOR_SOUNDS[color] = snd
        print(f"[OK] Audio sintetico: {color}")

def play_color_sound(color_name):
    if HAS_SOUND and color_name in COLOR_SOUNDS:
        COLOR_SOUNDS[color_name].play()

# ============================================================
# PRE-COMPUTAZIONE MATRICE LED (column-major, pannelli ruotati)
# ============================================================
def precompute_led_mapping():
    total_pixels = ARDUINO_COLS * ARDUINO_ROWS
    map_y = np.zeros(total_pixels, dtype=int)
    map_x = np.zeros(total_pixels, dtype=int)
    idx = 0
    for p in range(ARDUINO_PANELS_COUNT):
        panel_pos_x   = ARDUINO_PANEL_ORDER[p]
        start_x       = panel_pos_x * ARDUINO_PANEL_W
        starts_bottom = ARDUINO_PANEL_START_BOTTOM[p]
        for x_local in range(ARDUINO_PANEL_W):
            for y_local in range(ARDUINO_PANEL_H):
                eff_y = y_local
                if ARDUINO_SERPENTINE_Y and (x_local % 2 == 1):
                    eff_y = (ARDUINO_PANEL_H - 1) - y_local
                if starts_bottom:
                    eff_y = (ARDUINO_PANEL_H - 1) - eff_y
                global_x   = start_x + (ARDUINO_PANEL_W - 1) - x_local
                global_y   = eff_y
                map_y[idx] = global_y
                map_x[idx] = global_x
                idx += 1
    return map_y, map_x

LED_MAP_Y, LED_MAP_X = precompute_led_mapping()

# ============================================================
# COLORI DA RILEVARE (HSV) - per palline
# ============================================================
TUBE_EXCLUDE = {
    'lower1': np.array([0,   100, 130]),
    'upper1': np.array([10,  190, 230]),
    'lower2': np.array([165, 100, 130]),
    'upper2': np.array([172, 190, 230]),
}

COLOR_RANGES = {
    'red': {
        'lower1': np.array([0,   200, 50]),
        'upper1': np.array([5,   255, 180]),
        'lower2': np.array([165, 200, 50]),
        'upper2': np.array([180, 255, 180]),
        'bgr': (0, 0, 255),
        'rgb': (255, 0, 0),
        'min_ratio': 0.02,
        'min_motion_pixels': 200,
        'exclude_tube': True,
    },
    'yellow': {
        'lower': np.array([18,  15,  25]),
        'upper': np.array([55,  255, 255]),
        'bgr': (0, 255, 255),
        'rgb': (255, 255, 0),
        'min_ratio': 0.015,
        'min_motion_pixels': 0,
        'exclude_tube': False,
    },
    'blue': {
        'lower': np.array([85,  35,  25]),
        'upper': np.array([135, 255, 255]),
        'bgr': (255, 0, 0),
        'rgb': (0, 0, 255),
        'min_ratio': 0.03,
        'min_motion_pixels': 0,
        'exclude_tube': False,
    },
}

# ============================================================
# PARAMETRI ANIMAZIONE PALLINE
# ============================================================
CROSSHAIR_HALF     = None
MIN_COLOR_RATIO    = 0.03
FISH_DIRECTION     = "left"   # "right", "left", "auto"

ANIM_FPS           = 30
FISH_SPEED_PPS_MIN = 34
FISH_SPEED_PPS_MAX = 136
VEL_PXS_MIN        = 60
VEL_PXS_MAX        = 600

BG_WARMUP_FRAMES   = 30
BG_LEARN_ALPHA     = 0.001
BG_DIFF_THRESHOLD  = 30

# ============================================================
# SPRITE - Caricamento dinamico da cartelle
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SPRITE_FOLDERS = {
    'red':    os.path.join(BASE_DIR, 'RED'),
    'blue':   os.path.join(BASE_DIR, 'BLUE'),
    'yellow': os.path.join(BASE_DIR, 'YELLOW'),
}

def load_random_sprite(color_name):
    """Carica un PNG random dalla cartella del colore. Ritorna lista di (dy, dx, (r,g,b)) o None."""
    folder = SPRITE_FOLDERS.get(color_name)
    if not folder or not os.path.isdir(folder):
        print(f"[WARN] Cartella {color_name} non trovata")
        return None
    png_files = glob.glob(os.path.join(folder, '*.png'))
    if not png_files:
        print(f"[WARN] Nessun PNG in {color_name}/")
        return None
    path = random.choice(png_files)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim < 3 or img.shape[2] != 4:
        print(f"[WARN] {path} non RGBA, skip")
        return None
    ys, xs = np.where(img[:, :, 3] > 0)
    if len(xs) == 0:
        return None
    cx = (int(xs.min()) + int(xs.max())) // 2
    cy = (int(ys.min()) + int(ys.max())) // 2
    pixels = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        b, g, r, _ = img[y, x]
        pixels.append((y - cy, x - cx, (int(r), int(g), int(b))))
    print(f"[SPRITE] {os.path.basename(path)} da {color_name}/: {len(pixels)} px")
    return pixels


def draw_sprite(matrix, cx, cy, pixels, facing_right=True):
    """Disegna uno sprite (lista di pixel) sulla matrice LED."""
    rows, cols = matrix.shape[:2]
    s = 1 if facing_right else -1
    for dy, dx, color in pixels:
        r = cy + dy
        c = cx + s * dx
        if 0 <= r < rows and 0 <= c < cols:
            matrix[r, c] = color


# ============================================================
# SALVATAGGIO TRASPARENTE
# ============================================================
def save_transparent_drawing(canvas_led):
    """Salva il disegno come RGBA PNG (sfondo trasparente) in una cartella random."""
    folders = ['RED', 'BLUE', 'YELLOW']
    chosen = random.choice(folders)
    folder_path = os.path.join(BASE_DIR, chosen)
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"drawing_{timestamp}.png"
    filepath = os.path.join(folder_path, filename)

    rgb = canvas_led.get_frame_rgb()  # (8, 32, 3)
    h, w = rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    non_black = np.any(rgb > 0, axis=2)
    rgba[non_black, 3] = 255

    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(filepath, bgra)
    print(f"[SALVA] Disegno salvato in {chosen}/{filename}")


# ============================================================
# RILEVAMENTO COLORE (per palline)
# ============================================================
def create_tube_mask(hsv_roi):
    m1 = cv2.inRange(hsv_roi, TUBE_EXCLUDE['lower1'], TUBE_EXCLUDE['upper1'])
    m2 = cv2.inRange(hsv_roi, TUBE_EXCLUDE['lower2'], TUBE_EXCLUDE['upper2'])
    return cv2.bitwise_or(m1, m2)


def detect_colors_on_mask(hsv_roi, motion_mask, tube_mask=None):
    motion_pixels = cv2.countNonZero(motion_mask)
    if motion_pixels < 50:
        return []
    hsv_masked = cv2.bitwise_and(hsv_roi, hsv_roi, mask=motion_mask)
    found = []
    for name, params in COLOR_RANGES.items():
        min_motion = params.get('min_motion_pixels', 0)
        if motion_pixels < min_motion:
            continue
        if name == 'red':
            color_mask = cv2.bitwise_or(
                cv2.inRange(hsv_masked, params['lower1'], params['upper1']),
                cv2.inRange(hsv_masked, params['lower2'], params['upper2']))
            if params.get('exclude_tube', False) and tube_mask is not None:
                color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(tube_mask))
        else:
            color_mask = cv2.inRange(hsv_masked, params['lower'], params['upper'])
        threshold = params.get('min_ratio', MIN_COLOR_RATIO)
        color_pixels = cv2.countNonZero(color_mask)
        ratio = color_pixels / max(motion_pixels, 1)
        if ratio >= threshold and color_pixels > 20:
            found.append((name, ratio))
    return found


# ============================================================
# COMUNICAZIONE ARDUINO
# ============================================================
def create_arduino_serial():
    if not ARDUINO_ENABLED or not HAS_SERIAL:
        return None
    port = ARDUINO_PORT
    if port == "auto":
        raw = (glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*') +
               glob.glob('/dev/cu.usbmodem*') + glob.glob('/dev/cu.usbserial*') +
               glob.glob('/dev/tty.*') + glob.glob('/dev/cu.*'))
        candidates = [c for c in raw
                      if "debug-console" not in c and "Bluetooth" not in c and "BTH" not in c]
        if not candidates:
            print("[!] Nessuna porta seriale trovata")
            return None
        port = candidates[0]
        print(f"[AUTO] Porta seriale: {port}")
    try:
        ser = serial.Serial(port, ARDUINO_BAUD, timeout=0.01)
        time.sleep(2)
        print(f"[OK] Arduino connesso su {port}")
        ser.read_all()
        return ser
    except Exception as e:
        print(f"[!] Errore Arduino: {e}")
        return None


MAGIC_HEADER  = bytes([0xFF, 0x4C, 0x45])
_serial_busy  = False

def send_matrix_state(ser, matrix_rgb):
    global _serial_busy
    if ser is None:
        return
    try:
        if ser.in_waiting > 0:
            resp = ser.read_all()
            if b'K' in resp or b'O' in resp:
                _serial_busy = False
        if _serial_busy:
            return
        rgb_gamma     = gamma_table[matrix_rgb]
        mapped_pixels = rgb_gamma[LED_MAP_Y, LED_MAP_X]
        ser.write(MAGIC_HEADER + mapped_pixels.tobytes())
        _serial_busy  = True
    except Exception as e:
        print(f"[!] Errore seriale: {e}")
        _serial_busy = False


def send_black_and_close(ser):
    if ser is None:
        return
    try:
        black  = np.zeros((ARDUINO_ROWS, ARDUINO_COLS, 3), dtype=np.uint8)
        mapped = gamma_table[black][LED_MAP_Y, LED_MAP_X]
        ser.write(MAGIC_HEADER + mapped.tobytes())
        time.sleep(0.1)
        ser.close()
        print("[OK] LED spenti. Connessione chiusa.")
    except Exception:
        try:
            ser.close()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================
def main():
    global HAS_SOUND

    print("\n" + "=" * 50)
    print("  LAVAGNA LED + ANIMAZIONE PALLINE")
    print("  Disegna con le mani, salva con S")
    print("  Le palline animano i disegni sui LED!")
    print("=" * 50)

    # -- Init audio --
    if HAS_PYGAME:
        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            HAS_SOUND = True
            _init_audio()
        except Exception as e:
            print(f"[!] Audio non disponibile: {e}")
            HAS_SOUND = False

    # -- Arduino --
    ser = create_arduino_serial()

    # -- Webcam disegno (PC) --
    print(f"\n[CAM] Apertura webcam disegno (indice {DRAWING_CAM_INDEX})...")
    cap_drawing = cv2.VideoCapture(DRAWING_CAM_INDEX)
    if not cap_drawing.isOpened():
        print("[!] Webcam disegno non trovata!")
        send_black_and_close(ser)
        return
    cap_drawing.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_drawing.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[OK] Webcam disegno pronta")

    # -- Webcam palline (esterna) --
    print(f"[CAM] Apertura webcam palline (indice {MARBLE_CAM_INDEX})...")
    cap_marble = cv2.VideoCapture(MARBLE_CAM_INDEX)
    marble_cam_ok = cap_marble.isOpened()
    if marble_cam_ok:
        cap_marble.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap_marble.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_marble.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_marble.set(cv2.CAP_PROP_FPS, 60)
        print("[OK] Webcam palline pronta")
    else:
        print("[WARN] Webcam palline non trovata - solo modalita disegno")

    # -- Hand tracker e canvas --
    synth = AudioSynth()
    tracker = HandTracker(canvas_width=CANVAS_W, canvas_height=CANVAS_H)
    canvas_led = LEDCanvas(CANVAS_W, CANVAS_H)

    # -- Signal handler --
    def signal_handler(sig, frame_info):
        print("\n[INFO] CTRL+C ricevuto, pulizia...")
        send_black_and_close(ser)
        cap_drawing.release()
        if marble_cam_ok:
            cap_marble.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
        if HAS_SOUND:
            pygame.mixer.quit()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # -- Calibrazione sfondo webcam palline --
    background = None
    bg_float = None
    tube_mask_global = None

    if marble_cam_ok:
        print(f"\n[INFO] Cattura sfondo palline ({BG_WARMUP_FRAMES} frame)... NON muovere biglie!")
        bg_accum = None
        bg_count = 0
        for _ in range(BG_WARMUP_FRAMES + 10):
            ret, f = cap_marble.read()
            if not ret:
                continue
            f = cv2.flip(f, 1)
            f_blur = cv2.GaussianBlur(f, (5, 5), 0).astype(np.float32)
            if bg_accum is None:
                bg_accum = f_blur
                bg_count = 1
            else:
                bg_accum += f_blur
                bg_count += 1
        background = (bg_accum / bg_count).astype(np.uint8)
        bg_float = background.astype(np.float32)
        print(f"[OK] Sfondo catturato (media di {bg_count} frame)")

        bg_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        tube_mask_global = create_tube_mask(bg_hsv)
        tube_pixels = cv2.countNonZero(tube_mask_global)
        print(f"[OK] Tubo rilevato: {tube_pixels} pixel da escludere")

    # -- Stato animazione palline --
    active_animations = []
    COOLDOWN_FRAMES   = 20
    cooldowns         = {name: 0 for name in COLOR_RANGES}
    prev_centroid       = None
    velocity_px         = 0.0
    velocity_pxs        = 0.0
    motion_facing_right = True
    dx_history          = []
    DX_WINDOW           = 5

    # -- Stato disegno --
    last_erase_time = 0.0
    ERASE_COOLDOWN  = 1.5

    # -- Timing --
    frame_interval = 1.0 / ANIM_FPS
    last_frame_t   = time.time() - frame_interval
    fps_history    = []
    FPS_WINDOW     = 10
    crosshair_half = CROSSHAIR_HALF

    print("\n" + "-" * 50)
    print("  CONTROLLI:")
    print("  [1-9]   - Cambia colore pennello")
    print("  [+/-]   - Cambia dimensione pennello")
    print("  [C]     - Cancella lavagna")
    print("  [S]     - Salva disegno (PNG trasparente, cartella random)")
    print("  [Q/ESC] - Esci")
    print("")
    print("  GESTI:")
    print("  Pinch (indice+pollice) = Disegna")
    print("  Pollice in giu = Cancella lavagna")
    print("  Segno della pace (V) = Cambia colore")
    print("-" * 50 + "\n")
    print("[PRONTO] Disegna e lancia le biglie!")

    try:
        while True:
            loop_start   = time.time()
            dt           = min(max(loop_start - last_frame_t, 1e-3), 0.1)
            last_frame_t = loop_start

            fps_history.append(1.0 / max(dt, 1e-6))
            if len(fps_history) > FPS_WINDOW:
                fps_history.pop(0)
            fps_avg = sum(fps_history) / len(fps_history)

            # ============================================
            # 1. DISEGNO - webcam PC
            # ============================================
            ret_d, frame_d = cap_drawing.read()
            if not ret_d:
                time.sleep(0.01)
                continue
            frame_d = cv2.flip(frame_d, 1)

            hand_states = tracker.process_frame(frame_d)

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
                    print("[CANCELLA] Lavagna cancellata con POLLICE IN GIU!")

            for hand_state in hand_states:
                if hand_state.peace_sign:
                    current_idx = canvas_led.get_color_index()
                    next_idx = (current_idx + 1) % len(COLOR_PALETTE)
                    canvas_led.set_color_by_index(next_idx)
                    print(f"[PACE] Nuovo colore: {canvas_led.get_color_name()}")

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

            # ============================================
            # 2. PALLINE - webcam esterna
            # ============================================
            if marble_cam_ok and background is not None:
                ret_m, frame_m = cap_marble.read()
                if ret_m:
                    frame_m = cv2.flip(frame_m, 1)
                    fh, fw = frame_m.shape[:2]
                    mid_x, mid_y = fw // 2, fh // 2

                    if crosshair_half is None:
                        crosshair_half = min(fw // 2, fh // 2)

                    x1 = max(0,  mid_x - crosshair_half)
                    y1 = max(0,  mid_y - crosshair_half)
                    x2 = min(fw, mid_x + crosshair_half)
                    y2 = min(fh, mid_y + crosshair_half)

                    roi_bgr = frame_m[y1:y2, x1:x2]
                    roi_bg  = background[y1:y2, x1:x2]

                    diff = cv2.absdiff(roi_bgr, roi_bg)
                    diff_gray = np.max(diff, axis=2)
                    _, fgmask = cv2.threshold(diff_gray, BG_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                    fgmask = fgmask.astype(np.uint8)

                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    fgmask = cv2.erode(fgmask, kern, iterations=1)
                    fgmask = cv2.dilate(fgmask, kern, iterations=3)

                    # Aggiorna sfondo adattivo
                    no_motion = cv2.bitwise_not(fgmask)
                    frame_blur = cv2.GaussianBlur(frame_m, (5, 5), 0).astype(np.float32)
                    mask_3ch = np.stack([no_motion, no_motion, no_motion], axis=2).astype(np.float32) / 255.0
                    roi_frame_f = frame_blur[y1:y2, x1:x2]
                    roi_bg_f    = bg_float[y1:y2, x1:x2]
                    roi_mask    = mask_3ch
                    bg_float[y1:y2, x1:x2] = roi_bg_f * (1 - BG_LEARN_ALPHA * roi_mask) + \
                                               roi_frame_f * (BG_LEARN_ALPHA * roi_mask)
                    background[y1:y2, x1:x2] = bg_float[y1:y2, x1:x2].astype(np.uint8)

                    # Detect colori
                    roi_hsv = cv2.cvtColor(cv2.medianBlur(roi_bgr, 3), cv2.COLOR_BGR2HSV)
                    tube_mask_roi = tube_mask_global[y1:y2, x1:x2]
                    detected_list = detect_colors_on_mask(roi_hsv, fgmask, tube_mask_roi)

                    # Tracking centroide per direzione
                    moments = cv2.moments(fgmask)
                    if moments['m00'] > 100:
                        cx_roi = int(moments['m10'] / moments['m00'])
                        cy_roi = int(moments['m01'] / moments['m00'])
                        curr_centroid = (cx_roi, cy_roi)
                        if prev_centroid is not None:
                            dx_val = curr_centroid[0] - prev_centroid[0]
                            dy_val = curr_centroid[1] - prev_centroid[1]
                            velocity_px = float(np.sqrt(dx_val*dx_val + dy_val*dy_val))
                            if velocity_px > 120:
                                dx_history.clear()
                            dx_history.append(dx_val)
                            if len(dx_history) > DX_WINDOW:
                                dx_history.pop(0)
                            avg_dx = sum(dx_history) / len(dx_history)
                            if abs(avg_dx) > 0.5:
                                motion_facing_right = avg_dx > 0
                        prev_centroid = curr_centroid
                    else:
                        prev_centroid = None
                        velocity_px   = 0.0
                        dx_history.clear()

                    velocity_pxs = velocity_px / dt
                    v_clamped    = max(VEL_PXS_MIN, min(velocity_pxs, VEL_PXS_MAX))
                    v_norm       = (v_clamped - VEL_PXS_MIN) / (VEL_PXS_MAX - VEL_PXS_MIN)
                    speed_pps    = FISH_SPEED_PPS_MIN + v_norm * (FISH_SPEED_PPS_MAX - FISH_SPEED_PPS_MIN)

                    for name in cooldowns:
                        if cooldowns[name] > 0:
                            cooldowns[name] -= 1

                    for color_name, ratio in detected_list:
                        if cooldowns[color_name] == 0:
                            if FISH_DIRECTION == "right":
                                facing = True
                            elif FISH_DIRECTION == "left":
                                facing = False
                            else:
                                facing = motion_facing_right
                            sprite_pixels = load_random_sprite(color_name)
                            if sprite_pixels:
                                start_x = -8.0 if facing else float(ARDUINO_COLS + 8)
                                active_animations.append({
                                    'pixels':       sprite_pixels,
                                    'x':            start_x,
                                    'y':            ARDUINO_ROWS // 2,
                                    'speed_pps':    speed_pps,
                                    'facing_right': facing,
                                })
                                cooldowns[color_name] = COOLDOWN_FRAMES
                                play_color_sound(color_name)
                                dir_label = "->" if facing else "<-"
                                print(f"[TRIGGER] {color_name.upper()} ({int(ratio*100)}%) "
                                      f"| vel={velocity_pxs:.0f}px/s fish={speed_pps:.0f}px/s "
                                      f"| dir={dir_label}")

                    # Overlay mirino sulla webcam palline
                    if not HEADLESS:
                        cross_bgr = (80, 80, 80)
                        if detected_list:
                            cross_bgr = COLOR_RANGES[detected_list[0][0]]['bgr']
                        cv2.rectangle(frame_m, (x1, y1), (x2, y2), cross_bgr, 2)

            # ============================================
            # 3. MATRICE LED 96x8 - solo animazioni
            # ============================================
            matrix = np.zeros((ARDUINO_ROWS, ARDUINO_COLS, 3), dtype=np.uint8)

            next_anims = []
            for anim in active_animations:
                draw_sprite(matrix, int(anim['x']), anim['y'],
                            anim['pixels'], anim['facing_right'])
                if anim['facing_right']:
                    anim['x'] += anim['speed_pps'] * dt
                else:
                    anim['x'] -= anim['speed_pps'] * dt
                if anim['facing_right'] and anim['x'] < ARDUINO_COLS + 10:
                    next_anims.append(anim)
                elif not anim['facing_right'] and anim['x'] > -10:
                    next_anims.append(anim)
            active_animations = next_anims

            send_matrix_state(ser, matrix)

            # ============================================
            # 4. TIMING
            # ============================================
            elapsed = time.time() - loop_start
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            # ============================================
            # 5. GUI
            # ============================================
            if not HEADLESS:
                # Webcam disegno + overlay
                frame_preview = frame_d.copy()
                for hand_state in hand_states:
                    tracker.draw_overlay(frame_preview, hand_state)

                color_bgr = tuple(int(c) for c in canvas_led.current_color[::-1])
                info_text = f"Colore: {canvas_led.get_color_name()} | Pennello: {canvas_led.brush_size}px"
                cv2.putText(frame_preview, info_text, (10, frame_d.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                cv2.rectangle(frame_preview, (frame_d.shape[1] - 50, 10),
                              (frame_d.shape[1] - 10, 50), color_bgr, -1)
                cv2.rectangle(frame_preview, (frame_d.shape[1] - 50, 10),
                              (frame_d.shape[1] - 10, 50), (255, 255, 255), 1)
                cv2.imshow('Lavagna LED - Webcam', frame_preview)

                # Canvas preview
                cursor_x, cursor_y = -1, -1
                for h in hand_states:
                    if h.detected:
                        cursor_x, cursor_y = h.canvas_x, h.canvas_y
                        break
                canvas_preview = canvas_led.get_preview(
                    scale=15, cursor_x=cursor_x, cursor_y=cursor_y)
                cv2.imshow('Lavagna LED - Canvas', canvas_preview)

                # Webcam palline
                if marble_cam_ok and 'frame_m' in dir():
                    cv2.imshow('Palline - Webcam', frame_m)

                # Preview LED
                preview = cv2.resize(matrix, (480, 40), interpolation=cv2.INTER_NEAREST)
                preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
                cv2.imshow('LED Matrix', preview_bgr)

                # Tastiera
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('c'):
                    canvas_led.clear()
                    last_erase_time = time.time()
                    print("[CANCELLA] Lavagna cancellata (tasto C)")
                elif key == ord('s'):
                    save_transparent_drawing(canvas_led)
                elif key == ord('+') or key == ord('='):
                    new_size = min(5, canvas_led.brush_size + 1)
                    canvas_led.set_brush_size(new_size)
                    print(f"[PENNELLO] Dimensione: {new_size}px")
                elif key == ord('-'):
                    new_size = max(1, canvas_led.brush_size - 1)
                    canvas_led.set_brush_size(new_size)
                    print(f"[PENNELLO] Dimensione: {new_size}px")
                elif ord('1') <= key <= ord('9'):
                    idx = key - ord('1')
                    canvas_led.set_color_by_index(idx)
                    print(f"[COLORE] {canvas_led.get_color_name()}")
                elif key == ord('b') and marble_cam_ok:
                    print("[INFO] Ricalibrazione sfondo palline...")
                    bg_accum = None
                    bg_count = 0
                    for _ in range(BG_WARMUP_FRAMES):
                        ret2, f2 = cap_marble.read()
                        if not ret2:
                            continue
                        f2 = cv2.flip(f2, 1)
                        f2_blur = cv2.GaussianBlur(f2, (5, 5), 0).astype(np.float32)
                        if bg_accum is None:
                            bg_accum = f2_blur
                            bg_count = 1
                        else:
                            bg_accum += f2_blur
                            bg_count += 1
                    background = (bg_accum / bg_count).astype(np.uint8)
                    bg_float = background.astype(np.float32)
                    bg_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
                    tube_mask_global = create_tube_mask(bg_hsv)
                    print(f"[OK] Sfondo ricalibrato ({bg_count} frame)")

    except Exception as e:
        print(f"\n[!!!] ERRORE: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("[INFO] Pulizia in corso...")
        tracker.release()
        cap_drawing.release()
        if marble_cam_ok:
            cap_marble.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
        send_black_and_close(ser)
        if HAS_SOUND:
            pygame.mixer.quit()


if __name__ == "__main__":
    main()
