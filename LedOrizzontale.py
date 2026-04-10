#!/usr/bin/env python3
"""
LedOrizzontale - Mirino Centrale + Animazione LED (pannelli orizzontali 96x8)
Niente tracking: rileva rosso/blu/giallo nel mirino.
Quando un colore passa (anche ad alta velocita), lancia l'animazione pesce.
"""

import cv2
import numpy as np
import time
import os
import glob
import signal
import sys

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

HAS_SOUND = False   # aggiornato in main() dopo init mixer

# ============================================================
# CONFIGURAZIONE ARDUINO (SERIALE)
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
ARDUINO_PANEL_START_BOTTOM = [False] * 3
ARDUINO_SERPENTINE_Y       = True     # pannelli ruotati: serpentine su Y (colonne dispari invertono Y)

# ============================================================
# CONFIGURAZIONE WEBCAM
# ============================================================
CAMERA_SCAN = False  # False = usa webcam 0 direttamente. True = scansiona e scegli.

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

def _make_chime_sound(freq: float, duration: float = 0.8, sr: int = 44100):
    """Campanella sintetica come fallback se i file MP3 non sono disponibili."""
    t    = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = (np.sin(2*np.pi*freq*t)
            + 0.3  * np.sin(2*np.pi*freq*2*t)
            + 0.1  * np.sin(2*np.pi*freq*3*t)
            + 0.05 * np.sin(2*np.pi*freq*5*t))
    wave *= np.minimum(t / 0.005, 1.0) * np.exp(-t * 4.0)
    wave  = 0.5 * wave / np.max(np.abs(wave))
    i16   = (wave * 32767).astype(np.int16)
    stereo = np.column_stack([i16, i16])          # pygame vuole stereo
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

def play_color_sound(color_name: str):
    if HAS_SOUND and color_name in COLOR_SOUNDS:
        COLOR_SOUNDS[color_name].play()

# ============================================================
# PRE-COMPUTAZIONE MATRICE LED
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
        # Pannelli ruotati: il nastro LED percorre le colonne (column-major)
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
# COLORI DA RILEVARE (HSV)
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
# PARAMETRI ANIMAZIONE
# ============================================================
CROSSHAIR_HALF  = None
MIN_COLOR_RATIO = 0.03
FISH_DIRECTION  = "left"  # "right" = sempre destra, "left" = sempre sinistra, "auto" = segue il movimento

ANIM_FPS            = 30
FISH_SPEED_PPS_MIN  = 34
FISH_SPEED_PPS_MAX  = 136
VEL_PXS_MIN         = 60
VEL_PXS_MAX         = 600

BG_WARMUP_FRAMES    = 30
BG_LEARN_ALPHA      = 0.001
BG_DIFF_THRESHOLD   = 30

# ============================================================
# SPRITE ANIMALI
# ============================================================
# Mappa colore -> file sprite
_SPRITE_FILES = {
    'red':    'pesce.png',
    'blue':   'capitone.png',
    'yellow': 'girino.png',
}
_SPRITES: dict = {}  # colore -> lista di (dy, dx, (R, G, B))

def _load_sprite(name, filename):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, filename)
    img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim < 3 or img.shape[2] != 4:
        print(f"[WARN] {path} non trovato o non RGBA - sprite {name} non caricato")
        return
    ys, xs = np.where(img[:, :, 3] > 0)
    if len(xs) == 0:
        return
    cx = (int(xs.min()) + int(xs.max())) // 2
    cy = (int(ys.min()) + int(ys.max())) // 2
    pixels = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        b, g, r, _ = img[y, x]
        pixels.append((y - cy, x - cx, (int(r), int(g), int(b))))
    _SPRITES[name] = pixels
    print(f"[OK] Sprite {name} ({filename}): {len(pixels)} pixel, centro=({cx},{cy})")

for _color, _file in _SPRITE_FILES.items():
    _load_sprite(_color, _file)


def draw_fish(matrix, cx, cy, color_name, facing_right=True):
    """Disegna lo sprite associato al colore sulla matrice LED."""
    if color_name not in _SPRITES:
        return
    rows, cols = matrix.shape[:2]
    s = 1 if facing_right else -1
    for dy, dx, color in _SPRITES[color_name]:
        r = cy + dy
        c = cx + s * dx
        if 0 <= r < rows and 0 <= c < cols:
            matrix[r, c] = color


# ============================================================
# RILEVAMENTO COLORE
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
# SELEZIONE CAMERA
# ============================================================

def list_cameras():
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append(i)
            cap.release()
    return cameras


def select_camera():
    if not CAMERA_SCAN:
        print("[CAM] Webcam 0 (default - CAMERA_SCAN = False)")
        return 0

    print("\n[SCAN] Ricerca webcam...")
    cameras = list_cameras()
    if not cameras:
        print("[!] Nessuna webcam trovata, provo ID 0...")
        return 0
    print(f"[CAM] Trovate: {len(cameras)}")
    for cam_id in cameras:
        print(f"  [{cam_id}] Camera {cam_id}")
    if len(cameras) == 1:
        print(f"[OK] Camera {cameras[0]} selezionata")
        return cameras[0]
    while True:
        try:
            choice = input(f"> Seleziona camera (0-{cameras[-1]}): ")
            cam_id = int(choice)
            if cam_id in cameras:
                return cam_id
            print("[X] Camera non valida!")
        except ValueError:
            print("[X] Inserisci un numero!")


# ============================================================
# MAIN
# ============================================================
def main():
    global HAS_SOUND
    print("=== LEDORIZZONTALE - MIRINO CENTRALE ===")

    # -- Init audio (qui, non a livello modulo, per non bloccare l'avvio) --
    if HAS_PYGAME:
        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            HAS_SOUND = True
            _init_audio()
        except Exception as e:
            print(f"[!] Audio non disponibile: {e}")
            HAS_SOUND = False

    ser = create_arduino_serial()

    camera_id = select_camera()
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("[!] Errore apertura webcam")
        send_black_and_close(ser)
        return

    cap.set(cv2.CAP_PROP_FOURCC,         cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,    640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,   480)
    cap.set(cv2.CAP_PROP_FPS,            60)

    def signal_handler(sig, frame_info):
        print("\n[INFO] CTRL+C ricevuto, pulizia...")
        send_black_and_close(ser)
        cap.release()
        cv2.destroyAllWindows()
        if HAS_SOUND:
            pygame.mixer.quit()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    print(f"[INFO] Cattura sfondo ({BG_WARMUP_FRAMES} frame)... NON muovere biglie!")
    bg_accum = None
    bg_count = 0
    for _ in range(BG_WARMUP_FRAMES + 10):
        ret, f = cap.read()
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

    print("[INFO] Calcolo maschera tubo per esclusione rosso...")
    bg_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    tube_mask_global = create_tube_mask(bg_hsv)
    tube_pixels = cv2.countNonZero(tube_mask_global)
    print(f"[OK] Tubo rilevato: {tube_pixels} pixel da escludere")

    active_fish     = []
    COOLDOWN_FRAMES = 20
    cooldowns       = {name: 0 for name in COLOR_RANGES}

    prev_centroid       = None
    velocity_px         = 0.0
    velocity_pxs        = 0.0
    motion_facing_right = True
    dx_history          = []
    DX_WINDOW           = 5

    frame_interval = 1.0 / ANIM_FPS
    last_frame_t   = time.time() - frame_interval

    fps_history    = []
    FPS_WINDOW     = 10

    crosshair_half = CROSSHAIR_HALF

    print("[INFO] Pronto! Lancia le biglie.")
    print("[INFO] Tasti: 'q' esci | '+'/'-' mirino | 'f' fullframe | 'b' ricalibra sfondo")

    try:
        while True:
            loop_start   = time.time()
            dt           = min(max(loop_start - last_frame_t, 1e-3), 0.1)
            last_frame_t = loop_start

            fps_history.append(1.0 / max(dt, 1e-6))
            if len(fps_history) > FPS_WINDOW:
                fps_history.pop(0)
            fps_avg = sum(fps_history) / len(fps_history)

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)

            fh, fw = frame.shape[:2]
            mid_x, mid_y = fw // 2, fh // 2

            if crosshair_half is None:
                crosshair_half = min(fw // 2, fh // 2)

            x1 = max(0,  mid_x - crosshair_half)
            y1 = max(0,  mid_y - crosshair_half)
            x2 = min(fw, mid_x + crosshair_half)
            y2 = min(fh, mid_y + crosshair_half)

            roi_bgr = frame[y1:y2, x1:x2]
            roi_bg  = background[y1:y2, x1:x2]

            diff = cv2.absdiff(roi_bgr, roi_bg)
            diff_gray = np.max(diff, axis=2)
            _, fgmask = cv2.threshold(diff_gray, BG_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            fgmask = fgmask.astype(np.uint8)

            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgmask = cv2.erode(fgmask, kern, iterations=1)
            fgmask = cv2.dilate(fgmask, kern, iterations=3)

            no_motion = cv2.bitwise_not(fgmask)
            frame_blur = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32)
            mask_3ch = np.stack([no_motion, no_motion, no_motion], axis=2).astype(np.float32) / 255.0
            roi_frame_f = frame_blur[y1:y2, x1:x2]
            roi_bg_f    = bg_float[y1:y2, x1:x2]
            roi_mask    = mask_3ch
            bg_float[y1:y2, x1:x2] = roi_bg_f * (1 - BG_LEARN_ALPHA * roi_mask) + \
                                       roi_frame_f * (BG_LEARN_ALPHA * roi_mask)
            background[y1:y2, x1:x2] = bg_float[y1:y2, x1:x2].astype(np.uint8)

            roi_hsv = cv2.cvtColor(cv2.medianBlur(roi_bgr, 3), cv2.COLOR_BGR2HSV)
            tube_mask_roi = tube_mask_global[y1:y2, x1:x2]
            detected_list = detect_colors_on_mask(roi_hsv, fgmask, tube_mask_roi)

            moments = cv2.moments(fgmask)
            if moments['m00'] > 100:
                cx_roi = int(moments['m10'] / moments['m00'])
                cy_roi = int(moments['m01'] / moments['m00'])
                curr_centroid = (cx_roi, cy_roi)
                if prev_centroid is not None:
                    dx = curr_centroid[0] - prev_centroid[0]
                    dy = curr_centroid[1] - prev_centroid[1]
                    velocity_px = float(np.sqrt(dx*dx + dy*dy))
                    if velocity_px > 120:
                        dx_history.clear()   # salto anomalo: resetta storia
                    dx_history.append(dx)    # aggiorna direzione in ogni caso
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
                    start_x = -8.0 if facing else float(ARDUINO_COLS + 8)
                    active_fish.append({
                        'color_name':   color_name,
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
                          f"| dir={dir_label} pesci={len(active_fish)}")

            matrix = np.zeros((ARDUINO_ROWS, ARDUINO_COLS, 3), dtype=np.uint8)

            next_fish = []
            for fish in active_fish:
                draw_fish(matrix, int(fish['x']), fish['y'],
                          fish['color_name'], fish['facing_right'])
                if fish['facing_right']:
                    fish['x'] += fish['speed_pps'] * dt
                else:
                    fish['x'] -= fish['speed_pps'] * dt
                if fish['facing_right'] and fish['x'] < ARDUINO_COLS + 10:
                    next_fish.append(fish)
                elif not fish['facing_right'] and fish['x'] > -10:
                    next_fish.append(fish)
            active_fish = next_fish

            send_matrix_state(ser, matrix)

            elapsed = time.time() - loop_start
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            cross_bgr = (80, 80, 80)
            label = ""
            if detected_list:
                cross_bgr = COLOR_RANGES[detected_list[0][0]]['bgr']
                label = "  ".join(
                    f"{n.upper()} {int(r*100)}%" for n, r in detected_list)

            cv2.rectangle(frame, (x1, y1), (x2, y2), cross_bgr, 2)
            gap = 6
            cv2.line(frame, (mid_x - 25, mid_y), (mid_x - gap, mid_y), cross_bgr, 1)
            cv2.line(frame, (mid_x + gap, mid_y), (mid_x + 25, mid_y), cross_bgr, 1)
            cv2.line(frame, (mid_x, mid_y - 25), (mid_x, mid_y - gap), cross_bgr, 1)
            cv2.line(frame, (mid_x, mid_y + gap), (mid_x, mid_y + 25), cross_bgr, 1)
            if label:
                cv2.putText(frame, label, (x2 + 8, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cross_bgr, 2)

            center_h = int(roi_hsv[roi_hsv.shape[0]//2, roi_hsv.shape[1]//2, 0])
            center_s = int(roi_hsv[roi_hsv.shape[0]//2, roi_hsv.shape[1]//2, 1])
            center_v = int(roi_hsv[roi_hsv.shape[0]//2, roi_hsv.shape[1]//2, 2])
            cv2.putText(frame, f"H:{center_h} S:{center_s} V:{center_v}",
                        (x2 + 8, mid_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            motion_pix = cv2.countNonZero(fgmask)
            status = f"PESCI:{len(active_fish)}" if active_fish else "IDLE"
            cv2.putText(frame, f"FPS:{int(fps_avg)}  {status}  MOT:{motion_pix}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("LedOrizzontale - Mirino", frame)

            preview     = cv2.resize(matrix, (480, 40),
                                     interpolation=cv2.INTER_NEAREST)
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            cv2.imshow("LED Matrix", preview_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                crosshair_half = min(
                    int(crosshair_half) + 10, fw // 2, fh // 2)
            elif key == ord('-'):
                crosshair_half = max(crosshair_half - 10, 5)
            elif key == ord('f'):
                if crosshair_half >= min(fw // 2, fh // 2) - 10:
                    crosshair_half = 30
                else:
                    crosshair_half = min(fw // 2, fh // 2)
            elif key == ord('b'):
                print("[INFO] Ricalibrazione sfondo...")
                bg_accum = None
                bg_count = 0
                for _ in range(BG_WARMUP_FRAMES):
                    ret2, f2 = cap.read()
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
                tube_pixels = cv2.countNonZero(tube_mask_global)
                print(f"[OK] Sfondo ricalibrato ({bg_count} frame), tubo: {tube_pixels} px")

    except Exception as e:
        print(f"\n[!!!] ERRORE: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("[INFO] Pulizia in corso...")
        cap.release()
        cv2.destroyAllWindows()
        send_black_and_close(ser)
        if HAS_SOUND:
            pygame.mixer.quit()


if __name__ == "__main__":
    main()