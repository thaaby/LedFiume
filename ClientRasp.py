#!/usr/bin/env python3
"""
ClientRasp.py - Riceve PNG + Palline + LED (Raspberry Pi)
Watchdog monitora INCOMING/ per nuovi PNG e li sposta in RED/BLUE/YELLOW.
Webcam rileva palline colorate e anima sprite random sui LED 96x8. siummicanzis
"""

import cv2
import numpy as np
import time
import os
import glob
import random
import signal
import sys
import shutil
import threading

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    print("[!] watchdog non installato (pip install watchdog)")

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
# CONFIGURAZIONE
# ============================================================
CAMERA_SCAN = False
HEADLESS    = True

GAMMA       = 2.5
gamma_table = np.array([((i / 255.0) ** GAMMA) * 255
                        for i in np.arange(0, 256)]).astype("uint8")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
INCOMING_DIR = os.path.join(BASE_DIR, 'INCOMING')
ARCHIVE_DIR  = os.path.join(BASE_DIR, 'ARCHIVE')
MAX_IMAGES_PER_FOLDER = 3

# ============================================================
# SUONI (pygame.mixer)
# ============================================================
_AUDIO_DIR = os.path.join(BASE_DIR, 'audio')
_AUDIO_FILES = {
    'red':    os.path.join(_AUDIO_DIR, 'C.mp3'),
    'blue':   os.path.join(_AUDIO_DIR, 'E.mp3'),
    'yellow': os.path.join(_AUDIO_DIR, 'G.mp3'),
}
_FALLBACK_FREQS = {'red': 523.25, 'blue': 659.25, 'yellow': 783.99}

COLOR_SOUNDS = {}

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
                print(f"[WARN] {os.path.basename(path)}: {e}")
        snd = _make_chime_sound(_FALLBACK_FREQS[color])
        snd.set_volume(0.8)
        COLOR_SOUNDS[color] = snd
        print(f"[OK] Audio sintetico: {color}")

def play_color_sound(color_name):
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
CROSSHAIR_HALF     = None
MIN_COLOR_RATIO    = 0.03
FISH_DIRECTION     = "left"

ANIM_FPS           = 30
FISH_SPEED_PPS_MIN = 34
FISH_SPEED_PPS_MAX = 136
VEL_PXS_MIN        = 60
VEL_PXS_MAX        = 600

BG_WARMUP_FRAMES   = 30
BG_LEARN_ALPHA     = 0.001
BG_DIFF_THRESHOLD  = 30

# ============================================================
# SPRITE - Cartelle colore
# ============================================================
SPRITE_FOLDERS = {
    'red':    os.path.join(BASE_DIR, 'RED'),
    'blue':   os.path.join(BASE_DIR, 'BLUE'),
    'yellow': os.path.join(BASE_DIR, 'YELLOW'),
}

# Sprite di default: MAI cancellare/archiviare questi file
DEFAULT_SPRITES = {
    'red':    'pesce.png',
    'blue':   'capitone.png',
    'yellow': 'girino.png',
}
PROTECTED_FILES = set(DEFAULT_SPRITES.values())

def load_random_sprite(color_name):
    """Carica un PNG random dalla cartella del colore. Fallback allo sprite default."""
    folder = SPRITE_FOLDERS.get(color_name)
    if not folder or not os.path.isdir(folder):
        return None
    png_files = glob.glob(os.path.join(folder, '*.png'))
    if not png_files:
        # Fallback: sprite default
        default_name = DEFAULT_SPRITES.get(color_name)
        if default_name:
            fallback_path = os.path.join(folder, default_name)
            if os.path.exists(fallback_path):
                png_files = [fallback_path]
        if not png_files:
            return None
    path = random.choice(png_files)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim < 3 or img.shape[2] != 4:
        return None
    ys, xs = np.where(img[:, :, 3] > 0)
    if len(xs) == 0:
        return None
    cx = (int(xs.min()) + int(xs.max())) // 2
    pixels = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        b, g, r, _ = img[y, x]
        pixels.append((y, x - cx, (int(r), int(g), int(b))))
    print(f"[SPRITE] {os.path.basename(path)} da {color_name}/: {len(pixels)} px")
    return pixels


def draw_sprite(matrix, cx, cy, pixels, facing_right=True):
    rows, cols = matrix.shape[:2]
    s = 1 if facing_right else -1
    for dy, dx, color in pixels:
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


MAGIC_HEADER = bytes([0xFF, 0x4C, 0x45])
_serial_busy = False

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
def select_camera():
    if not CAMERA_SCAN:
        print("[CAM] Webcam 0 (default)")
        return 0
    print("\n[SCAN] Ricerca webcam...")
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append(i)
            cap.release()
    if not cameras:
        print("[!] Nessuna webcam trovata, provo 0...")
        return 0
    if len(cameras) == 1:
        print(f"[OK] Camera {cameras[0]}")
        return cameras[0]
    for c in cameras:
        print(f"  [{c}] Camera {c}")
    while True:
        try:
            choice = input(f"> Seleziona (0-{cameras[-1]}): ")
            cid = int(choice)
            if cid in cameras:
                return cid
        except ValueError:
            pass


# ============================================================
# WATCHDOG - Monitoraggio cartella INCOMING
# ============================================================
def enforce_folder_limit(folder_path):
    """Se la cartella ha piu di MAX_IMAGES_PER_FOLDER PNG, sposta i piu vecchi in ARCHIVE/.
    I file in PROTECTED_FILES (sprite default) non vengono mai archiviati."""
    png_files = glob.glob(os.path.join(folder_path, '*.png'))
    # Separa protetti da archiviabili
    archivable = [f for f in png_files if os.path.basename(f) not in PROTECTED_FILES]
    total = len(png_files)
    if total <= MAX_IMAGES_PER_FOLDER:
        return
    # Ordina per data di modifica (piu vecchio prima) - solo i non protetti
    archivable.sort(key=lambda f: os.path.getmtime(f))
    excess = total - MAX_IMAGES_PER_FOLDER
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    archived = 0
    for src in archivable:
        if archived >= excess:
            break
        dest = os.path.join(ARCHIVE_DIR, os.path.basename(src))
        # Evita conflitti di nome nell'archivio
        if os.path.exists(dest):
            name, ext = os.path.splitext(os.path.basename(src))
            dest = os.path.join(ARCHIVE_DIR, f"{name}_{int(time.time())}{ext}")
        try:
            shutil.move(src, dest)
            print(f"[ARCHIVE] {os.path.basename(src)} -> ARCHIVE/")
            archived += 1
        except Exception as e:
            print(f"[ARCHIVE] Errore: {e}")


class ColorFolderHandler(FileSystemEventHandler):
    """Monitora RED/BLUE/YELLOW: quando arriva un nuovo PNG, applica il limite.
    Gestisce sia on_created (file nuovo) che on_modified (file sovrascritto da SCP)."""

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self._last_check = 0

    def _handle(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith('.png'):
            return
        # Evita check multipli ravvicinati (SCP genera piu eventi per file)
        now = time.time()
        if now - self._last_check < 1.0:
            return
        self._last_check = now
        # Aspetta che il file sia completamente scritto
        time.sleep(1.0)
        enforce_folder_limit(self.folder_path)

    def on_created(self, event):
        self._handle(event)

    def on_modified(self, event):
        self._handle(event)

    def on_closed(self, event):
        self._handle(event)


class IncomingHandler(FileSystemEventHandler):
    """Quando un nuovo PNG arriva in INCOMING/, lo sposta in RED/BLUE/YELLOW random."""

    def on_created(self, event):
        if event.is_directory:
            return
        filepath = event.src_path
        if not filepath.lower().endswith('.png'):
            return

        # Aspetta che il file sia completamente scritto (SCP)
        time.sleep(0.5)

        if not os.path.exists(filepath):
            return

        folders = ['RED', 'BLUE', 'YELLOW']
        chosen = random.choice(folders)
        dest_dir = os.path.join(BASE_DIR, chosen)
        os.makedirs(dest_dir, exist_ok=True)

        # Nome: colore_N.png (es. RED_1.png, BLUE_2.png)
        existing = glob.glob(os.path.join(dest_dir, f"{chosen}_*.png"))
        next_num = len(existing) + 1
        new_filename = f"{chosen}_{next_num}.png"
        dest_path = os.path.join(dest_dir, new_filename)

        try:
            shutil.move(filepath, dest_path)
            print(f"[INCOMING] -> {chosen}/{new_filename}")
            # Applica il limite dopo lo spostamento
            enforce_folder_limit(dest_dir)
        except Exception as e:
            print(f"[INCOMING] Errore spostamento: {e}")


def start_watchdog():
    """Avvia il watchdog su INCOMING/ e sulle cartelle colore (limite immagini)."""
    if not HAS_WATCHDOG:
        print("[WARN] watchdog non disponibile - i PNG in INCOMING/ non verranno spostati")
        return None

    os.makedirs(INCOMING_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    observer = Observer()

    # Watchdog su INCOMING/
    observer.schedule(IncomingHandler(), INCOMING_DIR, recursive=False)
    print(f"[OK] Watchdog attivo su INCOMING/")

    # Watchdog sulle cartelle colore (limite MAX_IMAGES_PER_FOLDER)
    for color_name, folder in SPRITE_FOLDERS.items():
        os.makedirs(folder, exist_ok=True)
        observer.schedule(ColorFolderHandler(folder), folder, recursive=False)
        # Applica il limite anche all'avvio
        enforce_folder_limit(folder)

    print(f"[OK] Watchdog attivo su RED/, BLUE/, YELLOW/ (max {MAX_IMAGES_PER_FOLDER} img)")

    observer.daemon = True
    observer.start()
    return observer


# ============================================================
# MAIN
# ============================================================
def main():
    global HAS_SOUND

    print("=== CLIENT RASPBERRY - PALLINE + LED + WATCHDOG ===")

    # -- Audio --
    if HAS_PYGAME:
        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            HAS_SOUND = True
            _init_audio()
        except Exception as e:
            print(f"[!] Audio non disponibile: {e}")
            HAS_SOUND = False

    # -- Watchdog --
    observer = start_watchdog()

    # -- Arduino --
    ser = create_arduino_serial()

    # -- Webcam --
    camera_id = select_camera()
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("[!] Errore apertura webcam")
        send_black_and_close(ser)
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # -- Signal handler --
    def signal_handler(sig, frame_info):
        print("\n[INFO] CTRL+C ricevuto, pulizia...")
        send_black_and_close(ser)
        cap.release()
        if observer:
            observer.stop()
        if HAS_SOUND:
            pygame.mixer.quit()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # -- Calibrazione sfondo --
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
    print(f"[OK] Sfondo catturato ({bg_count} frame)")

    bg_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    tube_mask_global = create_tube_mask(bg_hsv)
    tube_pixels = cv2.countNonZero(tube_mask_global)
    print(f"[OK] Tubo rilevato: {tube_pixels} pixel da escludere")

    # -- Stato --
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
    crosshair_half = CROSSHAIR_HALF

    print("[PRONTO] Webcam + LED + Watchdog attivi. Lancia le biglie!")

    try:
        while True:
            loop_start   = time.time()
            dt           = min(max(loop_start - last_frame_t, 1e-3), 0.1)
            last_frame_t = loop_start

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

            # Sfondo adattivo
            no_motion = cv2.bitwise_not(fgmask)
            frame_blur = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32)
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

            # Tracking centroide
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
                        active_fish.append({
                            'pixels':       sprite_pixels,
                            'x':            start_x,
                            'y':            0,
                            'speed_pps':    speed_pps,
                            'facing_right': facing,
                        })
                        cooldowns[color_name] = COOLDOWN_FRAMES
                        play_color_sound(color_name)
                        dir_label = "->" if facing else "<-"
                        print(f"[TRIGGER] {color_name.upper()} ({int(ratio*100)}%) "
                              f"| vel={velocity_pxs:.0f}px/s fish={speed_pps:.0f}px/s "
                              f"| dir={dir_label}")

            # Matrice LED
            matrix = np.zeros((ARDUINO_ROWS, ARDUINO_COLS, 3), dtype=np.uint8)

            next_fish = []
            for fish in active_fish:
                draw_sprite(matrix, int(fish['x']), fish['y'],
                            fish['pixels'], fish['facing_right'])
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

            # Timing
            elapsed = time.time() - loop_start
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except Exception as e:
        print(f"\n[!!!] ERRORE: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("[INFO] Pulizia in corso...")
        cap.release()
        if observer:
            observer.stop()
            observer.join(timeout=2)
        send_black_and_close(ser)
        if HAS_SOUND:
            pygame.mixer.quit()


if __name__ == "__main__":
    main()
