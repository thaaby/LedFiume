#!/usr/bin/env python3
"""
LedFiume — Mirino Centrale + Animazione LED
Niente tracking: rileva rosso/blu/giallo nel mirino.
Quando un colore passa (anche ad alta velocità), lancia l'animazione pesce.
"""

import cv2
import numpy as np
import time
import os
import glob

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[!] pyserial non installato (pip install pyserial)")

# ============================================================
# CONFIGURAZIONE ARDUINO (SERIALE)
# ============================================================
ARDUINO_ENABLED            = True
ARDUINO_PORT               = "auto"
ARDUINO_BAUD               = 500000
ARDUINO_ROWS               = 32
ARDUINO_COLS               = 56          # 7 pannelli × 8 LED = 56 colonne totali
ARDUINO_PANEL_W            = 8
ARDUINO_PANEL_H            = 32
ARDUINO_PANELS_COUNT       = 7
ARDUINO_PANEL_ORDER        = [6, 5, 4, 3, 2, 1, 0]  # ordine fisico da destra a sinistra
ARDUINO_PANEL_START_BOTTOM = [False] * 7
ARDUINO_SERPENTINE_X       = True

GAMMA       = 2.5
gamma_table = np.array([((i / 255.0) ** GAMMA) * 255
                        for i in np.arange(0, 256)]).astype("uint8")

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
        for y_local in range(ARDUINO_PANEL_H):
            global_y = (ARDUINO_PANEL_H - 1) - y_local if starts_bottom else y_local
            for x_local in range(ARDUINO_PANEL_W):
                eff_x = x_local
                if ARDUINO_SERPENTINE_X and (y_local % 2 == 1):
                    eff_x = (ARDUINO_PANEL_W - 1) - x_local
                global_x   = start_x + eff_x
                map_y[idx] = global_y
                map_x[idx] = global_x
                idx += 1
    return map_y, map_x

LED_MAP_Y, LED_MAP_X = precompute_led_mapping()

# ============================================================
# COLORI DA RILEVARE (HSV)
# ============================================================
COLOR_RANGES = {
    'red': {
        'lower1': np.array([0,   120, 60]),
        'upper1': np.array([3,   255, 255]),
        'lower2': np.array([177, 120, 60]),
        'upper2': np.array([180, 255, 255]),
        'bgr': (0, 0, 255),
        'rgb': (255, 0, 0),
        'min_ratio': 0.04,
    },
    'yellow': {
        # Giallo: range Hue esteso (18-55), saturazione minima molto bassa (20)
        # perché le biglie gialle sotto luce artificiale appaiono quasi pastello.
        'lower': np.array([18,  20,  25]),
        'upper': np.array([55,  255, 255]),
        'bgr': (0, 255, 255),
        'rgb': (255, 255, 0),
        'min_ratio': 0.02,   # soglia più bassa: basta 2% di pixel gialli
    },
    'blue': {
        'lower': np.array([85,  45,  25]),
        'upper': np.array([135, 255, 255]),
        'bgr': (255, 0, 0),
        'rgb': (0, 0, 255),
        'min_ratio': 0.04,
    },
}

# ============================================================
# MIRINO — PARAMETRI
# ============================================================
CROSSHAIR_HALF  = None      # None = intero frame webcam (calcolato a runtime)
MIN_COLOR_RATIO = 0.04      # soglia globale (sovrascritta da 'min_ratio' per colore)

# ── Velocity → FISH_SPEED mapping ──
FISH_SPEED_MIN  = 1
FISH_SPEED_MAX  = 8
VEL_PIX_MIN     = 2         # px/frame sotto cui la biglia è considerata ferma
VEL_PIX_MAX     = 40        # px/frame oltre cui si usa la velocità massima

# ============================================================
# SPRITE PESCE (da pixilart-drawing.png)
# ============================================================
_FISH_BODY_COLORS = {(255, 126, 0), (237, 28, 36)}
_FISH_PIXELS: list = []

def _load_fish_sprite():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pixilart-drawing.png')
    img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim < 3 or img.shape[2] != 4:
        print(f"[WARN] {path} non trovato o non RGBA — sprite pesce non caricato")
        return
    ys, xs = np.where(img[:, :, 3] > 0)
    if len(xs) == 0:
        return
    cx = (int(xs.min()) + int(xs.max())) // 2
    cy = (int(ys.min()) + int(ys.max())) // 2
    for y, x in zip(ys.tolist(), xs.tolist()):
        b, g, r, _ = img[y, x]
        _FISH_PIXELS.append((y - cy, x - cx, (int(r), int(g), int(b))))
    print(f"[OK] Sprite pesce: {len(_FISH_PIXELS)} pixel, centro=({cx},{cy})")

_load_fish_sprite()


def draw_fish(matrix, cx, cy, body_color, facing_right=True):
    """Disegna il pesce sulla matrice LED. body_color = RGB tuple."""
    rows, cols = matrix.shape[:2]
    s = 1 if facing_right else -1
    for dy, dx, color in _FISH_PIXELS:
        r = cy - dy
        c = cx + s * dx
        if 0 <= r < rows and 0 <= c < cols:
            matrix[r, c] = body_color if color in _FISH_BODY_COLORS else color


# ============================================================
# RILEVAMENTO COLORE NEL MIRINO
# ============================================================
def detect_colors(hsv_roi):
    """Ritorna una lista di (color_name, ratio) per TUTTI i colori sopra soglia.
    Permette il rilevamento simultaneo di più biglie di colori diversi."""
    total = hsv_roi.shape[0] * hsv_roi.shape[1]
    if total == 0:
        return []

    found = []
    for name, params in COLOR_RANGES.items():
        threshold = params.get('min_ratio', MIN_COLOR_RATIO)
        if name == 'red':
            m = cv2.bitwise_or(
                cv2.inRange(hsv_roi, params['lower1'], params['upper1']),
                cv2.inRange(hsv_roi, params['lower2'], params['upper2']))
        else:
            m = cv2.inRange(hsv_roi, params['lower'], params['upper'])

        ratio = cv2.countNonZero(m) / total
        if ratio >= threshold:
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
_serial_busy  = False   # True dopo aver inviato, attendiamo ACK prima del prossimo invio

def send_matrix_state(ser, matrix_rgb):
    """Invio non-bloccante con handshake leggero.
    - Ogni volta controlla se Arduino ha risposto (byte 'K' nel buffer RX).
    - Solo se libero, invia il prossimo frame.
    - Non blocca MAI il loop principale: se Arduino è ancora occupato, si salta
      il frame seriale (l'animazione continua comunque a scorrere a 30fps)."""
    global _serial_busy
    if ser is None:
        return
    # Controlla risposta Arduino (non bloccante)
    if ser.in_waiting > 0:
        resp = ser.read_all()
        if b'K' in resp or b'O' in resp:   # 'K' o 'OK' comuni nei firmware
            _serial_busy = False
    if _serial_busy:
        return   # Arduino ancora impegnato → saltiamo questo invio
    rgb_gamma     = gamma_table[matrix_rgb]
    mapped_pixels = rgb_gamma[LED_MAP_Y, LED_MAP_X]
    ser.write(MAGIC_HEADER + mapped_pixels.tobytes())
    _serial_busy  = True


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== LEDFIUME — MIRINO CENTRALE ===")
    ser = create_arduino_serial()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Errore apertura webcam")
        return

    cap.set(cv2.CAP_PROP_FOURCC,         cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,    640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,   480)
    cap.set(cv2.CAP_PROP_FPS,            60)

    # ── Multipesce: lista di pesci attivi ──
    # Ogni pesce è un dict: {color_name, x, y, speed, facing_right}
    active_fish   = []
    next_right    = True          # direzione alternata per i nuovi pesci
    COOLDOWN_FRAMES = 20
    cooldowns     = {name: 0 for name in COLOR_RANGES}  # cooldown per-colore

    # ── Tracking velocità biglia ──
    prev_centroid  = None
    velocity_px    = 0.0

    # ── Controllo FPS loop ──
    TARGET_FPS     = 30
    frame_interval = 1.0 / TARGET_FPS
    fps_t          = time.time()
    last_frame_t   = time.time()

    # crosshair_half calcolato a runtime (None = intero frame)
    crosshair_half = CROSSHAIR_HALF  # sarà sovrascritto al primo frame

    # ── Background subtractor: vede SOLO gli oggetti in movimento (biglie) ──
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=False)

    print("[INFO] Pronto! Punta il mirino sulle biglie.")
    print("[INFO] Tasti: 'q' per uscire | '+' o '-' per ridimensionare il mirino | 'f' mirino a tutto schermo.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        fh, fw = frame.shape[:2]
        mid_x, mid_y = fw // 2, fh // 2

        # Primo frame: imposta crosshair a tutto schermo se non ancora calcolato
        if crosshair_half is None:
            crosshair_half = min(fw // 2, fh // 2)

        # ── 1. ESTRAI ROI MIRINO ─────────────────────────────────────
        x1 = max(0,  mid_x - crosshair_half)
        y1 = max(0,  mid_y - crosshair_half)
        x2 = min(fw, mid_x + crosshair_half)
        y2 = min(fh, mid_y + crosshair_half)

        roi_bgr = frame[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(cv2.medianBlur(roi_bgr, 3), cv2.COLOR_BGR2HSV)

        # ── Maschera MOVIMENTO ────────────────────────────────────────
        fgmask = fgbg.apply(roi_bgr)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.erode(fgmask, kern, iterations=1)
        fgmask = cv2.dilate(fgmask, kern, iterations=2)
        roi_hsv_moving = cv2.bitwise_and(roi_hsv, roi_hsv, mask=fgmask)

        detected_list = detect_colors(roi_hsv_moving)

        # ── TRACKING VELOCITÀ BIGLIA ──────────────────────────────────
        # Calcola il centroide dei pixel in movimento per misurare la velocità
        moments = cv2.moments(fgmask)
        if moments['m00'] > 100:   # abbastanza pixel per fidarci
            cx_roi = int(moments['m10'] / moments['m00'])
            cy_roi = int(moments['m01'] / moments['m00'])
            curr_centroid = (cx_roi, cy_roi)
            if prev_centroid is not None:
                dx = curr_centroid[0] - prev_centroid[0]
                dy = curr_centroid[1] - prev_centroid[1]
                velocity_px = float(np.sqrt(dx*dx + dy*dy))
            prev_centroid = curr_centroid
        else:
            prev_centroid = None
            velocity_px = 0.0

        # Mappa velocità in pixel/frame → fish_speed base
        v_clamped  = max(float(VEL_PIX_MIN), min(velocity_px, float(VEL_PIX_MAX)))
        v_norm     = (v_clamped - VEL_PIX_MIN) / (VEL_PIX_MAX - VEL_PIX_MIN)
        base_speed = FISH_SPEED_MIN + v_norm * (FISH_SPEED_MAX - FISH_SPEED_MIN)

        # ── 2. TRIGGER MULTIPESCE ────────────────────────────────────
        for name in cooldowns:
            if cooldowns[name] > 0:
                cooldowns[name] -= 1

        for color_name, ratio in detected_list:
            if cooldowns[color_name] == 0:
                facing = next_right
                next_right = not next_right
                start_x = -8.0 if facing else float(ARDUINO_COLS + 8)
                active_fish.append({
                    'color_name':   color_name,
                    'x':            start_x,
                    'y':            ARDUINO_ROWS // 2,
                    'speed':        base_speed,
                    'facing_right': facing,
                })
                cooldowns[color_name] = COOLDOWN_FRAMES
                print(f"[TRIGGER] {color_name.upper()} rilevato ({int(ratio*100)}%) "
                      f"| vel={velocity_px:.1f}px/f speed={base_speed:.1f} "
                      f"| pesci attivi={len(active_fish)}")

        # ── 3. RENDER MATRICE LED (tutti i pesci) ────────────────────────
        matrix = np.zeros((ARDUINO_ROWS, ARDUINO_COLS, 3), dtype=np.uint8)

        next_fish = []
        for fish in active_fish:
            fish_rgb = COLOR_RANGES[fish['color_name']]['rgb']
            draw_fish(matrix, int(fish['x']), fish['y'], fish_rgb, fish['facing_right'])
            fish['x'] += fish['speed'] if fish['facing_right'] else -fish['speed']
            # Rimuovi il pesce solo quando esce completamente dallo schermo
            if fish['facing_right'] and fish['x'] < ARDUINO_COLS + 10:
                next_fish.append(fish)
            elif not fish['facing_right'] and fish['x'] > -10:
                next_fish.append(fish)
        active_fish = next_fish

        send_matrix_state(ser, matrix)

        # ── Cap a TARGET_FPS per animazione fluida e costante ─────────────
        now = time.time()
        elapsed = now - last_frame_t
        sleep_t = frame_interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
        last_frame_t = time.time()

        # ── 4. OVERLAY MIRINO SU WEBCAM ──────────────────────────────
        cross_bgr = (80, 80, 80)
        label = ""
        if detected_list:
            cross_bgr = COLOR_RANGES[detected_list[0][0]]['bgr']
            label = "  ".join(f"{n.upper()} {int(r*100)}%" for n, r in detected_list)

        # Rettangolo mirino
        cv2.rectangle(frame, (x1, y1), (x2, y2), cross_bgr, 2)
        # Croce centrale (con gap)
        gap = 6
        cv2.line(frame, (mid_x - 25, mid_y), (mid_x - gap, mid_y), cross_bgr, 1)
        cv2.line(frame, (mid_x + gap, mid_y), (mid_x + 25, mid_y), cross_bgr, 1)
        cv2.line(frame, (mid_x, mid_y - 25), (mid_x, mid_y - gap), cross_bgr, 1)
        cv2.line(frame, (mid_x, mid_y + gap), (mid_x, mid_y + 25), cross_bgr, 1)
        # Label colore
        if label:
            cv2.putText(frame, label, (x2 + 8, mid_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cross_bgr, 2)

        # HSV live al centro del mirino (debug calibrazione)
        center_h = int(roi_hsv[roi_hsv.shape[0]//2, roi_hsv.shape[1]//2, 0])
        center_s = int(roi_hsv[roi_hsv.shape[0]//2, roi_hsv.shape[1]//2, 1])
        center_v = int(roi_hsv[roi_hsv.shape[0]//2, roi_hsv.shape[1]//2, 2])
        cv2.putText(frame, f"H:{center_h} S:{center_s} V:{center_v}",
                    (x2 + 8, mid_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        # FPS + stato
        now = time.time()
        fps = 1.0 / max(now - fps_t, 1e-6)
        fps_t = now
        status = f"PESCI:{len(active_fish)}" if active_fish else "IDLE"
        cv2.putText(frame, f"FPS:{int(fps)}  {status}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("LedFiume - Mirino", frame)

        # Anteprima matrice LED
        preview     = cv2.resize(matrix, (560, 320), interpolation=cv2.INTER_NEAREST)
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
        cv2.imshow("LED Matrix", preview_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            crosshair_half = min(int(crosshair_half) + 10, fw // 2, fh // 2)
        elif key == ord('-'):
            crosshair_half = max(crosshair_half - 10, 5)
        elif key == ord('f'):
            if crosshair_half >= min(fw // 2, fh // 2) - 10:
                crosshair_half = 30
            else:
                crosshair_half = min(fw // 2, fh // 2)

    # ── CLEANUP ──
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        black  = np.zeros((ARDUINO_ROWS, ARDUINO_COLS, 3), dtype=np.uint8)
        mapped = gamma_table[black][LED_MAP_Y, LED_MAP_X]
        ser.write(MAGIC_HEADER + mapped.tobytes())
        time.sleep(0.1)
        ser.close()
        print("[OK] LED spenti. Connessione chiusa.")


if __name__ == "__main__":
    main()
