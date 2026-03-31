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
        # Rosso puro ESTREMO: Hue strettissimo (0-2/178-180) e saturazione altissima
        # per ignorare QUALSIASI tubo arancione, anche se la webcam lo vede tendente al rosso.
        'lower1': np.array([0,   120, 60]),
        'upper1': np.array([3,   255, 255]),
        'lower2': np.array([177, 120, 60]),
        'upper2': np.array([180, 255, 255]),
        'bgr': (0, 0, 255),
        'rgb': (255, 0, 0),
    },
    'yellow': {
        # Giallo: S_min abbassato a 45 — il giallo desatura facilmente.
        'lower': np.array([20,  45,  40]),
        'upper': np.array([45,  255, 255]),
        'bgr': (0, 255, 255),
        'rgb': (255, 255, 0),
    },
    'blue': {
        # Blu: S_min abbassato a 45, V_min a 25 per blu scuri.
        'lower': np.array([85,  45,  25]),
        'upper': np.array([135, 255, 255]),
        'bgr': (255, 0, 0),
        'rgb': (0, 0, 255),
    },
}

# ============================================================
# MIRINO — PARAMETRI
# ============================================================
CROSSHAIR_HALF  = None      # None = intero frame webcam (calcolato a runtime)
MIN_COLOR_RATIO = 0.04      # almeno 4% dei pixel nel mirino

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
def detect_color(hsv_roi):
    """Ritorna (color_name, ratio) oppure (None, 0.0)."""
    total = hsv_roi.shape[0] * hsv_roi.shape[1]
    if total == 0:
        return None, 0.0

    best_name  = None
    best_ratio = 0.0

    for name, params in COLOR_RANGES.items():
        if name == 'red':
            m = cv2.bitwise_or(
                cv2.inRange(hsv_roi, params['lower1'], params['upper1']),
                cv2.inRange(hsv_roi, params['lower2'], params['upper2']))
        else:
            m = cv2.inRange(hsv_roi, params['lower'], params['upper'])

        ratio = cv2.countNonZero(m) / total
        if ratio > best_ratio:
            best_ratio = ratio
            best_name  = name

    if best_ratio >= MIN_COLOR_RATIO:
        return best_name, best_ratio
    return None, 0.0


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

def send_matrix_state(ser, matrix_rgb):
    """Invia ogni frame senza handshake bloccante: svuota il buffer RX in modo
    non-bloccante, poi trasmette sempre il frame corrente.
    Questo garantisce fluidità costante indipendentemente dalla latenza Arduino."""
    if ser is None:
        return
    # Svuota silenziosamente eventuali dati in arrivo (evita accumulo buffer)
    if ser.in_waiting > 0:
        ser.read_all()
    rgb_gamma     = gamma_table[matrix_rgb]
    mapped_pixels = rgb_gamma[LED_MAP_Y, LED_MAP_X]
    ser.write(MAGIC_HEADER + mapped_pixels.tobytes())


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

    # ── Stato animazione pesce ──
    fish_speed        = 2.0     # pixel LED per frame (dinamico)
    anim_active       = False
    anim_color_name   = None
    anim_x            = 0.0
    anim_y            = ARDUINO_ROWS // 2
    anim_facing_right = True

    COOLDOWN_FRAMES  = 20
    cooldown_counter = 0

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

        detected, ratio = detect_color(roi_hsv_moving)

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

        # Mappa velocità in pixel/frame → FISH_SPEED
        v_clamped  = max(float(VEL_PIX_MIN), min(velocity_px, float(VEL_PIX_MAX)))
        v_norm     = (v_clamped - VEL_PIX_MIN) / (VEL_PIX_MAX - VEL_PIX_MIN)
        fish_speed = FISH_SPEED_MIN + v_norm * (FISH_SPEED_MAX - FISH_SPEED_MIN)

        # ── 2. TRIGGER ANIMAZIONE ────────────────────────────────────
        if cooldown_counter > 0:
            cooldown_counter -= 1

        if detected and not anim_active and cooldown_counter == 0:
            anim_active     = True
            anim_color_name = detected
            anim_x          = -8.0 if anim_facing_right else float(ARDUINO_COLS + 8)
            cooldown_counter = COOLDOWN_FRAMES
            print(f"[TRIGGER] {detected.upper()} rilevato ({int(ratio*100)}%) | vel={velocity_px:.1f}px/f FISH_SPEED={fish_speed:.1f}")

        # ── 3. RENDER MATRICE LED ────────────────────────────────────
        matrix = np.zeros((ARDUINO_ROWS, ARDUINO_COLS, 3), dtype=np.uint8)

        if anim_active:
            fish_rgb = COLOR_RANGES[anim_color_name]['rgb']
            draw_fish(matrix, int(anim_x), anim_y, fish_rgb, anim_facing_right)
            anim_x += fish_speed if anim_facing_right else -fish_speed
            if (anim_facing_right and anim_x > ARDUINO_COLS + 10) or \
               (not anim_facing_right and anim_x < -10):
                anim_active       = False
                anim_facing_right = not anim_facing_right

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
        if detected:
            cross_bgr = COLOR_RANGES[detected]['bgr']
            label = f"{detected.upper()} {int(ratio * 100)}%"

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
        status = f"ANIM {anim_color_name}" if anim_active else "IDLE"
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
            crosshair_half = min(crosshair_half + 10, fw // 2, fh // 2)
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
