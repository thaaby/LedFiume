import cv2
import numpy as np
import time
import math
import os
import glob
from collections import deque
import scipy.spatial.distance as dist
from scipy.optimize import linear_sum_assignment

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[!] pyserial non installato — Arduino video disabilitato (pip install pyserial)")

# ============================================================
# CONFIGURAZIONE ARDUINO VIDEO (SERIALE)
# ============================================================
ARDUINO_ENABLED            = True
ARDUINO_PORT               = "auto"
ARDUINO_BAUD               = 500000
ARDUINO_ROWS               = 32
ARDUINO_COLS               = 32
ARDUINO_PANEL_W            = 8
ARDUINO_PANEL_H            = 32
ARDUINO_PANELS_COUNT       = 4
ARDUINO_PANEL_ORDER        = [3, 2, 1, 0]
ARDUINO_PANEL_START_BOTTOM = [False, False, False, False]
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
# COLORI DA TRACKARE (HSV)
# ============================================================
# Come leggere il debug overlay per calibrare:
#   - Il cerchio verde mostra la detection raw
#   - Il numero a fianco è la CIRCOLARITÀ (>0.42 passa)
#   - Il numero piccolo sotto è il RAGGIO in pixel a SCALE_FACTOR
#   - Se la biglia non viene rilevata: abbassar CIRC_MIN o allargare l'HSV
#   - Se vengono rilevati falsi positivi: alzare S_min/V_min o abbassare MAX_R
COLOR_RANGES = {
    'red': {
        # Due lati dello spettro hue — S/V alti evitano pelle e legno
        'lower1': np.array([0,   160,  80]),
        'upper1': np.array([8,   255, 255]),
        'lower2': np.array([172, 160,  80]),
        'upper2': np.array([180, 255, 255]),
        'bgr': (0, 0, 255),
        'rgb': (255, 0, 0),
    },
    'blue': {
        # Calibrato sulla biglia in foto: azzurro-ciano saturo.
        # S_min=120 esclude cieli, sfondi chiari, navy opaco.
        # V_min=90  esclude blu scuro di vestiti/jeans.
        # H 90-125  non cattura il verde (H<90) né il viola (H>130).
        'lower': np.array([90,  120,  90]),
        'upper': np.array([125, 255, 255]),
        'bgr': (255, 0, 0),
        'rgb': (0, 0, 255),
    },
    'yellow': {
        # S_min=110: esclude giallo-verde pallido di piante, pareti calde
        'lower': np.array([18, 110,  80]),
        'upper': np.array([38, 255, 255]),
        'bgr': (0, 255, 255),
        'rgb': (255, 255, 0),
    },
}

# ============================================================
# PARAMETRI GEOMETRICI BIGLIA
# ============================================================
# Biglia ~3.5cm diametro (da foto).
# Calcolo a SCALE_FACTOR=0.75 su webcam 640x480 con FOV ~70°:
#   a 25cm → raggio ~28px  |  a 50cm → raggio ~14px  |  a 80cm → raggio ~9px
# MIN_R=7  — sotto questa soglia è rumore o riflesso puntuale
# MAX_R=30 — sopra questa soglia è un oggetto grande (braccio, spalla, testa, vestito)
#            equivale a ~40px su frame pieno, che è già più grande di una testa a 1m
MIN_R    = 7
MAX_R    = 30

# Circolarità minima: 4π·A/P²
#   Biglia perfetta       → ~1.00
#   Biglia con blur 2:1   → ~0.70
#   Biglia con blur 3:1   → ~0.50 (limite inferiore realistico)
#   Patch di vestito/muro → ~0.15–0.35
CIRC_MIN = 0.42

# Solidità minima: area_contorno / area_convex_hull
#   Sfera (sempre convessa) → 0.90–1.00
#   Oggetto con rientranze  → 0.50–0.80
#   Persone/indumenti       → 0.60–0.85  (filtro secondario + circolarità)
SOLID_MIN = 0.82

# Soglie FAST: si attivano solo dentro la ROI predittiva di una biglia già tracciata
# e ad alta velocità. Fuori da quella zona i filtri normali rimangono intatti.
CIRC_MIN_FAST  = 0.25   # blur 4:1 → ~0.38, accetta striscia da lancio veloce
SOLID_MIN_FAST = 0.62   # forma allungata ma ancora prevalentemente convessa
MAX_R_FAST     = 48     # biglia con blur occupa più spazio del normale

# Velocità (px/frame, coordinate full-res) sopra cui si attivano soglie FAST
# e si allarga il ROI di predizione. A 60fps: 18 px/frame ≈ 69 cm/s
SPEED_THRESHOLD   = 18
# Il raggio del ROI di predizione cresce di N px (a SCALE_FACTOR) per ogni px/frame
SPEED_ROI_FACTOR  = 2.2

# ============================================================
# ANIMAZIONE PESCE — Clownfish Nemo multi-layer
# ============================================================
# Sprite rivolto a DESTRA. Anchor = punta sinistra della coda.
# Coordinate (dy, dx): dy negativo = verso l'alto, dx positivo = verso destra.
#
# Layer di disegno (in ordine, gli ultimi sovrascrivono i precedenti):
#   1. CORPO (body_color = colore pallina rilevata)
#   2. CODA interna (body_color)
#   3. STRISCIA BIANCA verticale
#   4. CONTORNO NERO (corpo + pinna dorsale + coda)
#   5. OCCHIO bianco
#   6. PUPILLA nera

# Pixel estratti dinamicamente dal PNG
FISH_PIXELS = []
FISH_WIDTH  = 18

def load_fish_sprite():
    global FISH_WIDTH, FISH_PIXELS
    sprite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pixilart-drawing.png')
    if os.path.exists(sprite_path):
        img = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
        if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0)
            if len(xs) > 0:
                min_x, max_x = np.min(xs), np.max(xs)
                anchor_x_img = min_x
                anchor_y_img = (np.min(ys) + np.max(ys)) // 2
                FISH_WIDTH   = max_x - min_x + 1
                
                for y, x in zip(ys, xs):
                    b, g, r, a = img[y, x]
                    dy = y - anchor_y_img
                    dx = x - anchor_x_img
                    FISH_PIXELS.append((dy, dx, (int(r), int(g), int(b))))
                print(f"[INFO] Sprite caricato con successo ({len(FISH_PIXELS)} pixel, larghezza {FISH_WIDTH})")

# Esegui caricamento all'avvio
load_fish_sprite()

# Ogni quanti frame avanza di 1 pixel — 3 @ 60fps ≈ 2.5s per attraversare i 32px
FISH_STEP_FRAMES  = 3
FISH_COOLDOWN_SEC = 5.0

# Stati della macchina a stati
FISH_IDLE      = 0
FISH_ANIMATING = 1
FISH_COOLDOWN  = 2

def draw_fish(matrix, anchor_x, anchor_y, body_color):
    """
    Disegna il pesce prelevato dal PNG pixel-perfect.
    Ignora body_color per mantenere i colori originali esatti del file.
    """
    for dy, dx, color in FISH_PIXELS:
        r, c = anchor_y + dy, anchor_x + dx
        if 0 <= r < 32 and 0 <= c < 32:
            matrix[r, c] = color

# ============================================================
# TRACKER  (KALMAN + HUNGARIAN)
# ============================================================
class TrackedObject:
    def __init__(self, start_pt, color_name, object_id):
        self.id                 = object_id
        self.color              = color_name
        self.disappeared_frames = 0
        self.hits               = 1

        # Stato: [x, y, vx, vy]  |  Misura: [x, y]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix  = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        # Q alto (0.15) → segue cambi bruschi di direzione
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.15
        # R esplicito → bilancia fiducia misura vs predizione
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.5
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32) * 0.1

        self.kf.statePre  = np.array([[start_pt[0]], [start_pt[1]], [0.], [0.]], np.float32)
        self.kf.statePost = np.array([[start_pt[0]], [start_pt[1]], [0.], [0.]], np.float32)
        self.predicted_pos = start_pt

    def speed(self):
        """Velocità stimata in px/frame (coordinate full-res) dal vettore Kalman."""
        vx = float(self.kf.statePost[2, 0])
        vy = float(self.kf.statePost[3, 0])
        return math.sqrt(vx * vx + vy * vy)

    def predict(self):
        p = self.kf.predict()
        self.predicted_pos = (int(p[0, 0]), int(p[1, 0]))
        return self.predicted_pos

    def correct(self, pt):
        self.kf.correct(np.array([[pt[0]], [pt[1]]], np.float32))
        self.predicted_pos      = pt
        self.disappeared_frames = 0
        self.hits              += 1


class MarbleTracker:
    def __init__(self, max_distance=200, max_disappeared=12):
        self.next_object_id  = 0
        self.objects         = {}
        self.max_distance    = max_distance
        self.max_disappeared = max_disappeared

    def register(self, pt, color_name):
        obj = TrackedObject(pt, color_name, self.next_object_id)
        self.objects[self.next_object_id] = obj
        self.next_object_id += 1

    def deregister(self, object_id):
        self.objects.pop(object_id, None)

    def update(self, input_centroids, input_colors):
        predicted = {oid: obj.predict() for oid, obj in list(self.objects.items())}

        if not input_centroids:
            for oid, obj in list(self.objects.items()):
                obj.disappeared_frames += 1
                if obj.disappeared_frames > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        if not self.objects:
            for pt, col in zip(input_centroids, input_colors):
                self.register(pt, col)
            return self.objects

        object_ids = list(self.objects.keys())
        obj_pts    = [predicted[oid] for oid in object_ids]
        D          = dist.cdist(np.array(obj_pts), np.array(input_centroids))
        rows, cols = linear_sum_assignment(D)

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            # max_distance adattivo: scala con la velocità stimata dal Kalman.
            # Una biglia veloce copre più pixel inter-frame → soglia più larga.
            speed_px      = self.objects[object_ids[r]].speed()
            effective_dist = self.max_distance + speed_px * 3.0
            if D[r, c] > effective_dist:
                continue
            oid = object_ids[r]
            self.objects[oid].correct(input_centroids[c])
            if input_colors[c]:
                self.objects[oid].color = input_colors[c]
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(D.shape[0])) - used_rows:
            oid = object_ids[r]
            self.objects[oid].disappeared_frames += 1
            if self.objects[oid].disappeared_frames > self.max_disappeared:
                self.deregister(oid)

        for c in set(range(D.shape[1])) - used_cols:
            self.register(input_centroids[c], input_colors[c])

        return self.objects


# ============================================================
# COMUNICAZIONE ARDUINO
# ============================================================
def create_arduino_serial():
    if not ARDUINO_ENABLED or not HAS_SERIAL:
        return None
    port = ARDUINO_PORT
    if port == "auto":
        raw_candidates = (glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*') +
                          glob.glob('/dev/cu.usbmodem*') + glob.glob('/dev/cu.usbserial*') +
                          glob.glob('/dev/tty.*') + glob.glob('/dev/cu.*'))
        # Filtra le porte virtuali del Mac
        candidates = [c for c in raw_candidates if "debug-console" not in c and "Bluetooth" not in c and "BTH" not in c]
        if not candidates:
            print("[!] Nessuna porta seriale trovata! (Arduino disconnesso?)")
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
        print(f"[!] Errore connessione Arduino: {e}")
        return None

MAGIC_HEADER  = bytes([0xFF, 0x4C, 0x45])
arduino_ready = True

def send_matrix_state(ser, matrix_rgb):
    global arduino_ready
    if ser is None:
        return
    if ser.in_waiting > 0:
        resp = ser.read_all()
        if b'K' in resp or b'READY' in resp:
            arduino_ready = True
    if not arduino_ready:
        return
    rgb_gamma     = gamma_table[matrix_rgb]
    mapped_pixels = rgb_gamma[LED_MAP_Y, LED_MAP_X]
    ser.write(MAGIC_HEADER + mapped_pixels.tobytes())
    arduino_ready = False


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== AVVIO TRACKING BIGLIE V3 — HIGH ACCURACY ===")
    ser = create_arduino_serial()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Errore apertura webcam")
        return

    cap.set(cv2.CAP_PROP_FOURCC,         cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,    640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,   480)
    cap.set(cv2.CAP_PROP_FPS,            60)
    cap.set(cv2.CAP_PROP_AUTO_WB,        0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,  0)

    # 0.75 → 480×360: +2.25x risoluzione rispetto a 0.5, centroidi più precisi
    SCALE_FACTOR = 0.75
    PROC_W = int(640 * SCALE_FACTOR)   # 480
    PROC_H = int(480 * SCALE_FACTOR)   # 360

    # MOG2: impara il background statico in ~200 frame, poi restituisce solo
    # le regioni che si muovono. varThreshold=40 calibrato per luce mista indoor.
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=400, varThreshold=40, detectShadows=False)

    # CLAHE sul canale V: normalizza i riflessi speculari sulla superficie lucida.
    # Senza CLAHE un punto bianco brillante "distrugge" il colore percepito.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

    tracker = MarbleTracker(max_distance=200, max_disappeared=12)

    # Kernel ellittici: più precisi di quelli rettangolari per oggetti circolari
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Precalcola aree minima/massima dai raggi fisici
    min_area = math.pi * MIN_R  * MIN_R
    max_area = math.pi * MAX_R * MAX_R

    # --- WARMUP BACKGROUND MODEL ---
    # Tenere la scena ferma (senza biglie) durante il warmup.
    # MOG2 assorbirebbe la biglia statica nel background se è già presente.
    WARMUP_FRAMES = 200
    print(f"[INFO] Calibrazione background ({WARMUP_FRAMES} frame)...")
    print("[INFO] Tenere la scena ferma senza biglie.")
    for i in range(WARMUP_FRAMES):
        ret, f = cap.read()
        if not ret:
            break
        f    = cv2.flip(f, 1)
        proc = cv2.resize(f, (PROC_W, PROC_H))
        proc = cv2.medianBlur(proc, 5)
        bg_sub.apply(proc)
        if i % 50 == 0:
            print(f"  ... {i}/{WARMUP_FRAMES}")
    print("[INFO] Pronto. Introduci le biglie.")
    print("[INFO] Tasti: 'q' esci | 'd' toggle debug mask")

    debug_mode  = True
    fps_history = deque(maxlen=30)
    prev_time   = time.time()

    # --- STATE MACHINE ANIMAZIONE PESCE ---
    fish_state      = FISH_IDLE
    fish_x          = -FISH_WIDTH   # posizione ancora del pesce (colonna coda)
    fish_frame_tick = 0             # conta i frame per il passo del pesce
    cooldown_start  = 0.0           # timestamp inizio cooldown
    fish_color_rgb  = (255, 255, 255)  # colore attivo del pesce (sovrascritto al trigger)
    matrix_render   = np.zeros((32, 32, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        curr_time = time.time()
        dt        = curr_time - prev_time
        fps_history.append(1.0 / dt if dt > 0 else 60.0)
        prev_time = curr_time

        # ── 1. PREPROCESSING ──────────────────────────────────────────────
        proc = cv2.resize(frame, (PROC_W, PROC_H))
        # medianBlur preserva i bordi cromatici: il Gaussian li smussa
        # spostando il centroide verso zone adiacenti di colore diverso
        proc = cv2.medianBlur(proc, 5)

        # ── 2. MOTION MASK ────────────────────────────────────────────────
        fg = bg_sub.apply(proc)
        _, fg       = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(fg, np.ones((22, 22), np.uint8), iterations=2)

        # ROI Kalman adattivo: il raggio cresce con la velocità stimata.
        # Mantiene tracking sia a bassa velocità (MOG2 non rileva movimento)
        # che ad alta velocità (la biglia si sposta molto tra un frame e l'altro).
        fast_roi_mask = np.zeros((PROC_H, PROC_W), dtype=np.uint8)
        for obj in tracker.objects.values():
            px      = int(max(0, min(PROC_W - 1, obj.predicted_pos[0] * SCALE_FACTOR)))
            py      = int(max(0, min(PROC_H - 1, obj.predicted_pos[1] * SCALE_FACTOR)))
            spd     = obj.speed()
            roi_r   = min(int(55 + spd * SPEED_ROI_FACTOR * SCALE_FACTOR), 130)
            cv2.circle(motion_mask, (px, py), roi_r, 255, -1)
            # fast_roi_mask: zona dove si usano soglie geometriche più permissive
            if spd > SPEED_THRESHOLD:
                cv2.circle(fast_roi_mask, (px, py), roi_r, 255, -1)

        # ── 3. CLAHE su canale V ──────────────────────────────────────────
        hsv    = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        hsv_eq  = cv2.merge([h, s, clahe.apply(v)])

        # ── 4. RILEVAMENTO COLORE + FILTRI GEOMETRICI ─────────────────────
        input_centroids = []
        input_colors    = []

        for color_name, params in COLOR_RANGES.items():
            if color_name == 'red':
                m1    = cv2.inRange(hsv_eq, params['lower1'], params['upper1'])
                m2    = cv2.inRange(hsv_eq, params['lower2'], params['upper2'])
                cmask = cv2.bitwise_or(m1, m2)
            else:
                cmask = cv2.inRange(hsv_eq, params['lower'], params['upper'])

            cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN,  kernel_open)
            cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, kernel_close)
            # Applica motion mask: analizza solo regioni in movimento + ROI predittive
            cmask = cv2.bitwise_and(cmask, motion_mask)

            contours, _ = cv2.findContours(
                cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                # Determina se questo contorno è dentro una zona "fast":
                # centroide grezzo a basso costo (prima di qualsiasi filtro)
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    in_fast = fast_roi_mask[
                        int(M["m01"] / M["m00"]),
                        int(M["m10"] / M["m00"])
                    ] > 0
                else:
                    in_fast = False

                # Soglie adattive: dentro fast_roi → più permissive (biglia veloce/sfocata)
                #                  fuori fast_roi → strette (protezione falsi positivi)
                circ_thr     = CIRC_MIN_FAST  if in_fast else CIRC_MIN
                solid_thr    = SOLID_MIN_FAST if in_fast else SOLID_MIN
                max_area_eff = math.pi * (MAX_R_FAST if in_fast else MAX_R) ** 2

                # FILTRO 1 — AREA
                if not (min_area < area < max_area_eff):
                    continue

                # FILTRO 2 — CIRCOLARITÀ: 4π·A/P²
                perimeter = cv2.arcLength(cnt, True)
                if perimeter < 1e-5:
                    continue
                circularity = 4.0 * math.pi * area / (perimeter ** 2)
                if circularity < circ_thr:
                    continue

                # FILTRO 3 — SOLIDITÀ: area / area_convex_hull
                hull      = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area < 1e-5:
                    continue
                solidity = area / hull_area
                if solidity < solid_thr:
                    continue

                # CENTRO via minEnclosingCircle
                (cx_s, cy_s), radius = cv2.minEnclosingCircle(cnt)
                cx = int(cx_s / SCALE_FACTOR)
                cy = int(cy_s / SCALE_FACTOR)
                input_centroids.append((cx, cy))
                input_colors.append(color_name)

                if debug_mode:
                    disp_r     = int(radius / SCALE_FACTOR)
                    dot_color  = (0, 140, 255) if in_fast else (0, 220, 0)  # arancio=fast, verde=normal
                    cv2.circle(frame, (cx, cy), disp_r, dot_color, 1)
                    mode_tag   = "F" if in_fast else "N"
                    label      = f"{mode_tag} c{circularity:.2f} s{solidity:.2f} r{int(radius)}"
                    cv2.putText(frame, label,
                                (cx + disp_r + 3, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, dot_color, 1)

        # ── 5. TRACKING (Kalman + Hungarian) ──────────────────────────────
        active_objects = tracker.update(input_centroids, input_colors)

        # ── 6. OVERLAY RILEVAMENTO (solo preview webcam, non influenza LED) ─────
        frame_h, frame_w = frame.shape[:2]
        for object_id, obj in active_objects.items():
            if obj.hits < 3:
                continue
            cx, cy    = obj.predicted_pos
            color_bgr = COLOR_RANGES[obj.color]['bgr']
            cv2.line(frame, (cx - 10, cy),      (cx + 10, cy),      color_bgr, 2)
            cv2.line(frame, (cx,      cy - 10), (cx,      cy + 10), color_bgr, 2)
            cv2.putText(frame, f"ID:{object_id}",
                        (cx - 10, cy - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)

        # ── 7. STATE MACHINE ANIMAZIONE PESCE ─────────────────────────────
        ball_visible = any(obj.hits >= 3 for obj in active_objects.values())
        now          = curr_time

        if fish_state == FISH_IDLE:
            matrix_render = np.zeros((32, 32, 3), dtype=np.uint8)
            if ball_visible:
                # Cattura il colore RGB della prima pallina valida rilevata
                fish_color_rgb = next(
                    COLOR_RANGES[obj.color]['rgb']
                    for obj in active_objects.values() if obj.hits >= 3
                )
                fish_state      = FISH_ANIMATING
                fish_x          = -FISH_WIDTH
                fish_frame_tick = 0
                print(f"[FISH] Avviato — colore {fish_color_rgb}")

        elif fish_state == FISH_ANIMATING:
            matrix_render = np.zeros((32, 32, 3), dtype=np.uint8)
            # Movimento sinusoidale verticale: simula il nuoto
            fish_cy = 16 + int(3 * math.sin(fish_x * 0.25))
            draw_fish(matrix_render, fish_x, fish_cy, fish_color_rgb)

            fish_frame_tick += 1
            if fish_frame_tick >= FISH_STEP_FRAMES:
                fish_frame_tick = 0
                fish_x         += 1

            if fish_x > 32:
                # Pesce uscito dal bordo destro → spegni e inizia cooldown
                fish_state     = FISH_COOLDOWN
                cooldown_start = now
                matrix_render  = np.zeros((32, 32, 3), dtype=np.uint8)
                print(f"[FISH] Cooldown {FISH_COOLDOWN_SEC}s")

        elif fish_state == FISH_COOLDOWN:
            matrix_render = np.zeros((32, 32, 3), dtype=np.uint8)
            if now - cooldown_start >= FISH_COOLDOWN_SEC:
                fish_state = FISH_IDLE

        # ── 8. OVERLAY STATISTICHE ────────────────────────────────────────
        avg_fps = sum(fps_history) / len(fps_history)
        state_labels = {FISH_IDLE: "IDLE", FISH_ANIMATING: "FISH", FISH_COOLDOWN: "COOLDOWN"}
        cooldown_rem = max(0.0, FISH_COOLDOWN_SEC - (now - cooldown_start)) if fish_state == FISH_COOLDOWN else 0
        state_str    = f"{state_labels[fish_state]}" + (f" {cooldown_rem:.1f}s" if fish_state == FISH_COOLDOWN else "")
        cv2.putText(frame,
                    f"FPS:{int(avg_fps)}  OBJ:{len(active_objects)}  {state_str}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Marble Tracking V3 - High Accuracy", frame)

        if debug_mode:
            motion_dbg = cv2.resize(motion_mask, (320, 240))
            cv2.imshow("Motion Mask (MOG2 + ROI Kalman)", motion_dbg)
        else:
            cv2.destroyWindow("Motion Mask (MOG2 + ROI Kalman)")

        matrix_preview     = cv2.resize(matrix_render, (320, 320),
                                        interpolation=cv2.INTER_NEAREST)
        matrix_preview_bgr = cv2.cvtColor(matrix_preview, cv2.COLOR_RGB2BGR)
        cv2.imshow("Matrix 32x32 DBG", matrix_preview_bgr)

        send_matrix_state(ser, matrix_render)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        # Spegni tutti i LED prima di chiudere: invia frame nero
        black_matrix = np.zeros((32, 32, 3), dtype=np.uint8)
        mapped       = gamma_table[black_matrix][LED_MAP_Y, LED_MAP_X]
        ser.write(MAGIC_HEADER + mapped.tobytes())
        time.sleep(0.1)  # attesa minima per garantire la ricezione
        ser.close()
        print("[OK] LED spenti. Connessione chiusa.")
    


if __name__ == "__main__":
    main()
