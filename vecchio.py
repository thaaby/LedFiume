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
ARDUINO_COLS               = 56          # 7 pannelli × 8 = 56
ARDUINO_PANEL_W            = 8
ARDUINO_PANEL_H            = 32
ARDUINO_PANELS_COUNT       = 7           # era 4
ARDUINO_PANEL_ORDER        = [6, 5, 4, 3, 2, 1, 0]
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
        # Rosso: 0-10 + 170-180 = 20° (era 14°). S_min 70 per tollerare riflessi.
        # Non si estende oltre Hue 10 per non entrare in zona arancione-tubo.
        'lower1': np.array([0,   70,  30]),
        'upper1': np.array([10,  255, 255]),
        'lower2': np.array([170, 70,  30]),
        'upper2': np.array([180, 255, 255]),
        'bgr': (0, 0, 255),
        'rgb': (255, 0, 0),
    },
    'blue': {
        # Azzurro
        'lower': np.array([85,  80,  40]),
        'upper': np.array([130, 255, 255]),
        'bgr': (255, 0, 0),
        'rgb': (0, 0, 255),
    },
    'yellow': {
        # Giallo: H da 20 (era 24). S_min 60 per tollerare riflessi.
        # Non scende sotto 20 per non entrare in zona arancione-tubo.
        'lower': np.array([20,  60,  50]),
        'upper': np.array([45, 255, 255]),
        'bgr': (0, 255, 255),
        'rgb': (255, 255, 0),
    },
}

# ============================================================
# PARAMETRI GEOMETRICI BIGLIA
# ============================================================
# Biglia ~2.3cm diametro.
# Calcolo a SCALE_FACTOR=0.75 su webcam 640x480 con FOV ~70°:
#   a 25cm → raggio ~16px | a 50cm → raggio ~8px | a 80cm → ~5px | a 100cm → ~4px
MIN_R    = 6
MAX_R    = 25
CIRC_MIN  = 0.50    # Abbassato: riflessi/occlusione nel tubo distorcono il contorno
SOLID_MIN = 0.60    # Abbassato: biglie adiacenti e bordi tubo riducono solidità

CIRC_MIN_FAST  = 0.45
SOLID_MIN_FAST = 0.65
MAX_R_FAST     = 48

SPEED_THRESHOLD   = 18
SPEED_ROI_FACTOR  = 2.2

# ============================================================
# SPRITE PESCE — caricato da pixilart-drawing.png (32×32, RGBA)
# ============================================================
# Pixel estratti al centro del bounding box del disegno.
# I colori arancione (255,126,0) e rosso (237,28,36) vengono sostituiti
# con il colore della pallina rilevata.
# facing_right=False specchia orizzontalmente (dx negato).

# Colori del sprite da rimpiazzare col colore pallina
_FISH_BODY_COLORS = {(255, 126, 0), (237, 28, 36)}

# Lista di (dy, dx, rgb) con origine al centro del bounding box
_FISH_PIXELS: list = []

def _load_fish_sprite():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pixilart-drawing.png')
    img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim < 3 or img.shape[2] != 4:
        print(f"[WARN] {path} non trovato o non RGBA — sprite non caricato")
        return
    ys, xs = np.where(img[:, :, 3] > 0)
    if len(xs) == 0:
        return
    cx = (int(xs.min()) + int(xs.max())) // 2
    cy = (int(ys.min()) + int(ys.max())) // 2
    for y, x in zip(ys.tolist(), xs.tolist()):
        b, g, r, _ = img[y, x]
        _FISH_PIXELS.append((y - cy, x - cx, (int(r), int(g), int(b))))
    print(f"[OK] Sprite PNG: {len(_FISH_PIXELS)} pixel, centro PNG=({cx},{cy})")

_load_fish_sprite()

# Soglia velocità Kalman (px/frame, full-res) per aggiornare la direzione.
# Sotto soglia → mantieni la direzione corrente (evita flickering).
VX_DIRECTION_THRESHOLD = 3.0


def draw_fish(matrix, cx, cy, body_color, facing_right=True):
    """
    Disegna il pesciolino dal PNG centrato in (cx, cy) sulla matrice 32×32.
    facing_right=False specchia orizzontalmente il pesce.
    body_color: RGB tuple — sostituisce l'arancione/rosso del PNG.
    """
    rows, cols = matrix.shape[:2]
    s = 1 if facing_right else -1
    for dy, dx, color in _FISH_PIXELS:
        r = cy - dy   # negato: Y immagine va giù, Y LED va su
        c = cx + s * dx
        if 0 <= r < rows and 0 <= c < cols:
            matrix[r, c] = body_color if color in _FISH_BODY_COLORS else color

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
        # Q alto (0.3) → segue cambi rapidi di direzione/velocità
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.3
        # R basso (1.0) → misure più fidate, meno smoothing
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32) * 0.1

        self.kf.statePre  = np.array([[start_pt[0]], [start_pt[1]], [0.], [0.]], np.float32)
        self.kf.statePost = np.array([[start_pt[0]], [start_pt[1]], [0.], [0.]], np.float32)
        self.predicted_pos = start_pt
        self.ema_pos = start_pt
        self.last_x  = start_pt[0]

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
        
        # EMA Filter per bypassare il ritardo del Kalman (Latenza 0)
        alpha = 0.80
        self.ema_pos = (
            self.ema_pos[0] * (1 - alpha) + pt[0] * alpha,
            self.ema_pos[1] * (1 - alpha) + pt[1] * alpha
        )
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

    print("[INFO] All'avvio, seleziona il TUBO con il mouse e premi SPAZIO o INVIO.")
    tube_mask = np.zeros((PROC_H, PROC_W), dtype=np.uint8)
    for _ in range(5): cap.read()  # Scarta i primissimi frame neri/corrotti
    ret, start_frame = cap.read()
    if ret:
        start_frame = cv2.flip(start_frame, 1)
        start_proc = cv2.resize(start_frame, (PROC_W, PROC_H))
        roi = cv2.selectROI("Seleziona il TUBO (FIUME), poi premi SPAZIO", start_proc, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Seleziona il TUBO (FIUME), poi premi SPAZIO")
        cv2.waitKey(1)
        
        tube_h_lower = np.array([0, 0, 0])
        tube_h_upper = np.array([0, 0, 0])
        if roi[2] > 0 and roi[3] > 0:
            x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            cv2.rectangle(tube_mask, (x, y), (x+w, y+h), 255, -1)
            print(f"[OK] Maschera Tubo applicata. Tracking confinato al rettangolo! (w:{w}, h:{h})")
            
            # --- CAMPIONAMENTO COLORE TUBO (H + S + V) ---
            tube_roi_bgr = start_proc[y:y+h, x:x+w]
            tube_roi_hsv = cv2.cvtColor(tube_roi_bgr, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([tube_roi_hsv], [0], None, [180], [0, 180])
            dominant_h = int(np.argmax(hist_h))
            median_s   = int(np.median(tube_roi_hsv[:, :, 1]))
            median_v   = int(np.median(tube_roi_hsv[:, :, 2]))

            # Range HSV stretto: esclude solo pixel MOLTO simili al tubo in H, S e V.
            # Una biglia rossa/gialla ha S/V diversi dal tubo → non viene mascherata.
            margin_h = 8
            margin_s = 40
            margin_v = 40
            tube_h_lower = np.array([max(0,   dominant_h - margin_h),
                                     max(0,   median_s   - margin_s),
                                     max(0,   median_v   - margin_v)])
            tube_h_upper = np.array([min(179, dominant_h + margin_h),
                                     min(255, median_s   + margin_s),
                                     min(255, median_v   + margin_v)])
            print(f"[OK] Colore Tubo campionato! H:{dominant_h} S:{median_s} V:{median_v}"
                  f" → range H[{tube_h_lower[0]}-{tube_h_upper[0]}]"
                  f" S[{tube_h_lower[1]}-{tube_h_upper[1]}]"
                  f" V[{tube_h_lower[2]}-{tube_h_upper[2]}]")
            
        else:
            print("[WARN] Nessuna selezione: tracking sull'intero perimetro.")
            tube_mask.fill(255)
    else:
        tube_mask.fill(255)

    # CLAHE sul canale V per gestire i riflessi lucidi delle biglie
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

    tracker = MarbleTracker(max_distance=250, max_disappeared=20)

    # Kernel ellittici: precisi per cerchi reali
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    min_area = math.pi * MIN_R  * MIN_R
    max_area = math.pi * MAX_R * MAX_R

    print("[INFO] Pronto! Avvio tracking. Introduci le biglie.")
    print("[INFO] Tasti: 'q' esci | 'd' toggle debug mask")

    debug_mode  = True
    fps_history = deque(maxlen=30)
    prev_time   = time.time()

    # --- TRACKING 1:1 PESCE ---
    fish_facing_right = True        # direzione corrente del pesce
    matrix_render     = np.zeros((ARDUINO_ROWS, ARDUINO_COLS, 3), dtype=np.uint8)

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
        proc = cv2.medianBlur(proc, 3)

        # ── 2. PREPARAZIONE MASCHERA TUBO / ROI ───────────────────────────
        # Anziché usare MOG2 (che creava forme quadrate sgranate all'inizio),
        # ci affidiamo ESCLUSIVAMENTE al recinto del tubo. 
        # Questo garantisce che la forma letta sia fin dal primo istante quella fisica 1:1.
        master_mask = tube_mask.copy()

        # ROI Kalman adattivo (assiste i filtri geometrici rilassati per biglie veloci)
        fast_roi_mask = np.zeros((PROC_H, PROC_W), dtype=np.uint8)
        for obj in tracker.objects.values():
            px  = int(max(0, min(PROC_W - 1, obj.predicted_pos[0] * SCALE_FACTOR)))
            py  = int(max(0, min(PROC_H - 1, obj.predicted_pos[1] * SCALE_FACTOR)))
            spd = obj.speed()
            if spd > SPEED_THRESHOLD:
                roi_r = min(int(70 + spd * SPEED_ROI_FACTOR * SCALE_FACTOR), 150)
                cv2.circle(fast_roi_mask, (px, py), roi_r, 255, -1)
                
        fast_roi_mask = cv2.bitwise_and(fast_roi_mask, tube_mask)

        # ── 3. CLAHE su canale V ──────────────────────────────────────────
        hsv    = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        hsv_eq  = cv2.merge([h, s, clahe.apply(v)])

        # Maschera tubo: esclude pixel simili al tubo in H+S+V (range stretto)
        detected_tube = cv2.inRange(hsv_eq, tube_h_lower, tube_h_upper)
        ignore_tube_mask = cv2.bitwise_not(detected_tube)

        # ── 4. RILEVAMENTO COLORE + FILTRI GEOMETRICI ─────────────────────
        input_centroids = []
        input_colors    = []
        debug_mask_all  = np.zeros((PROC_H, PROC_W), dtype=np.uint8)

        for color_name, params in COLOR_RANGES.items():
            if color_name == 'red':
                m1    = cv2.inRange(hsv_eq, params['lower1'], params['upper1'])
                m2    = cv2.inRange(hsv_eq, params['lower2'], params['upper2'])
                cmask = cv2.bitwise_or(m1, m2)
            else:
                cmask = cv2.inRange(hsv_eq, params['lower'], params['upper'])

            cmask = cv2.bitwise_and(cmask, master_mask)
            cmask = cv2.bitwise_and(cmask, ignore_tube_mask)
            
            cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN,  kernel_open)
            cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, kernel_close)
            debug_mask_all = cv2.bitwise_or(debug_mask_all, cmask)

            contours, _ = cv2.findContours(
                cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                # Centroide grezzo per determinare se dentro fast ROI
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    in_fast = fast_roi_mask[
                        int(M["m01"] / M["m00"]),
                        int(M["m10"] / M["m00"])
                    ] > 0
                else:
                    in_fast = False

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
                    dot_color  = (0, 140, 255) if in_fast else (0, 220, 0)
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
            if obj.hits < 2:
                continue
            cx, cy    = obj.predicted_pos
            color_bgr = COLOR_RANGES[obj.color]['bgr']
            cv2.line(frame, (cx - 10, cy),      (cx + 10, cy),      color_bgr, 2)
            cv2.line(frame, (cx,      cy - 10), (cx,      cy + 10), color_bgr, 2)
            cv2.putText(frame, f"ID:{object_id}",
                        (cx - 10, cy - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)

        # ── 7. TRACKING 1:1 ZERO-LATENCY ──────────────────────────────────
        matrix_render = np.zeros((ARDUINO_ROWS, ARDUINO_COLS, 3), dtype=np.uint8)
        ball_visible  = False

        for obj in active_objects.values():
            if obj.hits < 2:
                continue
            ball_visible = True

            # POSIZIONE CRUDA (Zero-Latenza) INVECE DEL KALMAN (che causa ritardo)
            bx, by = obj.ema_pos

            # DISEZIONE FLUIDA
            vx_raw = bx - obj.last_x
            if abs(vx_raw) > 1.5:  
                fish_facing_right = vx_raw > 0
            obj.last_x = bx

            # CORREZIONE ASPECT RATIO
            # Webcam: 1.33 (640x480) | LEDs: 1.75 (56x32)
            # Senza compensazione, il pesce si deforma spostandosi verticalmente.
            eff_h = int(frame_w * (ARDUINO_ROWS / ARDUINO_COLS))  # 640 * 32/56 = ~365px
            y_offset = (frame_h - eff_h) / 2.0  # (480-365)/2 = ~57px superiore/inferiore
            mapped_y = by - y_offset

            led_x = max(0, min(ARDUINO_COLS - 1, int(bx * ARDUINO_COLS / frame_w)))
            led_y = max(0, min(ARDUINO_ROWS - 1, int(mapped_y * ARDUINO_ROWS / eff_h)))

            fish_color_rgb = COLOR_RANGES[obj.color]['rgb']
            draw_fish(matrix_render, led_x, led_y, fish_color_rgb, fish_facing_right)
            break  # una biglia alla volta

        # ── 8. OVERLAY STATISTICHE ────────────────────────────────────────
        avg_fps = sum(fps_history) / len(fps_history)
        dir_str = "→" if fish_facing_right else "←"
        trk_str = f"TRACK {dir_str}" if ball_visible else "IDLE"
        cv2.putText(frame,
                    f"FPS:{int(avg_fps)}  OBJ:{len(active_objects)}  {trk_str}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Marble Tracking V3 - High Accuracy", frame)

        if debug_mode:
            cmask_dbg = cv2.resize(debug_mask_all, (320, 240))
            cv2.imshow("Color Mask (All Colors)", cmask_dbg)
        else:
            try: cv2.destroyWindow("Color Mask (All Colors)")
            except: pass

        matrix_preview     = cv2.resize(matrix_render, (560, 320),
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
        black_matrix = np.zeros((ARDUINO_ROWS, ARDUINO_COLS, 3), dtype=np.uint8)
        mapped       = gamma_table[black_matrix][LED_MAP_Y, LED_MAP_X]
        ser.write(MAGIC_HEADER + mapped.tobytes())
        time.sleep(0.1)  # attesa minima per garantire la ricezione
        ser.close()
        print("[OK] LED spenti. Connessione chiusa.")
    


if __name__ == "__main__":
    main()

