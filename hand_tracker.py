"""
hand_tracker.py - Tracking mani con MediaPipe Tasks API + riconoscimento gesti
Gesti: pinch (disegna), peace/V (cambia colore), thumbs down (cancella),
       indice esteso (gomma di precisione)
Usa VIDEO running mode per tracking temporale stabile.
Smoothing con 1-Euro Filter (anti-tremore adattivo).
Pinch normalizzato per dimensione mano (prospettiva-invariante).
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import os
import time
import math

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'hand_landmarker.task')

# ── Pinch (normalizzato per hand_size) ────────────────────────
PINCH_START_RATIO       = 0.18    # le punte devono essere molto vicine
PINCH_STOP_RATIO        = 0.28    # isteresi per evitare flickering

# ── Gesti (frame per attivazione) ────────────────────────────
PEACE_ACTIVATE_FRAMES   = 2       # via di mezzo
PEACE_COOLDOWN_SEC      = 0.5     # secondi di cooldown dopo cambio colore
ERASER_ACTIVATE_FRAMES  = 6
THUMBS_DOWN_ACTIVATE_FRAMES = 2
PEACE_POST_DRAW_GRACE   = 10

# ── 1-Euro Filter (anti-tremore adattivo) ────────────────────
# min_cutoff basso = piu liscio a bassa velocita (filtra tremore)
# beta alto = piu reattivo a alta velocita (segue i tratti)
FILTER_MIN_CUTOFF       = 1.0     # Hz - smoothing a riposo
FILTER_BETA             = 0.5     # reattivita a movimenti veloci
FILTER_D_CUTOFF         = 1.0     # Hz - smoothing della derivata
FILTER_MIN_CUTOFF_DRAW  = 0.6     # piu liscio quando si disegna

# ── Stabilita ────────────────────────────────────────────────
HAND_LOST_RESET_FRAMES  = 5
STABILIZE_FRAMES        = 3
POSITION_JUMP_THRESHOLD = 0.25

# ── Connessioni scheletro ────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
FINGERTIP_IDS = {4, 8, 12, 16, 20}


# ==============================================================
# 1-Euro Filter - smoothing adattivo basato sulla velocita
# Movimenti lenti (tremore) -> pesantemente filtrati
# Movimenti veloci (tratti) -> seguiti fedelmente
# ==============================================================
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x, t):
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        dt = max(t - self._t_prev, 1e-6)

        # Derivata filtrata
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Cutoff adattivo: piu veloce il movimento, meno smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def set_min_cutoff(self, val):
        self.min_cutoff = val


class HandState:
    """Stato di una singola mano rilevata in un frame."""
    __slots__ = ('hand_label', 'raw_x', 'raw_y', 'drawing',
                 'peace_sign', 'thumbs_down', 'precision_erasing',
                 'landmarks', 'gesture_label', 'pinch_dist')

    def __init__(self):
        self.hand_label = "Right"
        self.raw_x = 0.5
        self.raw_y = 0.5
        self.drawing = False
        self.peace_sign = False
        self.thumbs_down = False
        self.precision_erasing = False
        self.landmarks = None
        self.gesture_label = ""
        self.pinch_dist = 1.0


class _HandData:
    """Stato interno per-mano tra frame."""
    def __init__(self):
        self.peace_counter = 0
        self.eraser_counter = 0
        self.thumbs_counter = 0
        self.post_draw_grace = 0
        self.is_pinching = False
        self.lost_frames = 0
        self.stabilize_frames = 0
        self.peace_cooldown_until = 0.0  # timestamp fino a cui peace e' bloccato
        # 1-Euro filters (uno per asse)
        self.filter_x = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF)
        self.filter_y = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF)
        self.prev_raw_x = None
        self.prev_raw_y = None

    def reset_filters(self):
        self.filter_x.reset()
        self.filter_y.reset()
        self.prev_raw_x = None
        self.prev_raw_y = None


class HandTracker:
    def __init__(self, canvas_width=32, canvas_height=8, num_hands=1):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.num_hands = num_hands
        self._hand_data = {}
        self._detector = None
        self._last_ts = 0
        self._create_detector()

    def _create_detector(self):
        if self._detector is not None:
            self._detector.close()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._last_ts = 0

    def set_num_hands(self, n):
        if n == self.num_hands:
            return
        self.num_hands = n
        self._hand_data.clear()
        self._create_detector()
        print(f"[HANDS] Tracking {n} mano/i")

    def release(self):
        if self._detector:
            self._detector.close()

    # ── Processing ────────────────────────────────────────────

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int(time.time() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        now = time.time()  # per 1-Euro filter

        results = self._detector.detect_for_video(mp_image, ts)

        detected_labels = set()
        hand_states = []

        if results.hand_landmarks:
            for i, hand_lm in enumerate(results.hand_landmarks):
                label = "Right"
                if i < len(results.handedness):
                    label = results.handedness[i][0].category_name
                detected_labels.add(label)
                landmarks = hand_lm

                if label not in self._hand_data:
                    self._hand_data[label] = _HandData()
                hd = self._hand_data[label]

                # Mano riapparsa dopo assenza
                if hd.lost_frames >= HAND_LOST_RESET_FRAMES:
                    hd.reset_filters()
                    hd.stabilize_frames = STABILIZE_FRAMES
                    hd.is_pinching = False
                    hd.peace_counter = 0
                    hd.eraser_counter = 0
                    hd.thumbs_counter = 0
                hd.lost_frames = 0

                state = HandState()
                state.hand_label = label
                state.landmarks = [(lm.x, lm.y) for lm in landmarks]

                # ── Posizione: centro tra pollice(4) e indice(8) ──
                # Quando si fa pinch il punto di disegno e' nel mezzo
                raw_x = (landmarks[4].x + landmarks[8].x) / 2.0
                raw_y = (landmarks[4].y + landmarks[8].y) / 2.0

                # Rileva salto anomalo -> reset filtro
                if hd.prev_raw_x is not None:
                    jump = math.sqrt((raw_x - hd.prev_raw_x) ** 2 +
                                     (raw_y - hd.prev_raw_y) ** 2)
                    if jump > POSITION_JUMP_THRESHOLD:
                        hd.reset_filters()
                hd.prev_raw_x = raw_x
                hd.prev_raw_y = raw_y

                # Smoothing adattivo: piu liscio quando si disegna
                mc = FILTER_MIN_CUTOFF_DRAW if hd.is_pinching else FILTER_MIN_CUTOFF
                hd.filter_x.set_min_cutoff(mc)
                hd.filter_y.set_min_cutoff(mc)

                state.raw_x = hd.filter_x(raw_x, now)
                state.raw_y = hd.filter_y(raw_y, now)

                # ── Stabilizzazione ──────────────────────────
                if hd.stabilize_frames > 0:
                    hd.stabilize_frames -= 1
                    state.gesture_label = "..."
                    hand_states.append(state)
                    continue

                # ── Dimensione mano (per normalizzazione) ────
                wrist = landmarks[0]
                mid_mcp = landmarks[9]
                hand_size = math.sqrt(
                    (wrist.x - mid_mcp.x) ** 2 +
                    (wrist.y - mid_mcp.y) ** 2
                )
                hand_size = max(hand_size, 0.01)

                # ── Pinch normalizzato con isteresi ──────────
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                pinch_dist = math.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2
                )
                pinch_ratio = pinch_dist / hand_size
                state.pinch_dist = pinch_ratio

                if hd.is_pinching:
                    hd.is_pinching = pinch_ratio < PINCH_STOP_RATIO
                else:
                    hd.is_pinching = pinch_ratio < PINCH_START_RATIO
                is_pinching = hd.is_pinching

                # ── Dita estese ──────────────────────────────
                fingers_up = self._count_fingers(landmarks, label)

                # ── Peace/V (con soglie differenziate) ───────
                is_peace_raw = (self._is_peace_gesture(landmarks)
                                and not is_pinching)

                # ── Thumbs down ──────────────────────────────
                is_thumbs_raw = (self._is_thumbs_down(landmarks, fingers_up)
                                 and not is_pinching)

                # ── Eraser (solo indice) ─────────────────────
                is_eraser_raw = (fingers_up[1] and not fingers_up[2] and
                                 not fingers_up[3] and not fingers_up[4] and
                                 not is_pinching)

                # ── Debounce ─────────────────────────────────
                if is_pinching:
                    hd.post_draw_grace = PEACE_POST_DRAW_GRACE
                if hd.post_draw_grace > 0:
                    hd.post_draw_grace -= 1
                    is_peace_raw = False

                # Cooldown peace (2s dopo ogni cambio colore)
                if now < hd.peace_cooldown_until:
                    is_peace_raw = False

                if is_peace_raw:
                    hd.peace_counter += 1
                else:
                    hd.peace_counter = max(0, hd.peace_counter - 1)

                if is_eraser_raw:
                    hd.eraser_counter += 1
                else:
                    hd.eraser_counter = max(0, hd.eraser_counter - 3)

                if is_thumbs_raw:
                    hd.thumbs_counter += 1
                else:
                    hd.thumbs_counter = max(0, hd.thumbs_counter - 3)

                # ── Stato finale ─────────────────────────────
                state.drawing = is_pinching
                state.peace_sign = (hd.peace_counter >= PEACE_ACTIVATE_FRAMES)
                state.thumbs_down = (hd.thumbs_counter >= THUMBS_DOWN_ACTIVATE_FRAMES)
                state.precision_erasing = (hd.eraser_counter >= ERASER_ACTIVATE_FRAMES)

                if state.drawing:
                    state.gesture_label = "DRAW"
                elif state.precision_erasing:
                    state.gesture_label = "ERASE"
                elif state.peace_sign:
                    state.gesture_label = "COLOR"
                elif state.thumbs_down:
                    state.gesture_label = "CLEAR"
                else:
                    state.gesture_label = ""

                if state.peace_sign:
                    hd.peace_counter = 0
                    hd.peace_cooldown_until = now + PEACE_COOLDOWN_SEC
                if state.thumbs_down:
                    hd.thumbs_counter = 0

                hand_states.append(state)

        # Mani non rilevate
        for label, hd in self._hand_data.items():
            if label not in detected_labels:
                hd.lost_frames += 1
                hd.peace_counter = max(0, hd.peace_counter - 1)
                hd.eraser_counter = max(0, hd.eraser_counter - 3)
                hd.thumbs_counter = max(0, hd.thumbs_counter - 3)

        return hand_states

    # ── Riconoscimento dita ───────────────────────────────────

    def _count_fingers(self, landmarks, hand_label):
        """5 bool: [pollice, indice, medio, anulare, mignolo] estesi."""
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        fingers = [False] * 5

        if hand_label == "Right":
            fingers[0] = landmarks[tips[0]].x < landmarks[pips[0]].x
        else:
            fingers[0] = landmarks[tips[0]].x > landmarks[pips[0]].x

        for i in range(1, 5):
            fingers[i] = landmarks[tips[i]].y < landmarks[pips[i]].y

        return fingers

    def _finger_angle(self, landmarks, mcp, pip, tip):
        """Angolo di piegatura al PIP (in gradi). 180 = dritto, 90 = piegato."""
        ax, ay = landmarks[mcp].x - landmarks[pip].x, landmarks[mcp].y - landmarks[pip].y
        bx, by = landmarks[tip].x - landmarks[pip].x, landmarks[tip].y - landmarks[pip].y
        dot = ax * bx + ay * by
        mag_a = math.sqrt(ax * ax + ay * ay) + 1e-8
        mag_b = math.sqrt(bx * bx + by * by) + 1e-8
        cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
        return math.degrees(math.acos(cos_angle))

    def _is_peace_gesture(self, landmarks):
        """Peace/V basato sull'ANGOLO DI PIEGATURA delle dita.
        Dito dritto ~160-180, dito piegato ~60-100.
        Indipendente dall'orientamento della mano."""
        # Angoli al PIP: MCP -> PIP -> TIP
        index_angle  = self._finger_angle(landmarks, 5, 6, 8)
        middle_angle = self._finger_angle(landmarks, 9, 10, 12)
        ring_angle   = self._finger_angle(landmarks, 13, 14, 16)
        pinky_angle  = self._finger_angle(landmarks, 17, 18, 20)

        # Indice e medio devono essere DRITTI (angolo > 130)
        index_straight  = index_angle > 130
        middle_straight = middle_angle > 130

        # Anulare e mignolo devono essere PIEGATI (angolo < 135)
        ring_bent  = ring_angle < 135
        pinky_bent = pinky_angle < 135

        return index_straight and middle_straight and ring_bent and pinky_bent

    def _is_thumbs_down(self, landmarks, fingers_up):
        """Pollice giu: pollice punta in basso, tutte le altre dita piegate."""
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]

        # Pollice deve puntare verso il basso
        thumb_pointing_down = thumb_tip.y > thumb_mcp.y + 0.03
        thumb_below_wrist = thumb_tip.y > wrist.y

        # Altre dita piegate (angoli, come per peace)
        index_angle  = self._finger_angle(landmarks, 5, 6, 8)
        middle_angle = self._finger_angle(landmarks, 9, 10, 12)
        ring_angle   = self._finger_angle(landmarks, 13, 14, 16)
        pinky_angle  = self._finger_angle(landmarks, 17, 18, 20)
        others_bent = (index_angle < 140 and middle_angle < 140 and
                       ring_angle < 140 and pinky_angle < 140)

        return thumb_pointing_down and thumb_below_wrist and others_bent

    # ── Overlay ───────────────────────────────────────────────

    def draw_overlay(self, frame, hand_state):
        self._draw_skeleton(frame, hand_state)
        self._draw_crosshair(frame, hand_state)

    def _draw_skeleton(self, frame, hand_state):
        if hand_state.landmarks is None:
            return
        h, w = frame.shape[:2]
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand_state.landmarks]

        if hand_state.drawing:
            line_col, joint_col, tip_col = (0, 200, 0), (0, 255, 0), (100, 255, 100)
        elif hand_state.precision_erasing:
            line_col, joint_col, tip_col = (60, 60, 200), (80, 80, 255), (120, 120, 255)
        else:
            line_col, joint_col, tip_col = (140, 140, 140), (180, 180, 180), (210, 210, 210)

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], line_col, 1, cv2.LINE_AA)

        for i, pt in enumerate(pts):
            r = 3 if i in FINGERTIP_IDS else 2
            c = tip_col if i in FINGERTIP_IDS else joint_col
            cv2.circle(frame, pt, r, c, -1, cv2.LINE_AA)

        if hand_state.gesture_label:
            wx, wy = pts[0]
            cv2.putText(frame, hand_state.gesture_label,
                        (wx - 15, wy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        joint_col, 1, cv2.LINE_AA)

    def _draw_crosshair(self, frame, hand_state):
        h, w = frame.shape[:2]
        cx = int(hand_state.raw_x * w)
        cy = int(hand_state.raw_y * h)

        if hand_state.drawing:
            col = (0, 255, 0)
        elif hand_state.precision_erasing:
            col = (0, 0, 255)
        else:
            col = (180, 180, 180)

        s = 5
        cv2.line(frame, (cx - s, cy), (cx + s, cy), col, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - s), (cx, cy + s), col, 1, cv2.LINE_AA)
