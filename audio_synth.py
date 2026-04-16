"""
audio_synth.py - Sintesi audio per feedback sonoro durante il disegno
Genera un tono la cui frequenza dipende dalla posizione sulla griglia.
"""

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

# Range frequenze (Hz) - mappa posizione griglia a nota
FREQ_MIN = 220.0   # A3
FREQ_MAX = 880.0   # A5
SAMPLE_RATE = 44100
DURATION = 0.15


class AudioSynth:
    def __init__(self):
        self._enabled = False
        self._last_freq = 0
        self._playing = False
        if HAS_PYGAME:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 512)
                    pygame.mixer.init()
                self._channel = pygame.mixer.Channel(15)
                self._enabled = True
            except Exception:
                pass

    def play_note(self, gx, gy, width, height, active):
        """Suona una nota basata sulla posizione. active=False per fermare."""
        if not self._enabled:
            return

        if not active:
            if self._playing:
                self._channel.stop()
                self._playing = False
            return

        # Mappa x sulla frequenza
        t = gx / max(width - 1, 1)
        freq = FREQ_MIN + t * (FREQ_MAX - FREQ_MIN)
        freq_int = int(freq)

        # Evita di rigenerare se stessa frequenza
        if freq_int == self._last_freq and self._playing:
            return

        try:
            samples = int(SAMPLE_RATE * DURATION)
            t_arr = np.linspace(0, DURATION, samples, endpoint=False)
            wave = np.sin(2 * np.pi * freq * t_arr)
            # Fade in/out
            fade = min(64, samples // 4)
            wave[:fade] *= np.linspace(0, 1, fade)
            wave[-fade:] *= np.linspace(1, 0, fade)
            wave = (wave * 16000).astype(np.int16)
            sound = pygame.sndarray.make_sound(wave)
            sound.set_volume(0.3)
            self._channel.play(sound)
            self._last_freq = freq_int
            self._playing = True
        except Exception:
            pass
