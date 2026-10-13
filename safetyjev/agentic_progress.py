"""Bounded observation-only stagnation proxy; no simulator goal or safety labels."""
from collections import deque
import numpy as np
from PIL import Image


class ProgressWindow:
    def __init__(self, config):
        self.config = config
        self.rows = deque(maxlen=config['window_chunks'] + 1)
        self.report = {'ready': False, 'stalled': False}

    def observe(self, step, state, image):
        if self.rows and step == self.rows[-1][0]:
            return dict(self.report)
        if self.rows and step != self.rows[-1][0] + 8:
            raise ValueError('Progress observations must follow executed eight-action chunks')
        state = np.asarray(state, dtype=np.float32)
        pixels = np.asarray(image)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError('Invalid progress robot state')
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError('Progress camera must be RGB uint8')
        thumbnail = np.asarray(Image.fromarray(pixels).resize((32, 32), Image.Resampling.BOX), dtype=np.float32) / 255
        self.rows.append((step, state.copy(), thumbnail))
        self.report = {'ready': len(self.rows) == self.rows.maxlen, 'stalled': False,
                       'start_step': self.rows[0][0], 'end_step': step}
        if self.report['ready']:
            _, previous, previous_image = self.rows[0]
            metrics = {'joint_rms_rad': float(np.sqrt(np.mean((state[:7] - previous[:7]) ** 2))),
                       'gripper_delta_m': float(abs(state[14] - previous[14])),
                       'image_mae': float(np.abs(thumbnail - previous_image).mean())}
            self.report.update(metrics, stalled=all(value <= self.config[key] for key, value in metrics.items()))
        return dict(self.report)

    def reset(self):
        # Keep the current executed boundary; rejected proposals never enter this window.
        if self.rows:
            last = self.rows[-1]
            self.rows.clear()
            self.rows.append(last)
        self.report = {'ready': False, 'stalled': False}
