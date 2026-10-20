import time
class PID:
    def __init__(self, kp, ki, kd, out_min=-25, out_max=25):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def reset(self):
        self.integral = 0
        self.prev_error = 0
        self.prev_time = None

    def update(self, error):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            return 0
        dt = now - self.prev_time
        if dt <= 0: dt = 1e-3
        self.prev_time = now

        # PID tính toán
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(self.out_min, min(self.out_max, output))
        return output