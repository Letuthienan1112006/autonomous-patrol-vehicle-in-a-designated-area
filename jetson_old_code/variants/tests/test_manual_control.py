"""Offline checks on keyboard driving; no terminal, socket or model."""
import unittest

from manual_control import ManualDriver


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class DrivingTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.driver = ManualDriver(speed_step=4, steer_step=4, max_speed=16,
                                   max_steer=18, idle_stop_sec=1.2,
                                   steer_return_sec=0.5, clock=self.clock)

    def test_nothing_pressed_yet_commands_a_stop(self):
        self.assertEqual(self.driver.command(), (0, 0))

    def test_w_accelerates_in_steps_up_to_the_cap(self):
        for expected in (4, 8, 12, 16, 16):
            self.driver.feed("w")
            self.assertEqual(self.driver.command()[1], expected)

    def test_a_and_d_steer_both_ways_within_the_cap(self):
        for _ in range(10):
            self.driver.feed("d")
        self.assertEqual(self.driver.command()[0], 18)
        for _ in range(10):
            self.driver.feed("a")
        self.assertEqual(self.driver.command()[0], -18)

    def test_s_brakes_and_never_reverses(self):
        # The firmware clamps speed to >= 0 (three separate places), so a
        # negative command would silently become 0 anyway - brake is the
        # honest behaviour, not a fake reverse.
        self.driver.feed("w")
        self.driver.feed("w")
        self.driver.feed("s")
        steer, speed = self.driver.command()
        self.assertEqual(speed, 0)
        self.assertGreaterEqual(speed, 0)
        for _ in range(5):
            self.driver.feed("s")
            self.assertGreaterEqual(self.driver.command()[1], 0)

    def test_space_is_an_immediate_full_stop(self):
        self.driver.feed("w")
        self.driver.feed("d")
        self.driver.feed(" ")
        self.assertEqual(self.driver.command(), (0, 0))

    def test_dead_man_stops_the_car_when_keys_stop_arriving(self):
        # Holding a key streams auto-repeat characters; releasing it stops
        # the stream, and the car must not keep the last command forever.
        self.driver.feed("w")
        self.driver.feed("w")
        self.assertGreater(self.driver.command()[1], 0)
        self.clock.now += 1.19
        self.assertGreater(self.driver.command()[1], 0)
        self.clock.now += 0.02
        self.assertEqual(self.driver.command(), (0, 0))

    def test_steering_self_centres_while_still_driving(self):
        self.driver.feed("w")
        self.driver.feed("d")
        self.assertEqual(self.driver.command(), (4, 4))
        # Steer key released, but w is still being held: keep the speed,
        # straighten the wheel.
        self.clock.now += 0.6
        self.driver.feed("w")
        steer, speed = self.driver.command()
        self.assertEqual(steer, 0)
        self.assertGreater(speed, 0)

    def test_quit_keys_set_the_flag_and_zero_the_command(self):
        for key in ("q", "\x1b", "\x03"):
            driver = ManualDriver(clock=self.clock)
            driver.feed("w")
            driver.feed(key)
            self.assertTrue(driver.quit)
            self.assertEqual(driver.command(), (0, 0))

    def test_unknown_key_refreshes_the_dead_man_without_driving(self):
        self.driver.feed("w")
        before = self.driver.command()
        self.clock.now += 1.0
        self.driver.feed("k")
        self.clock.now += 0.3   # 1.3s since 'w', but only 0.3s since 'k'
        self.assertEqual(self.driver.command(), before)

    def test_timeouts_must_be_positive(self):
        with self.assertRaises(ValueError):
            ManualDriver(idle_stop_sec=0)
        with self.assertRaises(ValueError):
            ManualDriver(steer_return_sec=-1)


class TerminalRestoreTests(unittest.TestCase):
    def test_non_tty_stdin_is_handled_instead_of_crashing(self):
        # run control under nohup or a pipe has no tty; the loop should
        # still start (and simply receive no keys) rather than throw.
        import io
        from manual_control import RawKeyboard

        with RawKeyboard(stream=io.StringIO("wasd")) as kb:
            self.assertEqual(kb.pending_keys(), [])


class PatrolWiringTests(unittest.TestCase):
    """The override's POSITION in the loop is the whole correctness
    argument: after every automatic override so the keys have the last
    word, before telemetry so the logged speed is what was really sent.
    Moving it either way breaks silently, so pin it by parsing the file -
    no camera, socket or model import needed."""

    def setUp(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1]
                  / "patrol_robot.py").read_text().split("\n")
        self.lines = {}
        for i, line in enumerate(source):
            for name, needle in (
                ("auto_gate", "lane_guard.command(steer_angle, car_speed)"),
                ("manual", "manual_driver.command()"),
                ("telemetry", "# ── E2. TELEMETRY"),
            ):
                if needle in line and name not in self.lines:
                    self.lines[name] = i

    def test_all_three_anchors_exist(self):
        self.assertEqual(sorted(self.lines), ["auto_gate", "manual", "telemetry"])

    def test_manual_override_runs_after_the_automatic_gate(self):
        self.assertGreater(self.lines["manual"], self.lines["auto_gate"])

    def test_manual_override_runs_before_telemetry_is_written(self):
        self.assertLess(self.lines["manual"], self.lines["telemetry"])

    def test_manual_path_is_gated_and_does_not_touch_the_auto_path(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1]
                  / "patrol_robot.py").read_text()
        # Every use of the driver must sit behind the None check, so a run
        # without AGV_MANUAL_CONTROL cannot reach the keyboard at all.
        self.assertIn("if manual_driver is not None:", source)
        self.assertIn('os.environ.get("AGV_MANUAL_CONTROL", "")', source)


if __name__ == "__main__":
    unittest.main()
