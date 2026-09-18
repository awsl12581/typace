import os
import threading
import unittest

from typace.tasks import ThreadTaskManager


class ThreadTaskManagerTests(unittest.TestCase):
    def test_compute_map_preserves_order_and_uses_managed_workers(self) -> None:
        manager = ThreadTaskManager(compute_workers=2)
        try:
            results = manager.map_compute(
                lambda value: (value, threading.current_thread().name), range(4)
            )
            self.assertEqual([item[0] for item in results], [0, 1, 2, 3])
            self.assertTrue(
                all(item[1].startswith("typace-compute") for item in results)
            )
        finally:
            manager.shutdown()

    def test_serial_queue_runs_in_submission_order(self) -> None:
        manager = ThreadTaskManager(compute_workers=1)
        try:
            events: list[int] = []
            first = manager.submit_serial(lambda: events.append(1))
            second = manager.submit_serial(lambda: events.append(2))
            first.result()
            second.result()
            self.assertEqual(events, [1, 2])
        finally:
            manager.shutdown()

    def test_planning_uses_dedicated_process(self) -> None:
        manager = ThreadTaskManager(compute_workers=1)
        try:
            future = manager.submit_planning(os.getpid)
            self.assertNotEqual(future.result(), os.getpid())
        finally:
            manager.shutdown()


if __name__ == "__main__":
    unittest.main()
