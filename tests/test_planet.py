import unittest

from samples.planet import Earth, PlanetDemo, earth_pixel


class PlanetTests(unittest.IsolatedAsyncioTestCase):
    def test_projection_clips_outside_the_sphere(self) -> None:
        self.assertIsNone(earth_pixel(1.1, 0.0, 0.0))
        self.assertIsNotNone(earth_pixel(0.0, 0.0, 0.0))

    async def test_rotation_and_pause(self) -> None:
        app = PlanetDemo()
        async with app.run_test(size=(80, 36)) as pilot:
            earth = app.query_one(Earth)
            output = earth.render().plain
            self.assertTrue(any("\u2801" <= char <= "\u28ff" for char in output))
            self.assertNotIn("▀", output)
            start = earth.rotation
            earth.advance()
            self.assertNotEqual(earth.rotation, start)
            await pilot.press("space")
            paused = earth.rotation
            earth.advance()
            self.assertEqual(earth.rotation, paused)


if __name__ == "__main__":
    unittest.main()
