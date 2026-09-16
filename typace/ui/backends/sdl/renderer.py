"""FreeType rasterization of a pyte screen, presented by ModernGL."""

from functools import lru_cache

import freetype
from freetype.ft_enums.ft_load_flags import FT_LOAD_FLAGS
from freetype.ft_enums.ft_render_modes import FT_RENDER_MODES
import moderngl
import numpy as np
from numpy.typing import NDArray
from pyte.screens import Screen
from rich.color import Color

from ...config import WindowOptions


@lru_cache(maxsize=512)
def color(value: str, foreground: bool) -> tuple[int, int, int]:
    if value == "default":
        return (220, 220, 220) if foreground else (20, 22, 26)
    if len(value) == 6 and all(c in "0123456789abcdef" for c in value.lower()):
        value = "#" + value
    value = value.replace("bfight", "bright").replace("brown", "yellow")
    value = value.replace("bright", "bright_")
    rgb = Color.parse(value).get_truecolor()
    return rgb.red, rgb.green, rgb.blue


class ScreenRenderer:
    def __init__(self, options: WindowOptions) -> None:
        self.faces = [
            freetype.Face(path) for path in (options.font, *options.fallback_fonts)
        ]
        for face in self.faces:
            face.set_pixel_sizes(0, options.font_size)
        face = self.faces[0]
        face.load_char("M")
        self.cell_width = max(1, face.glyph.advance.x // 64)
        self.cell_height = max(1, face.size.height // 64)
        self.baseline = face.size.ascender // 64
        self.glyphs: dict[
            tuple[str, bool, bool], tuple[NDArray[np.uint8], int, int]
        ] = {}
        self.ctx = moderngl.create_context(require=330)
        self.program = self.ctx.program(
            vertex_shader="""#version 330 core
            out vec2 uv;
            void main() {
                vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                uv = vec2(p.x, 1.0 - p.y);
                gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
            }""",
            fragment_shader="""#version 330 core
            uniform sampler2D screen;
            in vec2 uv;
            out vec4 result;
            void main() { result = texture(screen, uv); }
            """,
        )
        self.vao = self.ctx.vertex_array(self.program, [])
        self.texture: moderngl.Texture | None = None

    def glyph(
        self, char: str, bold: bool, italic: bool
    ) -> tuple[NDArray[np.uint8], int, int]:
        key = (char, bold, italic)
        if key not in self.glyphs:
            face = next(
                (face for face in self.faces if face.get_char_index(ord(char))),
                self.faces[0],
            )
            face.set_transform(
                freetype.Matrix(65536, 13107 if italic else 0, 0, 65536),
                freetype.Vector(0, 0),
            )
            face.load_char(char, FT_LOAD_FLAGS["FT_LOAD_DEFAULT"])
            if bold:
                freetype.FT_GlyphSlot_Embolden(face.glyph._FT_GlyphSlot)
            face.glyph.render(FT_RENDER_MODES["FT_RENDER_MODE_NORMAL"])
            bitmap = face.glyph.bitmap
            data = np.array(bitmap.buffer, dtype=np.uint8)
            if bitmap.rows:
                data = data.reshape(bitmap.rows, abs(bitmap.pitch))[:, : bitmap.width]
                if bitmap.pitch < 0:
                    data = data[::-1]
            else:
                data = data.reshape(0, 0)
            self.glyphs[key] = (data, face.glyph.bitmap_left, face.glyph.bitmap_top)
        return self.glyphs[key]

    def draw(self, screen: Screen, width: int, height: int) -> None:
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[:] = color("default", False)
        cw, ch = self.cell_width, self.cell_height
        # Backgrounds first: continuation cells must not cover a wide glyph.
        for y in range(screen.lines):
            for x in range(screen.columns):
                cell = screen.buffer[y][x]
                bg = color(cell.fg, True) if cell.reverse else color(cell.bg, False)
                pixels[y * ch : (y + 1) * ch, x * cw : (x + 1) * cw] = bg
        for y in range(screen.lines):
            for x in range(screen.columns):
                cell = screen.buffer[y][x]
                fg = color(cell.bg, False) if cell.reverse else color(cell.fg, True)
                for char in cell.data:
                    if char == " ":
                        continue
                    mask, left, top = self.glyph(char, cell.bold, cell.italics)
                    gx, gy = x * cw + left, y * ch + self.baseline - top
                    x0, y0 = max(0, gx), max(0, gy)
                    x1, y1 = min(width, gx + mask.shape[1]), min(
                        height, gy + mask.shape[0]
                    )
                    if x1 > x0 and y1 > y0:
                        alpha = mask[y0 - gy : y1 - gy, x0 - gx : x1 - gx, None] / 255.0
                        region = pixels[y0:y1, x0:x1]
                        region[:] = region * (1 - alpha) + np.array(fg) * alpha
                for enabled, offset in (
                    (cell.underscore, ch - 2),
                    (cell.strikethrough, ch // 2),
                ):
                    if enabled:
                        pixels[
                            y * ch + offset : y * ch + offset + 1, x * cw : (x + 1) * cw
                        ] = fg
        if not screen.cursor.hidden:
            x, y = screen.cursor.x * cw, screen.cursor.y * ch
            pixels[y + ch - 2 : y + ch, x : x + cw] = color("default", True)
        if self.texture is None or self.texture.size != (width, height):
            if self.texture is not None:
                self.texture.release()
            self.texture = self.ctx.texture((width, height), 3, alignment=1)
            self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.write(pixels.tobytes(), alignment=1)
        self.texture.use()
        self.ctx.viewport = (0, 0, width, height)
        self.vao.render(mode=moderngl.TRIANGLES, vertices=3)

    def close(self) -> None:
        if self.texture is not None:
            self.texture.release()
        self.vao.release()
        self.program.release()
        self.ctx.release()
