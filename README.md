# typace

用 Textual 编写界面，用 typace 选择运行在真实终端还是 SDL3 + OpenGL 3.3 窗口。

## 使用方式

先写一个普通 Textual 应用，组件、CSS、布局和事件处理都遵循 Textual：

```python
from textual.app import App, ComposeResult
from textual.widgets import Input

from typace.ui import WindowOptions, run


class MyApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="请输入")


# 选择一个启动方式：
run(MyApp())  # 当前终端
run(MyApp(), terminal="new")  # 新终端

# run(
#     MyApp(),
#     backend="sdl",
#     window=WindowOptions(
#         font="assets/fonts/monospace.ttf",
#         fallback_fonts=("assets/fonts/seguisym.ttf",),
#     ),
# )
```

公共接口只有两个：

| 接口 | 职责 |
| --- | --- |
| `run(app, *, backend="terminal", terminal="current", window=None)` | 在当前终端、新终端或 SDL 窗口启动应用 |
| `WindowOptions(font=..., ...)` | 指定 SDL 窗口的字体、标题、尺寸 |

`backend="terminal"` 使用应用原有的 Textual 驱动；`terminal="current"` 在当前终端
运行，`terminal="new"` 在 Windows 新终端中重新执行当前脚本或模块。新终端模式启动后
当前调用返回 `None`，且不接受窗口配置。
`backend="sdl"` 必须提供 `WindowOptions`，启动时装入 SDL 驱动，结束或异常时恢复原驱动。
当前终端和 SDL 模式会阻塞运行；SDL 模式应从主线程启动。

`WindowOptions` 的可选字段：`fallback_fonts=()`、`title="typace"`、
`width=1000`、`height=600`、`font_size=18`。窗口宽高采用 SDL 窗口坐标，
字号采用帧缓冲像素；高 DPI 下两者可能不同。

## 项目结构

```text
typace/                         项目根目录
├─ README.md                    使用方法和目录说明
├─ requirements.txt             运行依赖
├─ AGENTS.md                    开发约束
├─ assets/fonts/
│  └─ seguisym.ttf              Braille 点阵字形
├─ samples/
│  └─ basic.py                  同一界面的两端启动示例
├─ tests/
│  └─ test_ui.py                接口、组件和图形集成测试
└─ typace/                      Python 包
   └─ ui/
      ├─ __init__.py            只导出公共 API
      ├─ runner.py              选择后端，启动应用
      ├─ config.py              WindowOptions 数据配置
      └─ backends/sdl/
         ├─ driver.py           窗口生命周期、输出与输入适配
         └─ renderer.py         字体栅格化、OpenGL 绘制
```

界面属于应用，启动选择属于 runner，窗口和绘制属于后端。库代码不依赖 `samples`。
终端端直接使用 Textual 原生驱动，因此不另建一层空的终端实现。

## 运行示例

所有命令均从仓库根目录、在已有 Conda `typace` 环境中执行：

```powershell
conda activate typace
python -m samples.basic --backend terminal
python -m samples.basic --backend terminal --new-terminal
python -m samples.basic --backend sdl
python -m samples.planet --backend sdl
```

示例包含输入框、按钮、状态文本和退出快捷键 `Ctrl+Q`。Windows 默认使用
Consolas，并从 `assets/fonts/` 加载点阵回退字体；其他平台通过
`--font assets/fonts/monospace.ttf` 指定字体，
完整的字体回退配置见 `WindowOptions`。

`samples.planet` 使用原生 Textual `Widget` 和 Braille 点阵字符绘制带光照的地球，按空格键
暂停或继续自转；同一示例也可将 backend 改为 `terminal`。

根目录的 `requirements.txt` 列出运行依赖。先检查已有环境，仅在缺少依赖时执行
`python -m pip install -r requirements.txt`。PySDL3 所需动态库应在环境准备时安装；
其首次自动下载需要网络。

## 后端分工与范围

终端模式：Textual 直接负责终端输出和输入，不加载图形依赖。

SDL 模式：Textual 输出 → pyte 解析 ANSI 和维护屏幕 → FreeType 栅格化字体 →
ModernGL 使用 GLSL 330 显示。SDL 的键盘、文本、鼠标和尺寸事件转换为 Textual 事件。

当前支持中英文及字体回退、颜色、粗体、斜体、下划线、删除线、反色、键盘快捷键、
鼠标点击/移动/竖向滚轮和窗口缩放。暂不提供系统剪贴板桥接、IME 预编辑显示、
彩色 emoji 或复杂文字塑形；一次运行一个 SDL 应用。

变化的画面先在 CPU 合成 RGB 纹理，再由 OpenGL 绘制。高频大规模动画应先测量
这一步的性能。桥接使用 Textual Driver API 和少量内部适配接口，升级 Textual
主版本时应运行图形集成测试。

## 检查

```powershell
python -m unittest discover -s tests -v
$env:TYPACE_TEST_SDL = "1"
python -m unittest discover -s tests -v
python -m black --check typace samples tests
```

图形集成测试需要 Windows 示例字体和可用的 OpenGL 驱动。测试创建隐藏的真实
SDL 窗口，验证文本输入、按钮点击、缩放和 GPU 帧缓冲。
Pylance 选择 Conda `typace` 解释器，具体要求见 `AGENTS.md`。
