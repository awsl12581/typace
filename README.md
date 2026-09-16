# typace

用 Textual 编写界面，用 typace 选择运行在真实终端还是 SDL3 + OpenGL 3.3 窗口。

## 启动应用

`app/` 只负责启动；应用组合位于 `typace.application`，通用星体能力位于
`typace.celestial`，Sol 只是 `typace.solar_system` 提供的一份目录数据。同一份
Textual 应用可以在终端或 SDL3 窗口中运行：

```bash
conda run --no-capture-output -n typace python -m app --backend terminal
conda run --no-capture-output -n typace python -m app --backend sdl
```

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
#         font="assets/fonts/seguisym.ttf",
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
├─ app/                         单纯的可执行启动器
│  ├─ __init__.py               入口包
│  └─ __main__.py               terminal / SDL 启动选择
├─ README.md                    使用方法和目录说明
├─ requirements.txt             运行依赖
├─ AGENTS.md                    开发约束
├─ assets/fonts/
│  └─ seguisym.ttf              Braille 点阵字形
├─ samples/
│  └─ basic.py                  同一界面的两端启动示例
├─ tests/
│  ├─ test_app.py               星体、轨道、交互和应用图形测试
│  └─ test_ui.py                wrapper 接口与图形集成测试
└─ typace/                      Python 包
   ├─ application.py            应用组件组合
   ├─ config.py                 共享字体配置
   ├─ keybindings.py            应用级全局按键
   ├─ celestial/                通用星体基础层
   │  ├─ bindings.py            星体视图按键
   │  ├─ model.py               类型化星体目录及通用加载器
   │  ├─ orbital.py             开普勒轨道计算与统一时刻快照
   │  ├─ rendering.py           Braille 星体系统渲染
   │  └─ view.py                通用 Textual 星体视图
   ├─ solar_system/             Sol 数据模块
   │  ├─ __init__.py            Sol 目录加载入口
   │  └─ data/sol.json          星体、纹理和行星环数据
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
```

示例包含输入框、按钮、状态文本和退出快捷键 `Ctrl+Q`。SDL 示例默认使用
`assets/fonts/seguisym.ttf`，也可通过 `--font` 指定其他字体；
完整的字体回退配置见 `WindowOptions`。

正式应用使用原生 Textual `Widget` 和 Braille 点阵字符绘制太阳、八大行星、主要卫星、
轨道、星体纹理、光照和土星环。“模拟控制”和“视图控制”是覆盖在星图上的独立浮动
窗口，不占用星图布局空间；`[` / `]` 在当前显示的面板间移动选中状态，洋红色边框表示
当前面板。窗口较窄时面板会像窗口一样重叠，选中的面板始终显示在最上层。

默认聚焦地球；`Tab` / `Shift+Tab` 切换星体，`G` 返回全局视图，`V` 切换视角，
`+` / `-` 缩放，方向键平移，`,` / `.` 调整时间倍率，空格暂停或继续。鼠标滚轮可以
缩放，点击星体可以聚焦。按 `Esc` 打开设置，可以切换简体中文或英文、设置相对 J2000
天数、时间倍率、暂停状态以及每个浮动面板的开闭；再次按 `Esc` 放弃修改并返回主界面。

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
python -m black --check app typace samples tests
pyright app typace samples tests
```

图形集成测试使用仓库内置字体，并需要可用的桌面显示和 OpenGL 驱动。测试创建隐藏的
真实 SDL 窗口，验证太阳系画面、文本输入、按钮点击、缩放和 GPU 帧缓冲。
Pylance 选择 Conda `typace` 解释器，具体要求见 `AGENTS.md`。

太阳系数据与渲染设计的第三方来源和许可证见 [`NOTICE.md`](NOTICE.md)。
