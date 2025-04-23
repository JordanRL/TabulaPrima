import datetime
import re
import time
from dataclasses import dataclass
from time import sleep
from typing import Dict, Any, List, Literal, Union
from zoneinfo import ZoneInfo

import pyperclip
from pyfiglet import Figlet
from readchar import readkey, key
from rich.align import Align
from rich.console import Console, Group, RenderableType, RichCast, ConsoleRenderable
from rich.box import Box, SQUARE, ROUNDED
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, \
    TimeRemainingColumn, Task
from rich.style import Style
from rich.table import Column
from rich.text import Text

from config_schema import ConsoleConfig, TimeFormat

EMPTY_BOX = Box(
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
)

BOTTOM_BORDER = Box(
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "━━━━\n"
)

TOP_BORDER = Box(
    "━━━━\n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
)

TOP_BOTTOM_BORDER = Box(
    "━━━━\n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "━━━━\n"
)

BOTTOM_PADDED_BORDER = Box(
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    " ━━ \n"
)

TOP_PADDED_BORDER = Box(
    " ━━ \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
)

TOP_BOTTOM_PADDED_BORDER = Box(
    " ━━ \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    " ━━ \n"
)

@dataclass
class StyledText:
    plain_text: str
    style: Style

    def __str__(self) -> str:
        return f"[{self.style}]{self.plain_text}[/{self.style}]"

    def splitlines(self) -> list["StyledText"]:
        lines = self.plain_text.splitlines()
        new_lines = []
        for line in lines:
            new_lines.append(StyledText(line, self.style))
        return new_lines

    def __len__(self) -> int:
        return TPConsole().measure(self.plain_text).maximum

    def substr(self, substr_len) -> "StyledText":
        if substr_len > len(self):
            return StyledText(self.plain_text[:substr_len], self.style)
        else:
            return self

    def __add__(self, other) -> "StyledText":
        if isinstance(other, Style):
            return StyledText(self.plain_text, self.style + other)
        elif isinstance(other, StyledText):
            return StyledText(self.plain_text + other.plain_text, self.style + other.style)
        else:
            raise ValueError("The other object must be a Style or a StyledText object.")


# Console colors for better readability
class Colors:
    HEADER = '#B87FD9'
    BLUE = '#61AFEF'
    CYAN = '#56B6C2'
    GREEN = '#98C379'
    YELLOW = '#E5C07B'
    RED = '#E06C75'
    ORANGE = '#D19A66'
    PURPLE = '#BE50AE'
    PURPLE2 = '#C678DD'
    PROGRESS_FILL = '#4B6BFF'
    MED_GREY = '#8A8F98'

    @staticmethod
    def header(text):
        return Colors.apply_style(text, Colors.HEADER)

    @staticmethod
    def info(text):
        return Colors.apply_style(text, Colors.BLUE)

    @staticmethod
    def yellow(text):
        return Colors.apply_style(text, Colors.YELLOW)

    @staticmethod
    def purple(text):
        return Colors.apply_style(text, Colors.PURPLE)

    @staticmethod
    def purple2(text):
        return Colors.apply_style(text, Colors.PURPLE2)

    @staticmethod
    def progress_fill(text):
        return Colors.apply_style(text, Colors.PROGRESS_FILL)

    @staticmethod
    def medium_grey(text):
        return Colors.apply_style(text, Colors.MED_GREY)

    @staticmethod
    def orange(text):
        return Colors.apply_style(text, Colors.ORANGE)

    @staticmethod
    def success(text):
        return Colors.apply_style(text, Colors.GREEN)

    @staticmethod
    def warning(text):
        return Colors.apply_style(text, Colors.YELLOW)

    @staticmethod
    def error(text):
        return Colors.apply_style(text, Colors.RED)

    @staticmethod
    def highlight(text):
        return Colors.apply_style(text, Colors.CYAN)

    @staticmethod
    def apply_style(text, style):
        """
        if isinstance(style, str):
            parsed_style = Style.parse(style)
        elif isinstance(style, Style):
            parsed_style = style
        else:
            raise ValueError("The style must be a string or a Style object.")
        if isinstance(text, str):
            return StyledText(text, parsed_style)
        elif isinstance(text, StyledText):
            return text + parsed_style
        else:
            raise ValueError("The text must be a string or a StyledText object.")
        """
        return f"[{style}]{text}[/{style}]"

    @staticmethod
    def apply_gradient_to_lines(text, color_top, color_bottom, padding_top=False, padding_bottom=False):
        from colour import Color
        # 1. Split the string into a list of lines
        lines = text.splitlines()

        # 1a. Filter out empty or whitespace-only lines BEFORE counting
        #     We keep the original line content, but filter based on the stripped version
        non_empty_lines = [line for line in lines if line.strip()]

        # 2. Count the number of NON-EMPTY lines for the gradient
        num_lines = len(non_empty_lines)

        # Handle edge case: no non-empty lines found after filtering
        if num_lines == 0:
            # Requirement is to add empty line top/bottom, even if no content
            return "\n\n"

        # Handle edge case: only one non-empty line
        if num_lines == 1:
            start_color_hex_only = color_top.lstrip('#').upper()
            formatted_line = f"[#{start_color_hex_only}]{non_empty_lines[0]}[/#{start_color_hex_only}]"
            # Add empty line padding top and bottom
            return f"\n{formatted_line}\n"

        # --- Proceed if 2 or more non-empty lines ---

        # 3. Create the gradient based on the count of non-empty lines
        start_color = Color(color_top)
        end_color = Color(color_bottom)
        gradient_objects = list(start_color.range_to(end_color, num_lines))
        gradient_hex = [color.hex_l for color in gradient_objects]

        # 4. Apply the gradient to each NON-EMPTY line using BBCode style formatting
        formatted_lines = []
        for i, line in enumerate(non_empty_lines):
            line_hex_code = gradient_hex[i].lstrip('#').upper()
            formatted_line = f"[#{line_hex_code}]{line}[/#{line_hex_code}]"
            formatted_lines.append(formatted_line)

        # 5. Rejoin the formatted (non-empty) lines with newline characters
        final_result = "\n".join(formatted_lines)

        # 5a. Add one empty line padding at the top and bottom
        if padding_top:
            final_result = f" \n{final_result}"
        if padding_bottom:
            final_result = f"{final_result}\n "

        return final_result

    @staticmethod
    def apply_gradient_to_chars(text, color_start, color_end):
        from colour import Color
        num_chars = len(text)

        color_start = Color(color_start)
        color_end = Color(color_end)
        gradient_objects = list(color_start.range_to(color_end, num_chars))
        gradient_hex = [color.hex_l for color in gradient_objects]

        formatted_chars = []
        for i, char in enumerate(text):
            color_hex = gradient_hex[i].lstrip('#').upper()
            formatted_chars.append(f"[#{color_hex}]{char}[/#{color_hex}]")

        final_result = "".join(formatted_chars)

        return final_result

class ConditionalTimeRemainingColumn(TimeRemainingColumn):
    """Renders time remaining estimate, but only for tasks where it makes sense."""

    def render(self, task: "Task") -> Text:
        """Show time remaining."""
        is_application_task = task.fields.get('is_app_task', False)

        if is_application_task == "False" or is_application_task == "false":
            is_application_task = False

        if is_application_task and task.total is not None:
            # Display step count for the application task
            step_count = f"{int(task.completed)}/{int(task.total)} steps"
            return Text(step_count, style="progress.remaining")  # Use same style or a custom one
        else:
            if is_application_task:
                return Text("-:--:--", style="progress.remaining")
            else:
                return super().render(task)

class TPConsole:
    _instance = None
    _console: Console|None = None
    _cfg: ConsoleConfig|None = None
    _live: Live|None = None
    _progress_bar: Progress|None = None
    _layout: Layout|None = None
    _progress_tasks = {}

    # Layout content
    _title_content_layout: Layout | None = None
    _main_content_layout: Layout | None = None
    _progress_content_layout: Layout | None = None
    _stats = {}
    _stats_panel: Layout|None = None
    _progress_panel: Layout|None = None
    _alternate_screens = []
    _alternate_screen_items = []
    _main_content_cols: List[Layout] = []
    _main_content_cols_panels: List[Panel] = []

    # Default text content
    _app_title = "Tabula Prima"
    _app_subtitle = "Training"
    _app_stage = "Bootstrapping"
    _messages: List[Dict[str, Any]] = []
    _section_titles = []
    _content_items: List[Dict[str, Any]] = []
    _tz_info = None
    _main_content_panel = Panel(
        "",
        expand=True, border_style=Colors.HEADER, box=TOP_BOTTOM_BORDER, padding=0
    )
    _tabula_prima_fig = Text.from_markup(
        Colors.apply_gradient_to_lines(Figlet(font='contessa').renderText("TabulaPrima"), Colors.BLUE, Colors.PURPLE2)
    )

    def __new__(cls, cfg: ConsoleConfig|None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._cfg = cfg if cfg is not None else ConsoleConfig()
            match cls._cfg.color_system:
                case "standard":
                    cls._console = Console(soft_wrap=True, color_system="standard")
                case "256":
                    cls._console = Console(soft_wrap=True, color_system="256")
                case "truecolor":
                    cls._console = Console(soft_wrap=True, color_system="truecolor")
                case _:
                    cls._console = Console(soft_wrap=True, color_system="auto")
            cls._tz_info = ZoneInfo(cls._cfg.timezone)
            cls._title_content_layout = Layout(cls._tabula_prima_fig, name="title", size=3)
            cls._main_content_layout = Layout(cls._main_content_panel, name="main")
            cls._progress_content_layout = Layout(name="progress", size=6)
            cls._layout = Layout()
            cls._layout.split_column(
                cls._title_content_layout,
                cls._main_content_layout,
                cls._progress_content_layout,
            )
            cls._layout["progress"].visible = False
            if cls._cfg.use_live_display:
                cls._live = Live(
                    cls._layout,
                    console=cls._console,
                    screen=True,
                    refresh_per_second=10,
                    redirect_stderr=False,
                    redirect_stdout=False
                )
                cls._live.start()

        return cls._instance

    def progress_start(self):
        self._create_progress_bar()
        if not self._cfg.use_live_display:
            self._progress_bar.start()

    def progress_stop(self):
        self._progress_bar.stop()
        if self._cfg.use_live_display:
            self._layout["progress"].visible = False
            self._update_main_render()

    def start_live(self):
        if self._live is None:
            self._live = Live(
                self._layout,
                console=self._console,
                screen=True,
                refresh_per_second=10,
                redirect_stderr=False,
                redirect_stdout=False
            )
            self._live.start()
            self._cfg.use_live_display = True

    def end_live(self):
        if self._live is not None:
            self._live.stop()
            self._live = None
            self._cfg.use_live_display = False

    def handle_exception(self):
        self._console.print_exception()

    def print(self, content: str):
        self._print_single_message(content)

    def print_list_item(self, title: str, content: str, returns: bool = False):
        text = f"{Colors.purple('»')} {Colors.info(title)}: {Colors.highlight(content)}"
        if not returns:
            self._print_single_message(text)
            return None
        else:
            return text

    def print_list(self, items: List[Dict[str, str]], returns: bool = False):
        output = []
        for item in items:
            if not returns:
                self.print_list_item(item["title"], item["content"])
            else:
                output.append(self.print_list_item(item["title"], item["content"], returns))
        if returns:
            return output
        else:
            return None

    def print_notification(self, content: str):
        text = f"{Colors.purple('ⓘ')} {Colors.info(content)}"
        self._print_single_message(text)

    def print_warning(self, content: str):
        text = f"{Colors.purple('⚠')} {Colors.warning(content)}"
        self._print_single_message(text)

    def print_error(self, content: str):
        text = f"{Colors.orange('ⓧ')} {Colors.error(content)}"
        self._print_single_message(text)

    def print_complete(self, content: str):
        text = f"{Colors.success('✔')} {Colors.info(content)}"
        self._print_single_message(text)

    def print_success(self, content: str):
        text = f"{Colors.success(content)}"
        self._print_single_message(text)

    def rule(self, content, style=Colors.HEADER):
        formatted_content = f"[{Colors.ORANGE}]{content}[/{Colors.ORANGE}]"
        if self._cfg.use_live_display:
            self._content_items.append({"type": "rule", "content": f" \n{formatted_content}\n ", "height": 3, "style": style})
            self._update_main_render()
        else:
            self._console.rule(formatted_content, style=style)

    def subrule(self, content, style=Colors.HEADER):
        formatted_content = f"[{Colors.PROGRESS_FILL}]{content}[/{Colors.PROGRESS_FILL}]"
        if self._cfg.use_live_display:
            self._content_items.append({"type": "subrule", "content": f" \n{formatted_content}", "height": 2, "style": style})
            self._update_main_render()
        else:
            self._console.print(formatted_content, style=style)

    def section(
            self,
            content: str,
            font: str = "cybermedium",
            color_top = Colors.BLUE,
            color_bottom = Colors.HEADER,
            padding_top= False,
            padding_bottom=True
    ):
        section_text = Colors.apply_gradient_to_lines(
            text=Figlet(font=font, width=self._console.width-2-20).renderText(content),
            color_top=color_top,
            color_bottom=color_bottom,
            padding_top=padding_top,
            padding_bottom=padding_bottom
        )
        if self._cfg.use_live_display:
            self._content_items.append({"type": "section", "height": len(section_text.splitlines()), "content": section_text})
            self._update_main_render()
        else:
            self._console.print(Align.center(Text.from_markup(section_text)))

    def clear_main_content(self):
        if self._cfg.use_live_display:
            self._content_items = []
            self._update_main_render()
        else:
            self._console.clear()

    def new_main_content(self, content: str|RenderableType|None = None, with_progress: bool = True):
        if self._cfg.use_live_display:
            self._alternate_screens.append(self._main_content_panel.renderable)
            self._alternate_screen_items.append(self._content_items)
            self._layout["progress"].visible = with_progress
            if isinstance(content, str):
                for line in content.splitlines():
                    self._content_items.append({"type": "text", "content": line, "height": 1, "time": None})
            elif isinstance(content, RenderableType):
                self._content_items = []
                self._main_content_panel.renderable = content
            else:
                self._content_items = []
                self._main_content_panel.renderable = Text("")
            self._live.refresh()
            screen_idx = len(self._alternate_screens) - 1
            return screen_idx
        else:
            self._console.clear()
            return None

    def update_column_content(self, col_idx: int, content: str|RenderableType|None = None):
        if self._cfg.use_live_display:
            if isinstance(content, str):
                for line in content.splitlines():
                    self._main_content_cols_panels[col_idx].renderable = Text.from_markup(line)
            elif isinstance(content, RenderableType):
                self._main_content_cols_panels[col_idx].renderable = content

    def add_column_to_main(self, width: int = 25, content: RenderableType|None = None):
        if self._cfg.use_live_display:
            col_idx = len(self._main_content_cols)
            new_col_name = f"col_{col_idx}"
            layout = Layout(name=new_col_name, size=width)
            if content is None:
                content = Panel(
                    "",
                    expand=True, border_style=Colors.HEADER, box=SQUARE, padding=0
                )
            layout.update(content)
            if len(self._main_content_layout.children) == 0:
                sub_main_layout = Layout(self._main_content_panel, name="sub_main")
                self._main_content_layout.split_row(sub_main_layout, layout)
            else:
                self._main_content_layout.add_split(layout)
            self._main_content_cols.append(layout)
            self._main_content_cols_panels.append(content)
            self._main_content_panel.width = self._console.width - sum([layout.size for layout in self._main_content_cols])
            self._update_main_render()
            return col_idx
        else:
            return None

    def remove_columns_from_main(self):
        if self._cfg.use_live_display:
            if len(self._main_content_cols) > 0:
                self._main_content_cols = []
                self._main_content_layout.unsplit()
                self._main_content_layout.update(self._main_content_panel)
                self._update_main_render()

    def update_app_title(self, title: str):
        self._app_title = title

    def update_app_subtitle(self, subtitle: str):
        self._app_subtitle = subtitle

    def update_app_stage(self, stage: str):
        self._app_stage = stage

    def update_app_stats(self, stats: dict):
        if self._stats_panel is not None:
            self._stats.update(stats)
            stats_string = f"{Colors.header(self._stats_title)}\n"
            items_count = 0
            for stat_name, stat_value in self._stats.items():
                if items_count % 8 == 0 and items_count > 0:
                    stats_string += "\n"
                elif items_count > 0:
                    stats_string += " " * 2 + "|" + " " * 2
                stats_string += f"{Colors.info(stat_name)}: {Colors.highlight(stat_value)}"
                items_count += 1
            self._stats_panel.update(Text.from_markup(stats_string))

    def create_progress_task(self, task_name: str, task_desc: str, total: float|None = None, is_app_task: bool = False, **kwargs):
        if self._progress_bar is None:
            self.progress_start()
        task_id = self._progress_bar.add_task(task_desc, total=total, is_app_task=is_app_task, **kwargs)
        self._progress_tasks[task_name] = {"id": task_id, "total": total, "description": task_desc, "completed": 0}

    def update_progress_task(self, task_name: str, completed: float|None = None, **kwargs):
        task_config = self._get_progress_task(task_name)
        if task_config is None:
            return False
        updates = {"completed": task_config["completed"] + kwargs["advance"] if "advance" in kwargs else 0}
        updates["completed"] = completed if completed is not None else updates["completed"]
        updates.update(kwargs)
        self._progress_tasks[task_name].update(updates)
        self._progress_bar.update(task_config["id"], completed=completed, **kwargs)
        target_task = None
        for task in self._progress_bar.tasks:  # Iterate through the current tasks list
            if task.id == task_config["id"]:
                target_task = task
                break  # Found the task
        if target_task is not None and target_task.finished and len(self._progress_tasks) > 1:
            self.remove_progress_task(task_name)
        return True

    def remove_progress_task(self, task_name: str):
        task_config = self._get_progress_task(task_name)
        if task_config is None:
            return False
        if task_config["total"] is not None:
            self._progress_bar.update(task_config["id"], completed=task_config["total"])
        self._progress_bar.stop_task(task_config["id"])
        self._progress_bar.remove_task(task_config["id"])
        del self._progress_tasks[task_name]
        return True

    def get_progress_task_properties(self, task_name: str) -> dict|None:
        task_config = self._get_progress_task(task_name)
        if task_config is None:
            return None
        return task_config

    def has_progress_task(self, task_name: str):
        return task_name in self._progress_tasks

    def prompt(self, prompt_text: str, default_value: str|None = None, password: bool = False, choices: List[str]|None = None):
        _prompt_text = f"{Colors.orange('TabulaPrima')}{Colors.header(':')} {Colors.success(prompt_text.strip())}"
        _choices_text = f"{Colors.header('[')} "
        for choice in choices:
            if choice != choices[0]:
                _choices_text += ", "
            if choice == default_value:
                _choices_text += f"{Colors.header(choice)} {Colors.success('(default)')}"
            else:
                _choices_text += f"{Colors.info(choice)}"
        _choices_text += f" {Colors.header('] (')} {Colors.highlight('ESC for default, UP and DOWN to cycle choices')}{Colors.header(')')}" if default_value is not None else f" {Colors.header(']')}"

        _input_buffer = []
        _output_text = _prompt_text + "\n" + _choices_text + "\n" + Colors.header('> ')
        _selected_choice = None
        _choice_index = 0

        _stats_panel_content = self._stats_panel.renderable if self._stats_panel is not None else Text("")

        while True:
            # This blocks the main thread until a key is pressed
            k = readkey()

            # We only want to handle a few special keys
            if k == key.ENTER:
                _selected_choice = "".join(_input_buffer)
                break
            elif k == key.BACKSPACE or k == key.DELETE:
                if len(_input_buffer) > 0:
                    _input_buffer.pop()
            elif k == key.ESC:
                _selected_choice = default_value
                break
            elif k == key.UP or k == key.LEFT:
                if not choices or len(choices) == 0:
                    continue
                _input_buffer = list(choices[_choice_index])
                if _choice_index == 0:
                    _choice_index = len(choices) - 1
                else:
                    _choice_index -= 1
            elif k == key.DOWN or k == key.RIGHT:
                if not choices or len(choices) == 0:
                    continue
                _input_buffer = list(choices[_choice_index])
                if _choice_index == len(choices) - 1:
                    _choice_index = 0
                else:
                    _choice_index += 1
            elif k == key.CTRL_V:
                _input_buffer = list(pyperclip.paste())
            elif k.isprintable():
                _input_buffer.append(k)

            if not password:
                self._stats_panel.update(Text.from_markup(_output_text + Colors.success("".join(_input_buffer))))
            else:
                self._stats_panel.update(Text.from_markup(_output_text + Colors.success("*" * len(_input_buffer))))

            self._live.refresh()

        self._stats_panel.update(_stats_panel_content)
        self._live.refresh()

        if _selected_choice is None:
            _selected_choice = default_value
        return _selected_choice

    def confirm(self, prompt_text: str, default_value: bool = False):
        _prompt_text = f"{Colors.orange('TabulaPrima')}{Colors.header(':')} {Colors.success(prompt_text.strip())}"
        _yes_text = f"{Colors.success('Y')}" if default_value else f"{Colors.info('y')}"
        _no_text = f"{Colors.error('N')}" if default_value else f"{Colors.info('n')}"
        _default_text = Colors.header('['+_yes_text+'/'+_no_text+']')
        _output_text = _prompt_text + "\n" + _default_text + "\n" + Colors.header('> ')

        _stats_panel_content = self._stats_panel.renderable if self._stats_panel is not None else Text("")
        _selected_choice = default_value

        self._stats_panel.update(Text.from_markup(_output_text))
        self._live.refresh()

        while True:
            # This blocks the main thread until a key is pressed
            k = readkey()

            # We only want to handle a few special keys
            if k == key.ENTER or k == key.ESC:
                break
            elif k == 'y' or k == 'Y':
                _selected_choice = True
                break
            elif k == 'n' or k == 'N':
                _selected_choice = False
                break

        self._stats_panel.update(_stats_panel_content)
        self._live.refresh()

        return _selected_choice

    def measure(self, content: str):
        return self._console.measure(content)

    def _get_progress_task(self, task_name: str):
        if task_name in self._progress_tasks:
            return self._progress_tasks[task_name]
        else:
            return None

    def _print_single_message(self, text, with_time=True):
        if self._cfg.use_live_display:
            self._content_items.append({"type": "text", "content": text, "height": len(text.splitlines()) if text != "" else 1, "time": time.time() if with_time else None})
            self._update_main_render()
        else:
            self._console.print(text)

    def _update_main_render(self):
        if self._cfg.use_live_display:
            message_line_count = self._console.height
            if self._layout['progress'].visible:
                message_line_count -= 6 # Progress/Stats panel
            message_line_count -= 3  # Title bar
            message_line_count -= 2  # Top and bottom border
            if len(self._main_content_layout.children) > 0:
                if self._main_content_layout.get("sub_main").size is not None:
                    message_line_width = self._main_content_layout.get("sub_main").size
                else:
                    message_line_width = self._console.width - sum([layout.size for layout in self._main_content_cols])
            else:
                message_line_width = self._main_content_panel.width or self._console.width
            message_line_width -= 2 # Left and right border
            content_items = self._content_items.copy()
            content_items.reverse()
            displayed_items = []
            line_count = 0
            for item in content_items:
                lines = item["content"].splitlines()
                lines.reverse()
                for line in lines:
                    if line_count > message_line_count:
                        break

                    if item["type"] == "section":
                        line = self._center_line(line)
                    elif item["type"] == "rule" or item["type"] == "subrule":
                        line = line.strip()
                        if len(line) == 0:
                            line_count += 1
                            displayed_items.append(Text(""))
                            continue
                        else:
                            line = self._format_rule(line, item["type"], item["style"])
                            line = self._center_line(line)
                    elif item["type"] == "text":
                        line = self._format_text(self._clean_text(line), item["time"])

                    rich_line = Text.from_markup(line)

                    if len(rich_line) > message_line_width:
                        sublines = rich_line.wrap(
                                console=self._console,
                                width=message_line_width
                        )
                        sublines = [*sublines]
                        sublines.reverse()
                        line_count += len(sublines)
                        displayed_items.extend(sublines)
                    else:
                        line_count += 1
                        displayed_items.append(line)
            displayed_items.reverse()
            renderables = displayed_items[-message_line_count:]
            self._main_content_panel.renderable = Group(*renderables)
            self._live.refresh()

    def _center_line(self, line):
        console_width = self._main_content_panel.width or self._console.width
        console_width -= 2 # Left and right border
        char_width = self._console.measure(line).maximum
        line_offset = (console_width - char_width) // 2
        return " " * line_offset + line

    def _clean_text(self, text, remove_styling=False):
        text.strip()
        if remove_styling:
            rich_text = Text.from_markup(text)
            return rich_text.plain
        else:
            return text

    def _format_rule(self, rule_text, rule_type: Literal["section", "rule", "subrule"], rule_style=Colors.HEADER):
        main_panel_width = self._main_content_panel.width or self._console.width
        available_width = ((main_panel_width-4-self._console.measure(rule_text).maximum)//2)
        if rule_type == "rule":
            bar_width = min(available_width, 20)
            rule_text = f"{Colors.apply_style('━'*bar_width, rule_style)} {rule_text} {Colors.apply_style('━'*bar_width, rule_style)}"
        elif rule_type == "subrule":
            bar_width = min(available_width, 10)
            rule_text = f"{Colors.apply_style('-'*bar_width, rule_style)} {rule_text} {Colors.apply_style('-'*bar_width, rule_style)}"
        return rule_text

    def _format_text(self, text, msg_time=None):
        total_pad = 1
        if self._cfg.show_time:
            if msg_time is not None:
                dt_utc = datetime.datetime.fromtimestamp(msg_time, tz=datetime.timezone.utc)
                dt_target_tz = dt_utc.astimezone(self._tz_info)
                time_format = self._cfg.time_format.value
                time_format = time_format.replace(":", f"{Colors.header(':')}").replace("%p", f"{Colors.header('%p')}")
                formatted_time_string = dt_target_tz.strftime(time_format)
                text = f"[{Colors.orange(formatted_time_string)}] {text}"
            else:
                match self._cfg.time_format:
                    case TimeFormat.NO_SECONDS:
                        total_pad += 11
                    case TimeFormat.NO_AM_PM:
                        total_pad += 11
                    case TimeFormat.NO_SECONDS_NO_AM_PM:
                        total_pad += 8
                    case TimeFormat.TWENTY_FOUR_HOUR:
                        total_pad += 11
                    case TimeFormat.TWENTY_FOUR_HOUR_NO_SECONDS:
                        total_pad += 8
                    case _:
                        total_pad += 14
        text = " " * total_pad + text
        return text

    def _create_progress_bar(self):
        spinner_col = SpinnerColumn(table_column=Column(max_width=3))
        desc_col = TextColumn(text_format="[progress.description]{task.description}", style=Colors.CYAN, table_column=Column(max_width=30, min_width=15))
        bar_col = BarColumn(bar_width=None, complete_style=Colors.PROGRESS_FILL)
        pct_col = TaskProgressColumn(table_column=Column(max_width=10))
        time_elapsed_col = TimeElapsedColumn(table_column=Column(max_width=15))
        time_remaining_col = ConditionalTimeRemainingColumn(table_column=Column(max_width=15))
        self._progress_bar = Progress(
            spinner_col, desc_col, bar_col, pct_col, time_elapsed_col, time_remaining_col,
            console=self._console, transient=True, expand=True
        )

        if self._cfg.use_live_display:
            if self._cfg.use_stats:
                self._stats_title = Text.from_markup(Colors.header("Statistics & Info"))
            else:
                self._stats_title = Text.from_markup(Colors.header(""))
            self._stats_panel = Layout(self._stats_title, name="stats", size=3)
            self._progress_panel = Layout(self._progress_bar, name="progress_bar", size=3)
            self._layout["progress"].split_column(
                self._progress_panel,
                self._stats_panel,
            )
            self._layout["progress"].visible = True

if __name__ == "__main__":
    TPConsole().print("Hello, world!")
    TPConsole().update_app_stage("Testing...")

    TPConsole().create_progress_task("Looping", "Looping Progress", total=100, transient=True)

    for i in range(100):
        TPConsole().update_progress_task("Looping", advance=1)
        TPConsole().update_app_stage(f"Processed {i} items")
        TPConsole().print(f"Iteration {i}")
        sleep(0.1)

    TPConsole().print("Done!")
    sleep(0.5)