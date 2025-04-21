import datetime
import time
from time import sleep
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

from pyfiglet import Figlet
from rich.align import Align
from rich.console import Console, Group
from rich.box import Box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, \
    TimeRemainingColumn, Task
from rich.table import Column, Table
from rich.text import Text


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
    BOLD = 'bold'
    UNDERLINE = 'underline'

    @staticmethod
    def header(text):
        return f"[{Colors.HEADER} {Colors.BOLD}]{text}[/{Colors.HEADER} {Colors.BOLD}]"

    @staticmethod
    def info(text):
        return f"[{Colors.BLUE}]{text}[/{Colors.BLUE}]"

    @staticmethod
    def yellow(text):
        return f"[{Colors.YELLOW}]{text}[/{Colors.YELLOW}]"

    @staticmethod
    def purple(text):
        return f"[{Colors.PURPLE}]{text}[/{Colors.PURPLE}]"

    @staticmethod
    def purple2(text):
        return f"[{Colors.PURPLE2}]{text}[/{Colors.PURPLE2}]"

    @staticmethod
    def progress_fill(text):
        return f"[{Colors.PROGRESS_FILL}]{text}[/{Colors.PROGRESS_FILL}]"

    @staticmethod
    def medium_grey(text):
        return f"[{Colors.MED_GREY}]{text}[/{Colors.MED_GREY}]"

    @staticmethod
    def orange(text):
        return f"[{Colors.ORANGE}]{text}[/{Colors.ORANGE}]"

    @staticmethod
    def success(text):
        return f"[{Colors.GREEN}]{text}[/{Colors.GREEN}]"

    @staticmethod
    def warning(text):
        return f"[{Colors.YELLOW}]{text}[/{Colors.YELLOW}]"

    @staticmethod
    def error(text):
        return f"[{Colors.RED}]{text}[/{Colors.RED}]"

    @staticmethod
    def highlight(text):
        return f"[{Colors.CYAN} {Colors.BOLD}]{text}[/{Colors.CYAN} {Colors.BOLD}]"

    @staticmethod
    def bold(text):
        return f"[{Colors.BOLD}]{text}[/{Colors.BOLD}]"

    @staticmethod
    def underline(text):
        return f"[{Colors.UNDERLINE}]{text}[/{Colors.UNDERLINE}]"

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
            final_result = f"\n{final_result}"
        if padding_bottom:
            final_result = f"{final_result}\n"

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
    _use_live: bool = False
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

    # Default text content
    _app_title = "Tabula Prima"
    _app_subtitle = "Training"
    _app_stage = "Bootstrapping"
    _messages: List[Dict[str, Any]] = []
    _tz_info = None
    _section_titles = []
    _main_content_panel = Panel(
        "",
        expand=True, border_style="none", box=EMPTY_BOX, padding=0
    )
    _tabula_prima_fig = Text.from_markup(
        Colors.apply_gradient_to_lines(Figlet(font='contessa').renderText("TabulaPrima"), Colors.BLUE, Colors.PURPLE2)
    )

    def __new__(cls, use_live: bool = False, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._console = Console(soft_wrap=True, color_system="truecolor")
            cls._use_live = use_live
            if use_live:
                cls._tz_info = ZoneInfo("America/Los_Angeles")
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

    def progress_start(
            self,
            use_stats=False,
    ):
        self._create_progress_bar(use_stats=use_stats)
        if not self._use_live:
            self._progress_bar.start()

    def progress_stop(self):
        self._progress_bar.stop()
        if self._use_live:
            self._layout["progress"].visible = False

    def handle_exception(self):
        self._console.print_exception()

    def print(self, content: str):
        self._print_single_message(content)

    def print_list_item(self, title: str, content: str):
        text = f"{Colors.purple('»')} {Colors.info(title)}: {Colors.highlight(content)}"
        self._print_single_message(text)

    def print_list(self, items: List[Dict[str, str]]):
        for item in items:
            self.print_list_item(item["title"], item["content"])

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

    def rule(self, content, style=Colors.HEADER):
        if self._use_live:
            self._messages.append(self._format_message("", False))
            self._messages.append(self._format_message(f"[{style}]>>[/{style}] {content} [{style}]<<[/{style}]", False))
            self._messages.append(self._format_message("", False))
            self._update_main_render()
        else:
            self._console.rule(content, style=style)

    def section(self, content: str):
        title_text = Colors.apply_gradient_to_lines(Figlet(font='cybermedium', width=160).renderText(content), Colors.BLUE, Colors.HEADER, False, True)
        if self._use_live:
            idx = len(self._section_titles)
            self._section_titles.append(title_text)
            for j in range(len(title_text.splitlines())):
                self._messages.append(self._format_section_title(title_text, idx))
            self._update_main_render()
        else:
            self._console.print(Align.center(Text.from_markup(title_text)))

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
        self._progress_tasks[task_name] = task_id

    def update_progress_task(self, task_name: str, completed: float|None = None, **kwargs):
        task_id = self._progress_tasks[task_name]
        self._progress_bar.update(task_id, completed=completed, **kwargs)
        target_task = None
        for task in self._progress_bar.tasks:  # Iterate through the current tasks list
            if task.id == task_id:
                target_task = task
                break  # Found the task
        if target_task is not None and target_task.finished and len(self._progress_tasks) > 1:
            self._progress_bar.remove_task(task_id)
            del self._progress_tasks[task_name]

    def remove_progress_task(self, task_name: str):
        task_id = self._progress_tasks[task_name] if hasattr(self._progress_tasks, task_name) else None
        if task_id is None:
            return None
        self._progress_bar.stop_task(task_id)
        self._progress_bar.remove_task(task_id)
        del self._progress_tasks[task_name]

    def has_progress_task(self, task_name: str):
        return task_name in self._progress_tasks

    def _print_single_message(self, text, with_time=True):
        if self._use_live:
            self._messages.append(self._format_message(text, with_time))
            self._update_main_render()
        else:
            self._console.print(text)

    def _format_message(self, message, with_time=True):
        return {
            "message": message.replace("\n", ""),
            "time": time.time() if with_time else None,
        }

    def _format_section_title(self, title, idx):
        return {
            "message": "",
            "time": None,
            "ref": idx,
            "height": len(title.splitlines())
        }

    def _update_main_render(self):
        if self._use_live:
            message_line_count = self._console.size.height
            if self._layout['progress'].visible:
                message_line_count -= 6 # Progress/Stats panel
            message_line_count -= 3  # Title bar
            message_line_count -= 3  # Top and bottom padding
            displayed_messages = self._messages[-message_line_count:]
            section = 0
            section_processed = False
            titles = []
            texts = []
            text = ""
            for j in range(len(displayed_messages)):
                if "ref" in displayed_messages[j].keys():
                    if not section_processed:
                        texts.append(text)
                        text = ""
                        section_processed = True
                        section += 1
                        titles.append(self._section_titles[displayed_messages[j]["ref"]])
                    else:
                        continue
                else:
                    section_processed = False
                    if text != "":
                        text += "\n"
                    if displayed_messages[j]["time"] is not None:
                        dt_utc = datetime.datetime.fromtimestamp(displayed_messages[j]['time'], tz=datetime.timezone.utc)
                        dt_target_tz = dt_utc.astimezone(self._tz_info)
                        text += f" [[{Colors.ORANGE}]{dt_target_tz.strftime('%I:%M:%S %p')}[/{Colors.ORANGE}]] {displayed_messages[j]['message']}"
                    else:
                        text += f"               {displayed_messages[j]['message']}"
            texts.append(text)
            if section == 0:
                self._main_content_panel.renderable = Text.from_markup(text)
            else:
                renderables = []
                max_text_index = len(texts)-1
                max_title_index = len(titles)-1
                max_index = max(max_text_index, max_title_index)
                for k in range(max_index+1):
                    if k <= max_text_index:
                        renderables.append(Text.from_markup(texts[k]))
                    if k <= max_title_index:
                        renderables.append(Align.center(Text.from_markup(titles[k])))
                self._main_content_panel.renderable = Group(*renderables)


    def _create_progress_bar(self, use_stats):
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

        if self._use_live:
            if use_stats:
                self._stats_title = Text.from_markup(Colors.header("Statistics & Info"))
                self._stats_panel = Layout(self._stats_title, name="stats", size=3)
                self._progress_panel = Layout(self._progress_bar, name="progress_bar", size=3)
                self._layout["progress"].split_column(
                    self._progress_panel,
                    self._stats_panel,
                )
            else:
                self._progress_content_layout.update(self._progress_bar)
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