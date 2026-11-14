import time
from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown


class MarkdownTypewriter:
    """Streams text chunks smoothly while rendering Rich Markdown in real-time."""

    def __init__(
        self,
        delay: float = 0.005,
        console: Optional[Console] = None,
        refresh_per_second: float = 30,
    ) -> None:
        self.delay = delay
        self.console = console or Console()
        self.refresh_per_second = refresh_per_second
        self.accumulated = ""
        self.live: Optional[Live] = None

    def start(self) -> None:
        if self.live is None:
            self.live = Live(
                Markdown(""),
                console=self.console,
                refresh_per_second=self.refresh_per_second,
                vertical_overflow="visible",
                auto_refresh=True,
            )
            self.live.start()

    def write(self, text: str) -> None:
        if not text:
            return
        if self.live is None:
            self.start()

        if self.delay <= 0:
            self.accumulated += text
            if self.live is not None:
                self.live.update(Markdown(self.accumulated))
            return

        for char in text:
            self.accumulated += char
            if self.live is not None:
                self.live.update(Markdown(self.accumulated))
            time.sleep(self.delay)

    def close(self) -> None:
        if self.live is not None:
            try:
                self.live.stop()
            finally:
                self.live = None

    def __enter__(self) -> "MarkdownTypewriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def print_typewriter(
    text: str,
    delay: float = 0.05,
    request_new_line: bool = False,
    markdown: bool = False,
    console: Optional[Console] = None,
) -> None:
    """Print text with smooth typewriter effect. If markdown=True, render styled Rich Markdown."""
    if not text:
        if request_new_line:
            print()
        return

    if markdown:
        with MarkdownTypewriter(delay=delay, console=console) as writer:
            writer.write(text)
        if request_new_line:
            print()
    else:
        for char in text:
            print(char, end="", flush=True)
            if delay > 0:
                time.sleep(delay)

        if request_new_line:
            print()


def print_typewriter_markdown(
    text: str,
    delay: float = 0.005,
    request_new_line: bool = True,
    console: Optional[Console] = None,
) -> None:
    """Convenience function to print Markdown formatted text with smooth typing."""
    print_typewriter(
        text,
        delay=delay,
        request_new_line=request_new_line,
        markdown=True,
        console=console,
    )


