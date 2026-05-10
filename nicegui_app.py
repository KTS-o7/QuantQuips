# nicegui_app.py
from __future__ import annotations

from nicegui import app, ui


NAV_ITEMS = [
    ("Home", "/"),
    ("Backtesting", "/backtest"),
    ("Genetic Algorithm", "/ga"),
    ("LLM", "/llm"),
    ("About", "/about"),
]


def layout() -> None:
    """Render shared header and left drawer. Call inside every @ui.page."""
    with ui.header().classes("items-center justify-between bg-grey-9 text-white q-px-md"):
        ui.label("QuantQuips").classes("text-h6 text-bold")
        ui.button(icon="dark_mode", on_click=lambda: ui.dark_mode().toggle()).props("flat round")

    with ui.left_drawer(top_corner=True).classes("bg-grey-10 text-white q-pa-md"):
        ui.label("Navigation").classes("text-subtitle2 text-grey-5 q-mb-sm")
        for label, path in NAV_ITEMS:
            ui.link(label, path).classes("text-white block q-py-xs")


@ui.page("/")
def page_home() -> None:
    layout()
    ui.label("Home — coming in Task 3").classes("text-h6 q-pa-lg")


@ui.page("/backtest")
def page_backtest() -> None:
    layout()
    ui.label("Backtesting — coming in Task 4").classes("text-h6 q-pa-lg")


@ui.page("/ga")
def page_ga() -> None:
    layout()
    ui.label("Genetic Algorithm — coming in Task 5").classes("text-h6 q-pa-lg")


@ui.page("/llm")
def page_llm() -> None:
    layout()
    ui.label("LLM — coming in Task 6").classes("text-h6 q-pa-lg")


@ui.page("/about")
def page_about() -> None:
    layout()
    ui.label("About — coming in Task 7").classes("text-h6 q-pa-lg")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="QuantQuips", dark=True, port=8080, reload=False)
