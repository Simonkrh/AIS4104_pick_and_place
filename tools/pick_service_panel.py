#!/usr/bin/env python3
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger


SERVICES = [
    ("move_to_start_pose", "Move to start", "/pick_moveit_executor_node/move_to_start_pose"),
    ("open_gripper", "Open gripper", "/pick_moveit_executor_node/open_gripper"),
    ("close_gripper", "Close gripper", "/pick_moveit_executor_node/close_gripper"),
    ("execute_approach", "Approach", "/pick_moveit_executor_node/execute_approach"),
    (
        "execute_centered_approach",
        "Centered approach",
        "/pick_moveit_executor_node/execute_centered_approach",
    ),
    ("execute_grasp", "Grasp", "/pick_moveit_executor_node/execute_grasp"),
    ("execute_pick", "Full pick", "/pick_moveit_executor_node/execute_pick"),
    (
        "search_workspace",
        "Search workspace",
        "/pick_moveit_executor_node/search_workspace",
    ),
    ("run_dice_test", "Run dice test", "/pick_moveit_executor_node/run_dice_test"),
    ("stop_dice_test", "Stop dice test", "/pick_moveit_executor_node/stop_dice_test"),
    ("run_sorting", "Run sorting", "/pick_moveit_executor_node/run_sorting"),
    ("stop_sorting", "Stop sorting", "/pick_moveit_executor_node/stop_sorting"),
]


class PickServicePanel:
    def __init__(self) -> None:
        rclpy.init()
        self.node = rclpy.create_node("pick_service_panel")
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
        )
        self.executor_thread.start()

        self.clients = {
            key: self.node.create_client(Trigger, service_name)
            for key, _label, service_name in SERVICES
        }
        self.pending: dict[str, object] = {}
        self.events: queue.Queue[tuple[str, str, object]] = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Pick service panel")
        self.root.minsize(520, 430)
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.shutting_down = False

        self.status_var = tk.StringVar(value="Waiting for services.")
        self.buttons: dict[str, ttk.Button] = {}
        self._build_ui()
        self._append_log("Panel is ready. Start the pick and place launch first.")
        self._poll_events()
        self._refresh_buttons()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            outer,
            text="Pick-and-place services",
            font=("TkDefaultFont", 14, "bold"),
        )
        title.pack(anchor=tk.W)

        status = ttk.Label(outer, textvariable=self.status_var)
        status.pack(anchor=tk.W, pady=(4, 10))

        button_grid = ttk.Frame(outer)
        button_grid.pack(fill=tk.X)

        for index, (key, label, _service_name) in enumerate(SERVICES):
            button = ttk.Button(
                button_grid,
                text=label,
                command=lambda service_key=key: self.call_service(service_key),
            )
            row = index // 2
            column = index % 2
            button.grid(row=row, column=column, sticky=tk.EW, padx=4, pady=4)
            self.buttons[key] = button

        button_grid.columnconfigure(0, weight=1)
        button_grid.columnconfigure(1, weight=1)

        log_frame = ttk.LabelFrame(outer, text="Log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.log_text = tk.Text(log_frame, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            log_frame,
            orient=tk.VERTICAL,
            command=self.log_text.yview,
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.status_var.set(text)

    def call_service(self, key: str) -> None:
        client = self.clients[key]
        if key in self.pending:
            return
        if not client.service_is_ready():
            self._append_log(f"{SERVICES_BY_KEY[key][1]} is not ready yet.")
            return

        self._append_log(f"Calling {SERVICES_BY_KEY[key][1]}.")
        future = client.call_async(Trigger.Request())
        self.pending[key] = future
        future.add_done_callback(
            lambda done_future, service_key=key: self.events.put(
                ("done", service_key, done_future)
            )
        )
        self._update_buttons()

    def _poll_events(self) -> None:
        if self.shutting_down:
            return

        while True:
            try:
                event, key, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if event == "done":
                self.pending.pop(key, None)
                label = SERVICES_BY_KEY[key][1]
                try:
                    result = payload.result()
                except Exception as exc:
                    self._append_log(f"{label} had a problem. {exc}")
                else:
                    outcome = "Done" if result.success else "Stopped"
                    message = result.message.strip() or "No message."
                    self._append_log(f"{label}. {outcome}. {message}")

        self._update_buttons()
        self.root.after(100, self._poll_events)

    def _update_buttons(self) -> None:
        running_stop_key = ""
        if "run_dice_test" in self.pending:
            running_stop_key = "stop_dice_test"
        if "run_sorting" in self.pending:
            running_stop_key = "stop_sorting"
        ready_count = 0

        for key, _label, _service_name in SERVICES:
            client = self.clients[key]
            service_ready = client.service_is_ready()
            if service_ready:
                ready_count += 1

            disabled = not service_ready or key in self.pending
            if running_stop_key and key != running_stop_key:
                disabled = True

            self.buttons[key].configure(state=tk.DISABLED if disabled else tk.NORMAL)

        if not self.pending:
            self.status_var.set(f"{ready_count} of {len(SERVICES)} services are ready.")

    def _refresh_buttons(self) -> None:
        if self.shutting_down:
            return

        self._update_buttons()
        self.root.after(1000, self._refresh_buttons)

    def shutdown(self) -> None:
        self.shutting_down = True
        self._append_log("Shutting down panel.")
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


SERVICES_BY_KEY = {key: (key, label, service_name) for key, label, service_name in SERVICES}


def main() -> None:
    panel = PickServicePanel()
    panel.run()


if __name__ == "__main__":
    main()
