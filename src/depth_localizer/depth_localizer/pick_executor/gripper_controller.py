from __future__ import annotations

import socket
import time


class GripperControllerMixin:
    def _send_gripper_command(self, label: str) -> tuple[bool, str]:
        if label == "open":
            width = 36.9
            force = 80
        elif label == "close":
            width = 0.0
            force = 30
        else:
            return False, f"Unsupported gripper command: {label}."

        script = f"""sec gripper_{label}():
            on_tool_xmlrpc = rpc_factory("xmlrpc", "http://localhost:41414")
            on_tool_xmlrpc.twofg_grip_external(0, {width}, {force}, 100)
            end
            """

        try:
            with socket.create_connection(
                (self.robot_ip, self.ROBOT_SCRIPT_PORT),
                timeout=self.ROBOT_SCRIPT_TIMEOUT_SEC,
            ) as sock:
                sock.sendall(script.encode("utf-8"))
        except OSError as exc:
            return False, f"Failed to send gripper {label} command. {exc}"

        message = f"Gripper {label} command sent."
        self._publish_status(message)
        time.sleep(self.GRIPPER_SETTLE_SEC)
        return True, message
