#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${script_dir}/.." && pwd)"

if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

if [ -f "${workspace_dir}/install/setup.bash" ]; then
    source "${workspace_dir}/install/setup.bash"
fi

ros2 service call /pick_moveit_executor_node/move_to_start_pose std_srvs/srv/Trigger "{}"
