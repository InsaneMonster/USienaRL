#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# USienaRL is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import os

# Import usienarl

from usienarl.utils import command_line_parse

# Import required src
# Require error handling to support both deployment and pycharm versions

try:
    from cart_pole_random import run as random_run
    from cart_pole_dddql import run as dddql_run
    from cart_pole_ddql import run as ddql_run
    from cart_pole_dql import run as dql_run
    from cart_pole_desarsa import run as desarsa_run
    from cart_pole_dsarsa import run as dsarsa_run
    from cart_pole_vpg import run as vpg_run
    from cart_pole_ppo import run as ppo_run
except ImportError:
    from benchmarks.cart_pole_random import run as random_run
    from benchmarks.cart_pole_dddql import run as dddql_run
    from benchmarks.cart_pole_ddql import run as ddql_run
    from benchmarks.cart_pole_dql import run as dql_run
    from benchmarks.cart_pole_desarsa import run as desarsa_run
    from benchmarks.cart_pole_dsarsa import run as dsarsa_run
    from benchmarks.cart_pole_vpg import run as vpg_run
    from benchmarks.cart_pole_ppo import run as ppo_run

if __name__ == "__main__":
    # Remove tensorflow deprecation warnings
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Run all cart pole experiments
    random_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    dddql_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    ddql_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    dql_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    desarsa_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    dsarsa_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    vpg_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
    ppo_run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
