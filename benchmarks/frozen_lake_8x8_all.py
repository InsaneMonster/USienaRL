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

from usienarl import command_line_parse

# Import required src
# Require error handling to support both deployment and pycharm versions

try:
    from frozen_lake_8x8_dddql import run as dddql_run
    from frozen_lake_8x8_ddql import run as ddql_run
    from frozen_lake_8x8_dql import run as dql_run
    from frozen_lake_8x8_desarsa import run as desarsa_run
    from frozen_lake_8x8_dsarsa import run as dsarsa_run
    from frozen_lake_8x8_tql import run as tql_run
    from frozen_lake_8x8_tsarsa import run as tsarsa_run
    from frozen_lake_8x8_tesarsa import run as tesarsa_run
    from frozen_lake_8x8_vpg import run as vpg_run
    from frozen_lake_8x8_ppo import run as ppo_run
except ImportError:
    from benchmarks.frozen_lake_8x8_dddql import run as dddql_run
    from benchmarks.frozen_lake_8x8_ddql import run as ddql_run
    from benchmarks.frozen_lake_8x8_dql import run as dql_run
    from benchmarks.frozen_lake_8x8_desarsa import run as desarsa_run
    from benchmarks.frozen_lake_8x8_dsarsa import run as dsarsa_run
    from benchmarks.frozen_lake_8x8_tql import run as tql_run
    from benchmarks.frozen_lake_8x8_tsarsa import run as tsarsa_run
    from benchmarks.frozen_lake_8x8_tesarsa import run as tesarsa_run
    from benchmarks.frozen_lake_8x8_vpg import run as vpg_run
    from benchmarks.frozen_lake_8x8_ppo import run as ppo_run

if __name__ == "__main__":
    # Remove tensorflow deprecation warnings
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Run all frozen lake 8x8 experiments
    dddql_run(workspace_path, experiment_iterations_number,
              render_during_training, render_during_validation, render_during_test)
    ddql_run(workspace_path, experiment_iterations_number,
             render_during_training, render_during_validation, render_during_test)
    dql_run(workspace_path, experiment_iterations_number,
            render_during_training, render_during_validation, render_during_test)
    desarsa_run(workspace_path, experiment_iterations_number,
                render_during_training, render_during_validation, render_during_test)
    dsarsa_run(workspace_path, experiment_iterations_number,
               render_during_training, render_during_validation, render_during_test)
    tql_run(workspace_path, experiment_iterations_number,
            render_during_training, render_during_validation, render_during_test)
    tesarsa_run(workspace_path, experiment_iterations_number,
                render_during_training, render_during_validation, render_during_test)
    tsarsa_run(workspace_path, experiment_iterations_number,
               render_during_training, render_during_validation, render_during_test)
    vpg_run(workspace_path, experiment_iterations_number,
            render_during_training, render_during_validation, render_during_test)
    ppo_run(workspace_path, experiment_iterations_number,
            render_during_training, render_during_validation, render_during_test)
