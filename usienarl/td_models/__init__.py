#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# USienaRL is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import scripts

from .tabular_q_learning import TabularQLearning
from .deep_q_learning import DeepQLearning
from .double_deep_q_learning import DoubleDeepQLearning
from .dueling_deep_q_learning import DuelingDeepQLearning

from .tabular_sarsa import TabularSARSA
from .deep_sarsa import DeepSARSA

from .tabular_expected_sarsa import TabularExpectedSARSA
from .deep_expected_sarsa import DeepExpectedSARSA
