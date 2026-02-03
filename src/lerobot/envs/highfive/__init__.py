# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""High-Five Robot Environment for SO-ARM101.

A MuJoCo-based environment where a robot arm tracks and intercepts
a moving hand target for a high-five/fist-bump interaction.
"""

from lerobot.envs.highfive.highfive_env import HighFiveEnv, create_highfive_envs

__all__ = ["HighFiveEnv", "create_highfive_envs"]
