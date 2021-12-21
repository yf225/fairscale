# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Import most common subpackages
################################################################################

from typing import List

from setup import find_version

__version__ = find_version("version.json")
__all__: List[str] = []
