import math
import numpy as np
from typing import Optional

from minicortex.core.node import Node
from minicortex.core.descriptors.ports import InputPort, OutputPort
from minicortex.core.descriptors.properties import Range, Integer, Bool
from minicortex.core.descriptors.displays import Vector2D, Text
from minicortex.core.descriptors.actions import Action
from minicortex.core.descriptors import branch


class Test(Node):

    text = Text("Lab")

    def init(self):
        self.i = 0
        1/0

    def process(self):
        self.i += 1
        self.text = f"{self.i}"

        pass
