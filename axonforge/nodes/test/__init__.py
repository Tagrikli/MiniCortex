import math
import numpy as np
from typing import Optional

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.properties import Range, Integer, Bool
from axonforge.core.descriptors.displays import Vector2D, Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch


class Test(Node):

    text = Text("Lab")

    def init(self):
        self.i = 0
        1/0

    def process(self):
        self.i += 1
        self.text = f"{self.i}"

        pass
