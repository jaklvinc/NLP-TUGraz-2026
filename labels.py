#Label vocabulary to fix data

from typing import Literal

Label = Literal["Fact", "Opinion"]
LABELS: tuple[Label, ...] = ("Fact", "Opinion")
