import pandas as pd
import numpy as np

class recorder:

    def __init__(
        self,
        recording_state,
        X,
    ):
        self.recording_state = recording_state
        self.df = pd.DataFrame(
            {}, index=X.index,
        )

    def record_snapshot(self, yhat, cycle):

        if not self.recording_state:
            pass
        else:
            self.df[cycle] = yhat
