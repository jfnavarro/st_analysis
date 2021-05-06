"""
Defines a ST Dataset and functions to interact with it
"""

import pandas as pd
import numpy as np
from typing import List, Dict

class STDataset:
    """
    """

    def __init__(self,
                 counts: pd.DataFrame,
                 coordinates: pd.Series
                 ) -> None:

        self.counts = counts
        self.coordinates = coordinates
        self.genes = counts.columns
        self.index = counts.index

    def __getitem__(self,
                    idx: List[int],
                   )-> Dict:
        """Get sample with specified index
        Parameter:
        ---------
        idx : List[int]
            list of indices for samples to
            be returned
        Returns:
        -------
        Dictionary with sampe expression (x),
        label (meta), size factor (sf) and specified
        indices (gidx)
        """
        sample = {'x' : self.cnt[idx,:],
                  'meta' : self.zidx[idx],
                  'sf' : self.libsize[idx],
                  'gidx' : t.tensor(idx),
                 }

        return sample

    def __len__(self,
               )-> int:
        """Length of CountData object"""

        return self.M

