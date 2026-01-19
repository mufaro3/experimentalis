"""
This is just for a collection of functions that don't particularly belong
in any other module, but are useful to have around (just in case).
"""

def calculate_t_score(a, da, b, db):
    """
    Calculates the `t`-score used for determining the similarity between
    two measurements via the formula

    .. math::

        t' = \frac{\abs{a - b}}{\sqrt{da^2 + db^2}}
    """
    return np.abs(a - b) / np.sqrt(da ** 2 + db ** 2)
