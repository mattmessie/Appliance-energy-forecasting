"""Chronos foundation model (Part 7): zero-shot, target-only rolling forecast.

Chronos (amazon/chronos-t5-*) is a pretrained time-series foundation model
used here zero-shot -- no fitting/training step at all, unlike SARIMAX or
the XGBoost model. Given a numeric context (the history so far), it directly
samples possible future trajectories; point forecasts and confidence
intervals are read off the sample quantiles.

Used TARGET-ONLY. Unlike SARIMAX (which can take exog=) or the feature-based
model (which can take arbitrary covariate columns), Chronos's forecasting
API takes only a single numeric context series -- it has no mechanism for
passing weather/sensor covariates alongside it. This is a genuine modelling
difference worth naming explicitly in the report (Part 9 Q4/Q5), not a
shortcut: Chronos cannot use the covariate information the other models can.

Same rolling design as the rest of the pipeline (see report.md, Section 4):
14 daily origins, each forecasting the next 24 hours from an expanding
history. There is no parameter state to update between origins (Chronos is
stateless/zero-shot), so "rolling" here just means re-calling predict() with
a longer context at each origin.

NOTE ON EXECUTION: this module requires downloading pretrained weights from
Hugging Face Hub at runtime (`ChronosPipeline.from_pretrained(...)`), which
this development sandbox's network policy blocks (huggingface.co is not on
the allowlist). The `chronos-forecasting` package itself WAS installed here
from PyPI (no Hugging Face access needed for that), which let the code
below be checked directly against the real installed library's source
(`chronos/chronos.py`, `chronos/base.py`) rather than written from memory --
in particular this caught that `predict()`/`predict_quantiles()` take an
`inputs=` argument, not `context=` as an earlier draft assumed. The rolling/
indexing/leakage logic is also unit-tested against a stub pipeline with the
exact same input/output tensor shapes as the real one (see
tests/test_foundation.py). What has NOT been run here is inference against
the actual pretrained weights -- that must happen on a machine with normal
internet access (see scripts/run_chronos.py).
"""

import numpy as np
import pandas as pd


def load_chronos_pipeline(model_name: str = "amazon/chronos-t5-tiny", device_map: str = "cpu"):
    """Load a pretrained Chronos pipeline.

    `chronos-t5-tiny` (~8M params) is the default: smallest download,
    fastest inference, sufficient for a zero-shot comparison baseline.
    Swap to `amazon/chronos-t5-small` for a possibly-stronger forecast if
    the extra download/runtime is acceptable.

    Deliberately doesn't pass a dtype override -- `transformers` recently
    renamed `torch_dtype` to `dtype` (deprecating the old name), and since
    this code will run on whatever transformers version happens to be
    installed locally, it's safer to just take the library default (fp32 on
    CPU) than hardcode a kwarg name that may not match every version.
    """
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(model_name, device_map=device_map)
    return pipeline


def rolling_chronos_forecast(
    pipeline,
    train: pd.Series,
    test: pd.Series,
    block_size: int = 24,
    max_context: int = 512,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> dict:
    """Roll a zero-shot Chronos pipeline forward across the test period.

    At each daily origin, the model is given the trailing `max_context`
    hours of history (capped so the input doesn't exceed what the model was
    trained on -- Chronos truncates internally too, but capping explicitly
    keeps memory/runtime predictable) and asked for quantile forecasts over
    the next 24 hours via `predict_quantiles`. The point forecast is the
    median (0.5 quantile); confidence bounds are `quantile_low`/
    `quantile_high` (default 5%/95% -> a 90% interval).

    Parameters
    ----------
    pipeline : ChronosPipeline
        Loaded via `load_chronos_pipeline`.
    train : pd.Series
        Initial training series.
    test : pd.Series
        Full test period (length must be a multiple of `block_size`).
    block_size : int
        Forecast horizon per origin, in hours.
    max_context : int
        Maximum trailing history (in hours) passed as context at each origin.
    quantile_low, quantile_high : float
        Quantiles used for the confidence interval.

    Returns
    -------
    dict with "forecast" (median), "lower", "upper" -- each a pd.Series
    covering the whole test period.
    """
    if len(test) % block_size != 0:
        raise ValueError(
            f"Test period length ({len(test)}) is not a multiple of "
            f"block_size ({block_size})."
        )

    n_blocks = len(test) // block_size
    history = train.copy()

    forecast_parts, lower_parts, upper_parts = [], [], []
    quantile_levels = [quantile_low, 0.5, quantile_high]

    for k in range(n_blocks):
        block_index = test.index[k * block_size : (k + 1) * block_size]

        context_values = history.iloc[-max_context:].values.astype("float32")
        quantiles = _predict_quantiles(
            pipeline, context_values, block_size, quantile_levels
        )
        # quantiles shape: (block_size, 3) -> columns [low, median, high]

        forecast_parts.append(pd.Series(quantiles[:, 1], index=block_index))
        lower_parts.append(pd.Series(quantiles[:, 0], index=block_index))
        upper_parts.append(pd.Series(quantiles[:, 2], index=block_index))

        # Reveal this block's actuals before moving to the next origin.
        history = pd.concat([history, test.loc[block_index]])

    return {
        "forecast": pd.concat(forecast_parts).rename("foundation_model"),
        "lower": pd.concat(lower_parts).rename("foundation_model_lower"),
        "upper": pd.concat(upper_parts).rename("foundation_model_upper"),
    }


def _predict_quantiles(pipeline, context_values, prediction_length, quantile_levels):
    """Call the Chronos pipeline's predict_quantiles() and return a plain
    (prediction_length, len(quantile_levels)) numpy array.

    Isolated into its own function so tests can stub it without needing
    torch or the real model installed. Uses the documented
    chronos-forecasting API: `inputs=` (a 1D tensor for a single series),
    NOT `context=` -- an earlier draft of this module used the wrong
    keyword; verified against the installed package's source directly.
    """
    import torch

    context_tensor = torch.tensor(context_values)
    quantile_tensor, _mean = pipeline.predict_quantiles(
        inputs=context_tensor,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    # quantile_tensor shape: (batch_size, prediction_length, num_quantiles);
    # batch_size is 1 for a single series.
    return quantile_tensor[0].numpy()
