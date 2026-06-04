from .common_imports import *


def extract_metrics(log_dir):
    # Initialize the event accumulator
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Get the list of tags
    tags = ea.Tags()["scalars"]

    # Extract the metrics
    metrics = {}
    for tag in tags:
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]

    return metrics
