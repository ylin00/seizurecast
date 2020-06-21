"""This script prints memory consumption during the execution.
For debug only. Must be manually inserted into the profiling code.
"""
from pympler import tracker
tr = tracker.SummaryTracker()

tr.print_diff()  # <-- insert this line into the place where memory consumption is a concern.
