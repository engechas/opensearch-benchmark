import sys


class IterationBasedStopWatch:
    __test__ = False

    def __init__(self, max_iterations):
        self.iterations = 0
        self.max_iterations = max_iterations

    def start(self):
        self.iterations = 0

    def split_time(self):
        if self.iterations < self.max_iterations:
            self.iterations += 1
            return 0
        else:
            return sys.maxsize


class TestClock:
    __test__ = False

    def __init__(self, stop_watch):
        self._stop_watch = stop_watch

    def stop_watch(self):
        return self._stop_watch
