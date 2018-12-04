import time, logging, json

class benchmarker:
    def __init__(self, name=None, printer=logging.info, warner=logging.warn):
        self.benchmark_start = None
        self.lap_start = None
        self.times = []
        self.name = name
        self.prefix = "[BENCHMARK"+(" "+name if name else "")+"] "
        self.printer = printer
        self.warn = warner

    def start(self, message=None, data=None, display=False):
        self.benchmark_start = self.lap_start = time.time()
        self.times = [{"time": self.benchmark_start, "message": message, "data": data}]
        if display:
            self.printer(self.prefix + "Starting benchmark at " + str(self.benchmark_start) + (": " + message if message else ""))

    def benchmark(self, message=None, data=None):
        benchmark_time = time.time()
        if self.benchmark_start == None:
            self.warn("No benchmark currently running"+(" for benchmarker " + name))
        else:
            lap = benchmark_time - self.lap_start
            self.lap_start = benchmark_time
            self.times.append({"time": benchmark_time, "message": message, "data": data, "lap": lap})
            self.printer(self.prefix + (message or "Lap time") + ": " + str(lap))

    def end(self, message=None, data=None, benchmark=True, display=False):
        if self.benchmark_start == None:
            self.warn("No benchmark currently running"+(" for benchmarker " + name))
        else:
            if benchmark:
                self.benchmark(message, data)
            elif display:
                benchmark_time = time.time()
                lap = benchmark_time - self.lap_start
                self.times.append({"time": benchmark_time, "message": message, "data": data, "lap": lap})
            if display:
                self.printer(self.prefix + "Ended; total time was " + str(self.times[-1]["time"]-self.benchmark_start) + (": " + message if message else ""))

    def __str__(self):
        return json.dumps(self.times)
