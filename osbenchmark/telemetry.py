# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
# Licensed to Elasticsearch B.V. under one or more contributor
# license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Elasticsearch B.V. licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import collections
import logging
import fnmatch
import os
import threading

import tabulate

from osbenchmark import metrics, time, exceptions
from osbenchmark.metrics import MetaInfoScope
from osbenchmark.utils import io, sysstats, console, opts, process
from osbenchmark.utils.versions import components


def list_telemetry():
    console.println("Available telemetry devices:\n")
    devices = [[device.command, device.human_name, device.help] for device in [JitCompiler, Gc, FlightRecorder,
                                                                               Heapdump, RecoveryStats,
                                                                               CcrStats, SegmentStats, TransformStats,
                                                                               SearchableSnapshotsStats]]
    console.println(tabulate.tabulate(devices, ["Command", "Name", "Description"]))
    console.println("\nKeep in mind that each telemetry device may incur a runtime overhead which can skew results.")


class Telemetry:
    def __init__(self, enabled_devices=None, devices=None):
        if devices is None:
            devices = []
        if enabled_devices is None:
            enabled_devices = []
        self.enabled_devices = enabled_devices
        self.devices = devices

    def instrument_candidate_java_opts(self):
        opts = []
        for device in self.devices:
            if self._enabled(device):
                additional_opts = device.instrument_java_opts()
                # properly merge values with the same key
                opts.extend(additional_opts)
        return opts

    def on_pre_node_start(self, node_name):
        for device in self.devices:
            if self._enabled(device):
                device.on_pre_node_start(node_name)

    def attach_to_node(self, node):
        for device in self.devices:
            if self._enabled(device):
                device.attach_to_node(node)

    def detach_from_node(self, node, running):
        for device in self.devices:
            if self._enabled(device):
                device.detach_from_node(node, running)

    def on_benchmark_start(self):
        for device in self.devices:
            if self._enabled(device):
                device.on_benchmark_start()

    def on_benchmark_stop(self):
        for device in self.devices:
            if self._enabled(device):
                device.on_benchmark_stop()

    def store_system_metrics(self, node, metrics_store):
        for device in self.devices:
            if self._enabled(device):
                device.store_system_metrics(node, metrics_store)

    def _enabled(self, device):
        return device.internal or device.command in self.enabled_devices


########################################################################################
#
# Telemetry devices
#
########################################################################################

class TelemetryDevice:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def instrument_java_opts(self):
        return {}

    def on_pre_node_start(self, node_name):
        pass

    def attach_to_node(self, node):
        pass

    def detach_from_node(self, node, running):
        pass

    def on_benchmark_start(self):
        pass

    def on_benchmark_stop(self):
        pass

    def store_system_metrics(self, node, metrics_store):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = logging.getLogger(__name__)


class InternalTelemetryDevice(TelemetryDevice):
    internal = True


class SamplerThread(threading.Thread):
    def __init__(self, recorder):
        threading.Thread.__init__(self)
        self.stop = False
        self.recorder = recorder

    def finish(self):
        self.stop = True
        self.join()

    def run(self):
        # noinspection PyBroadException
        try:
            while not self.stop:
                self.recorder.record()
                time.sleep(self.recorder.sample_interval)
        except BaseException:
            logging.getLogger(__name__).exception("Could not determine %s", self.recorder)


class FlightRecorder(TelemetryDevice):
    internal = False
    command = "jfr"
    human_name = "Flight Recorder"
    help = "Enables Java Flight Recorder (requires an Oracle JDK or OpenJDK 11+)"

    def __init__(self, telemetry_params, log_root, java_major_version):
        super().__init__()
        self.telemetry_params = telemetry_params
        self.log_root = log_root
        self.java_major_version = java_major_version

    def instrument_java_opts(self):
        io.ensure_dir(self.log_root)
        log_file = os.path.join(self.log_root, "profile.jfr")

        # JFR was integrated into OpenJDK 11 and is not a commercial feature anymore.
        if self.java_major_version < 11:
            console.println("\n***************************************************************************\n")
            console.println("[WARNING] Java flight recorder is a commercial feature of the Oracle JDK.\n")
            console.println("You are using Java flight recorder which requires that you comply with\nthe licensing terms stated in:\n")
            console.println(console.format.link("http://www.oracle.com/technetwork/java/javase/terms/license/index.html"))
            console.println("\nBy using this feature you confirm that you comply with these license terms.\n")
            console.println("Otherwise, please abort and rerun Benchmark without the \"jfr\" telemetry device.")
            console.println("\n***************************************************************************\n")

            time.sleep(3)

        console.info("%s: Writing flight recording to [%s]" % (self.human_name, log_file), logger=self.logger)

        java_opts = self.java_opts(log_file)

        self.logger.info("jfr: Adding JVM arguments: [%s].", java_opts)
        return java_opts

    def java_opts(self, log_file):
        recording_template = self.telemetry_params.get("recording-template")
        java_opts = ["-XX:+UnlockDiagnosticVMOptions", "-XX:+DebugNonSafepoints"]
        jfr_cmd = ""
        if self.java_major_version < 11:
            java_opts.append("-XX:+UnlockCommercialFeatures")

        if self.java_major_version < 9:
            java_opts.append("-XX:+FlightRecorder")
            java_opts.append("-XX:FlightRecorderOptions=disk='true',maxage=0s,maxsize=0,dumponexit='true',dumponexitpath={}".format(log_file))
            jfr_cmd = "-XX:StartFlightRecording=defaultrecording='true'"
            if recording_template:
                self.logger.info("jfr: Using recording template [%s].", recording_template)
                jfr_cmd += ",settings={}".format(recording_template)
            else:
                self.logger.info("jfr: Using default recording template.")
        else:
            jfr_cmd += "-XX:StartFlightRecording=maxsize=0,maxage=0s,disk='true',dumponexit='true',filename={}".format(log_file)
            if recording_template:
                self.logger.info("jfr: Using recording template [%s].", recording_template)
                jfr_cmd += ",settings={}".format(recording_template)
            else:
                self.logger.info("jfr: Using default recording template.")
        java_opts.append(jfr_cmd)
        return java_opts


class JitCompiler(TelemetryDevice):
    internal = False
    command = "jit"
    human_name = "JIT Compiler Profiler"
    help = "Enables JIT compiler logs."

    def __init__(self, log_root):
        super().__init__()
        self.log_root = log_root

    def instrument_java_opts(self):
        io.ensure_dir(self.log_root)
        log_file = os.path.join(self.log_root, "jit.log")
        console.info("%s: Writing JIT compiler log to [%s]" % (self.human_name, log_file), logger=self.logger)
        return ["-XX:+UnlockDiagnosticVMOptions", "-XX:+TraceClassLoading", "-XX:+LogCompilation",
                "-XX:LogFile={}".format(log_file), "-XX:+PrintAssembly"]


class Gc(TelemetryDevice):
    internal = False
    command = "gc"
    human_name = "GC log"
    help = "Enables GC logs."

    def __init__(self, telemetry_params, log_root, java_major_version):
        super().__init__()
        self.telemetry_params = telemetry_params
        self.log_root = log_root
        self.java_major_version = java_major_version

    def instrument_java_opts(self):
        io.ensure_dir(self.log_root)
        log_file = os.path.join(self.log_root, "gc.log")
        console.info("%s: Writing GC log to [%s]" % (self.human_name, log_file), logger=self.logger)
        return self.java_opts(log_file)

    def java_opts(self, log_file):
        if self.java_major_version < 9:
            return ["-Xloggc:{}".format(log_file), "-XX:+PrintGCDetails", "-XX:+PrintGCDateStamps", "-XX:+PrintGCTimeStamps",
                    "-XX:+PrintGCApplicationStoppedTime", "-XX:+PrintGCApplicationConcurrentTime",
                    "-XX:+PrintTenuringDistribution"]
        else:
            log_config = self.telemetry_params.get("gc-log-config", "gc*=info,safepoint=info,age*=trace")
            # see https://docs.oracle.com/javase/9/tools/java.htm#JSWOR-GUID-BE93ABDC-999C-4CB5-A88B-1994AAAC74D5
            return [f"-Xlog:{log_config}:file={log_file}:utctime,uptimemillis,level,tags:filecount=0"]


class Heapdump(TelemetryDevice):
    internal = False
    command = "heapdump"
    human_name = "Heap Dump"
    help = "Captures a heap dump."

    def __init__(self, log_root):
        super().__init__()
        self.log_root = log_root

    def detach_from_node(self, node, running):
        if running:
            heap_dump_file = os.path.join(self.log_root, "heap_at_exit_{}.hprof".format(node.pid))
            console.info("{}: Writing heap dump to [{}]".format(self.human_name, heap_dump_file), logger=self.logger)
            cmd = "jmap -dump:format=b,file={} {}".format(heap_dump_file, node.pid)
            if process.run_subprocess_with_logging(cmd):
                self.logger.warning("Could not write heap dump to [%s]", heap_dump_file)


class SegmentStats(TelemetryDevice):
    internal = False
    command = "segment-stats"
    human_name = "Segment Stats"
    help = "Determines segment stats at the end of the benchmark."

    def __init__(self, log_root, client):
        super().__init__()
        self.log_root = log_root
        self.client = client

    def on_benchmark_stop(self):
        # noinspection PyBroadException
        try:
            segment_stats = self.client.cat.segments(index="_all", v=True)
            stats_file = os.path.join(self.log_root, "segment_stats.log")
            console.info(f"{self.human_name}: Writing segment stats to [{stats_file}]", logger=self.logger)
            with open(stats_file, "wt") as f:
                f.write(segment_stats)
        except BaseException:
            self.logger.exception("Could not retrieve segment stats.")


class CcrStats(TelemetryDevice):
    internal = False
    command = "ccr-stats"
    human_name = "CCR Stats"
    help = "Regularly samples Cross Cluster Replication (CCR) related stats"

    """
    Gathers CCR stats on a cluster level
    """

    def __init__(self, telemetry_params, clients, metrics_store):
        """
        :param telemetry_params: The configuration object for telemetry_params.
            May optionally specify:
            ``ccr-stats-indices``: JSON string specifying the indices per cluster to publish statistics from.
            Not all clusters need to be specified, but any name used must be be present in target.hosts.
            Example:
            {"ccr-stats-indices": {"cluster_a": ["follower"],"default": ["leader"]}
            ``ccr-stats-sample-interval``: positive integer controlling the sampling interval. Default: 1 second.
        :param clients: A dict of clients to all clusters.
        :param metrics_store: The configured metrics store we write to.
        """
        super().__init__()

        self.telemetry_params = telemetry_params
        self.clients = clients
        self.sample_interval = telemetry_params.get("ccr-stats-sample-interval", 1)
        if self.sample_interval <= 0:
            raise exceptions.SystemSetupError(
                "The telemetry parameter 'ccr-stats-sample-interval' must be greater than zero but was {}.".format(self.sample_interval))
        self.specified_cluster_names = self.clients.keys()
        self.indices_per_cluster = self.telemetry_params.get("ccr-stats-indices", False)
        if self.indices_per_cluster:
            for cluster_name in self.indices_per_cluster.keys():
                if cluster_name not in clients:
                    raise exceptions.SystemSetupError(
                        "The telemetry parameter 'ccr-stats-indices' must be a JSON Object with keys matching "
                        "the cluster names [{}] specified in --target-hosts "
                        "but it had [{}].".format(",".join(sorted(clients.keys())), cluster_name))
            self.specified_cluster_names = self.indices_per_cluster.keys()

        self.metrics_store = metrics_store
        self.samplers = []

    def on_benchmark_start(self):
        recorder = []
        for cluster_name in self.specified_cluster_names:
            recorder = CcrStatsRecorder(cluster_name, self.clients[cluster_name], self.metrics_store, self.sample_interval,
                                        self.indices_per_cluster[cluster_name] if self.indices_per_cluster else None)
            sampler = SamplerThread(recorder)
            self.samplers.append(sampler)
            sampler.setDaemon(True)
            # we don't require starting recorders precisely at the same time
            sampler.start()

    def on_benchmark_stop(self):
        if self.samplers:
            for sampler in self.samplers:
                sampler.finish()


class CcrStatsRecorder:
    """
    Collects and pushes CCR stats for the specified cluster to the metric store.
    """

    def __init__(self, cluster_name, client, metrics_store, sample_interval, indices=None):
        """
        :param cluster_name: The cluster_name that the client connects to, as specified in target.hosts.
        :param client: The OpenSearch client for this cluster.
        :param metrics_store: The configured metrics store we write to.
        :param sample_interval: integer controlling the interval, in seconds, between collecting samples.
        :param indices: optional list of indices to filter results from.
        """

        self.cluster_name = cluster_name
        self.client = client
        self.metrics_store = metrics_store
        self.sample_interval= sample_interval
        self.indices = indices
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return "ccr stats"

    def record(self):
        """
        Collect CCR stats for indexes (optionally) specified in telemetry parameters and push to metrics store.
        """

        # ES returns all stats values in bytes or ms via "human: 'false'"

        # pylint: disable=import-outside-toplevel
        import elasticsearch

        try:
            ccr_stats_api_endpoint = "/_ccr/stats"
            filter_path = "follow_stats"
            stats = self.client.transport.perform_request("GET", ccr_stats_api_endpoint, params={"human": "'false'",
                                                                                                 "filter_path": filter_path})
        except elasticsearch.TransportError:
            msg = "A transport error occurred while collecting CCR stats from the endpoint [{}?filter_path={}] on " \
                  "cluster [{}]".format(ccr_stats_api_endpoint, filter_path, self.cluster_name)
            self.logger.exception(msg)
            raise exceptions.BenchmarkError(msg)

        if filter_path in stats and "indices" in stats[filter_path]:
            for indices in stats[filter_path]["indices"]:
                try:
                    if self.indices and indices["index"] not in self.indices:
                        # Skip metrics for indices not part of user supplied whitelist (ccr-stats-indices) in telemetry params.
                        continue
                    self.record_stats_per_index(indices["index"], indices["shards"])
                except KeyError:
                    self.logger.warning(
                        "The 'indices' key in %s does not contain an 'index' or 'shards' key "
                        "Maybe the output format of the %s endpoint has changed. Skipping.", ccr_stats_api_endpoint, ccr_stats_api_endpoint
                    )

    def record_stats_per_index(self, name, stats):
        """
        :param name: The index name.
        :param stats: A dict with returned CCR stats for the index.
        """

        for shard_stats in stats:
            if "shard_id" in shard_stats:
                doc = {
                    "name": "ccr-stats",
                    "shard": shard_stats
                }
                shard_metadata = {
                    "cluster": self.cluster_name,
                    "index": name
                }

                self.metrics_store.put_doc(doc, level=MetaInfoScope.cluster, meta_data=shard_metadata)


class RecoveryStats(TelemetryDevice):
    internal = False
    command = "recovery-stats"
    human_name = "Recovery Stats"
    help = "Regularly samples shard recovery stats"

    """
    Gathers recovery stats on a cluster level
    """

    def __init__(self, telemetry_params, clients, metrics_store):
        """
        :param telemetry_params: The configuration object for telemetry_params.
            May optionally specify:
            ``recovery-stats-indices``: JSON structure specifying the index pattern per cluster to publish stats from.
            Not all clusters need to be specified, but any name used must be be present in target.hosts. Alternatively,
            the index pattern can be specified as a string can be specified in case only one cluster is involved.
            Example:
            {"recovery-stats-indices": {"cluster_a": ["follower"],"default": ["leader"]}

            ``recovery-stats-sample-interval``: positive integer controlling the sampling interval. Default: 1 second.
        :param clients: A dict of clients to all clusters.
        :param metrics_store: The configured metrics store we write to.
        """
        super().__init__()

        self.telemetry_params = telemetry_params
        self.clients = clients
        self.sample_interval = telemetry_params.get("recovery-stats-sample-interval", 1)
        if self.sample_interval <= 0:
            raise exceptions.SystemSetupError(
                "The telemetry parameter 'recovery-stats-sample-interval' must be greater than zero but was {}."
                    .format(self.sample_interval))
        self.specified_cluster_names = self.clients.keys()
        indices_per_cluster = self.telemetry_params.get("recovery-stats-indices", False)
        # allow the user to specify either an index pattern as string or as a JSON object
        if isinstance(indices_per_cluster, str):
            self.indices_per_cluster = {opts.TargetHosts.DEFAULT: indices_per_cluster}
        else:
            self.indices_per_cluster = indices_per_cluster

        if self.indices_per_cluster:
            for cluster_name in self.indices_per_cluster.keys():
                if cluster_name not in clients:
                    raise exceptions.SystemSetupError(
                        "The telemetry parameter 'recovery-stats-indices' must be a JSON Object with keys matching "
                        "the cluster names [{}] specified in --target-hosts "
                        "but it had [{}].".format(",".join(sorted(clients.keys())), cluster_name))
            self.specified_cluster_names = self.indices_per_cluster.keys()

        self.metrics_store = metrics_store
        self.samplers = []

    def on_benchmark_start(self):
        for cluster_name in self.specified_cluster_names:
            recorder = RecoveryStatsRecorder(cluster_name, self.clients[cluster_name], self.metrics_store,
                                             self.sample_interval,
                                             self.indices_per_cluster[cluster_name] if self.indices_per_cluster else "")
            sampler = SamplerThread(recorder)
            self.samplers.append(sampler)
            sampler.setDaemon(True)
            # we don't require starting recorders precisely at the same time
            sampler.start()

    def on_benchmark_stop(self):
        if self.samplers:
            for sampler in self.samplers:
                sampler.finish()


class RecoveryStatsRecorder:
    """
    Collects and pushes recovery stats for the specified cluster to the metric store.
    """

    def __init__(self, cluster_name, client, metrics_store, sample_interval, indices=None):
        """
        :param cluster_name: The cluster_name that the client connects to, as specified in target.hosts.
        :param client: The OpenSearch client for this cluster.
        :param metrics_store: The configured metrics store we write to.
        :param sample_interval: integer controlling the interval, in seconds, between collecting samples.
        :param indices: optional list of indices to filter results from.
        """

        self.cluster_name = cluster_name
        self.client = client
        self.metrics_store = metrics_store
        self.sample_interval = sample_interval
        self.indices = indices
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return "recovery stats"

    def record(self):
        """
        Collect recovery stats for indexes (optionally) specified in telemetry parameters and push to metrics store.
        """
        # pylint: disable=import-outside-toplevel
        import elasticsearch

        try:
            stats = self.client.indices.recovery(index=self.indices, active_only=True, detailed=False)
        except elasticsearch.TransportError:
            msg = "A transport error occurred while collecting recovery stats on cluster [{}]".format(self.cluster_name)
            self.logger.exception(msg)
            raise exceptions.BenchmarkError(msg)

        for idx, idx_stats in stats.items():
            for shard in idx_stats["shards"]:
                doc = {
                    "name": "recovery-stats",
                    "shard": shard
                }
                shard_metadata = {
                    "cluster": self.cluster_name,
                    "index": idx,
                    "shard": shard["id"]
                }
                self.metrics_store.put_doc(doc, level=MetaInfoScope.cluster, meta_data=shard_metadata)


class NodeStats(TelemetryDevice):
    """
    Gathers different node stats.
    """

    internal = False
    command = "node-stats"
    human_name = "Node Stats"
    help = "Regularly samples node stats"
    warning = """You have enabled the node-stats telemetry device with OpenSearch < 1.1.0. Requests to the
          _nodes/stats OpenSearch endpoint trigger additional refreshes and WILL SKEW results.
    """

    def __init__(self, telemetry_params, clients, metrics_store):
        super().__init__()
        self.telemetry_params = telemetry_params
        self.clients = clients
        self.specified_cluster_names = self.clients.keys()
        self.metrics_store = metrics_store
        self.samplers = []

    def on_benchmark_start(self):
        default_client = self.clients["default"]
        distribution_version = "7.7.0"
        major, minor = components(distribution_version)[:2]

        if major < 7 or (major == 7 and minor < 2):
            console.warn(NodeStats.warning, logger=self.logger)

        for cluster_name in self.specified_cluster_names:
            recorder = NodeStatsRecorder(self.telemetry_params, cluster_name, self.clients[cluster_name], self.metrics_store)
            sampler = SamplerThread(recorder)
            self.samplers.append(sampler)
            sampler.setDaemon(True)
            # we don't require starting recorders precisely at the same time
            sampler.start()

    def on_benchmark_stop(self):
        if self.samplers:
            for sampler in self.samplers:
                sampler.finish()


class NodeStatsRecorder:
    def __init__(self, telemetry_params, cluster_name, client, metrics_store):
        self.sample_interval = telemetry_params.get("node-stats-sample-interval", 1)
        if self.sample_interval <= 0:
            raise exceptions.SystemSetupError(
                "The telemetry parameter 'node-stats-sample-interval' must be greater than zero but was {}.".format(self.sample_interval))

        self.include_indices = telemetry_params.get("node-stats-include-indices", False)
        self.include_indices_metrics = telemetry_params.get("node-stats-include-indices-metrics", False)

        if self.include_indices_metrics:
            if isinstance(self.include_indices_metrics, str):
                self.include_indices_metrics_list = opts.csv_to_list(self.include_indices_metrics)
            else:
                # we don't validate the allowable metrics as they may change across ES versions
                raise exceptions.SystemSetupError(
                    "The telemetry parameter 'node-stats-include-indices-metrics' must be a comma-separated string but was {}".format(
                        type(self.include_indices_metrics))
                    )
        else:
            self.include_indices_metrics_list = ["docs", "store", "indexing", "search", "merges", "query_cache",
                                                 "fielddata", "segments", "translog", "request_cache"]

        self.include_thread_pools = telemetry_params.get("node-stats-include-thread-pools", True)
        self.include_buffer_pools = telemetry_params.get("node-stats-include-buffer-pools", True)
        self.include_breakers = telemetry_params.get("node-stats-include-breakers", True)
        self.include_network = telemetry_params.get("node-stats-include-network", True)
        self.include_process = telemetry_params.get("node-stats-include-process", True)
        self.include_mem_stats = telemetry_params.get("node-stats-include-mem", True)
        self.include_gc_stats = telemetry_params.get("node-stats-include-gc", True)
        self.include_indexing_pressure = telemetry_params.get("node-stats-include-indexing-pressure", True)
        self.client = client
        self.metrics_store = metrics_store
        self.cluster_name = cluster_name

    def __str__(self):
        return "node stats"

    def record(self):
        current_sample = self.sample()
        for node_stats in current_sample:
            node_name = node_stats["name"]
            metrics_store_meta_data = {
                "cluster": self.cluster_name,
                "node_name": node_name
            }
            collected_node_stats = collections.OrderedDict()
            collected_node_stats["name"] = "node-stats"

            if self.include_indices or self.include_indices_metrics:
                collected_node_stats.update(
                    self.indices_stats(node_name, node_stats, include=self.include_indices_metrics_list))
            if self.include_thread_pools:
                collected_node_stats.update(self.thread_pool_stats(node_name, node_stats))
            if self.include_breakers:
                collected_node_stats.update(self.circuit_breaker_stats(node_name, node_stats))
            if self.include_buffer_pools:
                collected_node_stats.update(self.jvm_buffer_pool_stats(node_name, node_stats))
            if self.include_mem_stats:
                collected_node_stats.update(self.jvm_mem_stats(node_name, node_stats))
            if self.include_gc_stats:
                collected_node_stats.update(self.jvm_gc_stats(node_name, node_stats))
            if self.include_network:
                collected_node_stats.update(self.network_stats(node_name, node_stats))
            if self.include_process:
                collected_node_stats.update(self.process_stats(node_name, node_stats))
            if self.include_indexing_pressure:
                collected_node_stats.update(self.indexing_pressure(node_name, node_stats))

            self.metrics_store.put_doc(dict(collected_node_stats),
                                       level=MetaInfoScope.node,
                                       node_name=node_name,
                                       meta_data=metrics_store_meta_data)

    def flatten_stats_fields(self, prefix=None, stats=None):
        """
        Flatten provided dict using an optional prefix and top level key filters.

        :param prefix: The prefix for all flattened values. Defaults to None.
        :param stats: Dict with values to be flattened, using _ as a separator. Defaults to {}.
        :return: Return flattened dictionary, separated by _ and prefixed with prefix.
        """

        def iterate():
            for section_name, section_value in stats.items():
                if isinstance(section_value, dict):
                    new_prefix = "{}_{}".format(prefix, section_name)
                    # https://www.python.org/dev/peps/pep-0380/
                    yield from self.flatten_stats_fields(prefix=new_prefix, stats=section_value).items()
                # Avoid duplication for metric fields that have unit embedded in value as they are also recorded elsewhere
                # example: `breakers_parent_limit_size_in_bytes` vs `breakers_parent_limit_size`
                elif isinstance(section_value, (int, float)) and not isinstance(section_value, bool):
                    yield "{}{}".format(prefix + "_" if prefix else "", section_name), section_value

        if stats:
            return dict(iterate())
        else:
            return dict()

    def indices_stats(self, node_name, node_stats, include):
        idx_stats = node_stats["indices"]
        ordered_results = collections.OrderedDict()
        for section in include:
            if section in idx_stats:
                ordered_results.update(self.flatten_stats_fields(prefix="indices_" + section, stats=idx_stats[section]))

        return ordered_results

    def thread_pool_stats(self, node_name, node_stats):
        return self.flatten_stats_fields(prefix="thread_pool", stats=node_stats["thread_pool"])

    def circuit_breaker_stats(self, node_name, node_stats):
        return self.flatten_stats_fields(prefix="breakers", stats=node_stats["breakers"])

    def jvm_buffer_pool_stats(self, node_name, node_stats):
        return self.flatten_stats_fields(prefix="jvm_buffer_pools", stats=node_stats["jvm"]["buffer_pools"])

    def jvm_mem_stats(self, node_name, node_stats):
        return self.flatten_stats_fields(prefix="jvm_mem", stats=node_stats["jvm"]["mem"])

    def jvm_gc_stats(self, node_name, node_stats):
        return self.flatten_stats_fields(prefix="jvm_gc", stats=node_stats["jvm"]["gc"])

    def network_stats(self, node_name, node_stats):
        return self.flatten_stats_fields(prefix="transport", stats=node_stats.get("transport"))

    def process_stats(self, node_name, node_stats):
        return self.flatten_stats_fields(prefix="process_cpu", stats=node_stats["process"]["cpu"])

    def indexing_pressure(self, node_name, node_stats):
        return self.flatten_stats_fields(prefix="indexing_pressure", stats=node_stats["indexing_pressure"])

    def sample(self):
        # pylint: disable=import-outside-toplevel
        import elasticsearch
        try:
            stats = {
                "_nodes" : {
                    "total" : 3,
                    "successful" : 3,
                    "failed" : 0
                },
                "cluster_name" : "397869430111:juno-mcm",
                "nodes" : {
                    "3g-J2dUQS_m2su_yvnqRUA" : {
                        "timestamp" : 1639459994258,
                        "name" : "29f760e1a8adb54ed33d8eab83248fa0",
                        "roles" : [ "data", "ingest", "master", "remote_cluster_client" ],
                        "indices" : {
                            "docs" : {
                                "count" : 404,
                                "deleted" : 3
                            },
                            "store" : {
                                "size_in_bytes" : 185622
                            },
                            "indexing" : {
                                "index_total" : 3184,
                                "index_time_in_millis" : 2130,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 18,
                                "time_in_millis" : 69,
                                "exists_total" : 18,
                                "exists_time_in_millis" : 69,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 67,
                                "query_time_in_millis" : 34,
                                "query_current" : 0,
                                "fetch_total" : 1,
                                "fetch_time_in_millis" : 2,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 73,
                                "total_docs" : 10,
                                "total_size_in_bytes" : 53955,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 377487360
                            },
                            "refresh" : {
                                "total" : 106,
                                "total_time_in_millis" : 441,
                                "external_total" : 87,
                                "external_total_time_in_millis" : 418,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 16,
                                "periodic" : 0,
                                "total_time_in_millis" : 16
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 20,
                                "total_time_in_millis" : 4
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 5,
                                "memory_in_bytes" : 27524,
                                "terms_memory_in_bytes" : 15920,
                                "stored_fields_memory_in_bytes" : 2504,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 2112,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 6988,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 0,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 220,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 220,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 1078,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 1
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "os" : {
                            "timestamp" : 1639459994258,
                            "cpu" : {
                                "percent" : 3,
                                "load_average" : {
                                    "1m" : 0.04,
                                    "5m" : 0.07,
                                    "15m" : 0.11
                                }
                            },
                            "mem" : {
                                "total_in_bytes" : 16480382976,
                                "free_in_bytes" : 412884992,
                                "used_in_bytes" : 16067497984,
                                "free_percent" : 3,
                                "used_percent" : 97
                            },
                            "swap" : {
                                "total_in_bytes" : 2147479552,
                                "free_in_bytes" : 2146955264,
                                "used_in_bytes" : 524288
                            },
                            "cgroup" : {
                                "cpuacct" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "usage_nanos" : 27963478780746
                                },
                                "cpu" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "cfs_period_micros" : 100000,
                                    "cfs_quota_micros" : -1,
                                    "stat" : {
                                        "number_of_elapsed_periods" : 0,
                                        "number_of_times_throttled" : 0,
                                        "time_throttled_nanos" : 0
                                    }
                                },
                                "memory" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "limit_in_bytes" : "9223372036854771712",
                                    "usage_in_bytes" : "11649228800"
                                }
                            }
                        },
                        "process" : {
                            "timestamp" : 1639459994259,
                            "open_file_descriptors" : 1724,
                            "max_file_descriptors" : 1024000,
                            "cpu" : {
                                "percent" : 0,
                                "total_in_millis" : 6614350
                            },
                            "mem" : {
                                "total_virtual_in_bytes" : 12349054976
                            }
                        },
                        "jvm" : {
                            "timestamp" : 1639459994260,
                            "uptime_in_millis" : 476870141,
                            "mem" : {
                                "heap_used_in_bytes" : 894324928,
                                "heap_used_percent" : 10,
                                "heap_committed_in_bytes" : 8572502016,
                                "heap_max_in_bytes" : 8572502016,
                                "non_heap_used_in_bytes" : 333200872,
                                "non_heap_committed_in_bytes" : 395907072,
                                "pools" : {
                                    "young" : {
                                        "used_in_bytes" : 52322208,
                                        "max_in_bytes" : 139591680,
                                        "peak_used_in_bytes" : 139591680,
                                        "peak_max_in_bytes" : 139591680
                                    },
                                    "survivor" : {
                                        "used_in_bytes" : 2067384,
                                        "max_in_bytes" : 17432576,
                                        "peak_used_in_bytes" : 17432576,
                                        "peak_max_in_bytes" : 17432576
                                    },
                                    "old" : {
                                        "used_in_bytes" : 839935336,
                                        "max_in_bytes" : 8415477760,
                                        "peak_used_in_bytes" : 839935336,
                                        "peak_max_in_bytes" : 8415477760
                                    }
                                }
                            },
                            "threads" : {
                                "count" : 145,
                                "peak_count" : 145
                            },
                            "gc" : {
                                "collectors" : {
                                    "young" : {
                                        "collection_count" : 15161,
                                        "collection_time_in_millis" : 816018
                                    },
                                    "old" : {
                                        "collection_count" : 4,
                                        "collection_time_in_millis" : 332
                                    }
                                }
                            },
                            "buffer_pools" : {
                                "mapped" : {
                                    "count" : 11,
                                    "used_in_bytes" : 162531,
                                    "total_capacity_in_bytes" : 162531
                                },
                                "direct" : {
                                    "count" : 124,
                                    "used_in_bytes" : 5763821,
                                    "total_capacity_in_bytes" : 5763819
                                }
                            },
                            "classes" : {
                                "current_loaded_count" : 37833,
                                "total_loaded_count" : 37837,
                                "total_unloaded_count" : 4
                            }
                        },
                        "thread_pool" : {
                            "ad-threadpool" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "analyze" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_started" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_store" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "flush" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 16
                            },
                            "force_merge" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 8
                            },
                            "generic" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 15,
                                "completed" : 127991
                            },
                            "get" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 10
                            },
                            "listener" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "management" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 1,
                                "rejected" : 0,
                                "largest" : 5,
                                "completed" : 545500
                            },
                            "open_distro_job_scheduler" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "refresh" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1424007
                            },
                            "search" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 117
                            },
                            "search_throttled" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "snapshot" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1
                            },
                            "snapshot_segments" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 28
                            },
                            "snapshot_shards" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 262
                            },
                            "sql-worker" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "ultrawarm_migration" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "warmer" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "write" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 63
                            }
                        },
                        "fs" : {
                            "timestamp" : 1639459994260,
                            "total" : {
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528403456,
                                "available_in_bytes" : 10511626240
                            },
                            "data" : [ {
                                "type" : "ext4",
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528403456,
                                "available_in_bytes" : 10511626240
                            } ],
                            "io_stats" : {
                                "devices" : [ {
                                    "device_name" : "dm-0",
                                    "operations" : 46888,
                                    "read_operations" : 1,
                                    "write_operations" : 46887,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 189620
                                } ],
                                "total" : {
                                    "operations" : 46888,
                                    "read_operations" : 1,
                                    "write_operations" : 46887,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 189620
                                }
                            }
                        },
                        "http" : {
                            "current_open" : 0,
                            "total_opened" : 0
                        },
                        "breakers" : {
                            "request" : {
                                "limit_size_in_bytes" : 5143501209,
                                "limit_size" : "4.7gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "fielddata" : {
                                "limit_size_in_bytes" : 3429000806,
                                "limit_size" : "3.1gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.03,
                                "tripped" : 0
                            },
                            "in_flight_requests" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 1416,
                                "estimated_size" : "1.3kb",
                                "overhead" : 2.0,
                                "tripped" : 0
                            },
                            "accounting" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 27524,
                                "estimated_size" : "26.8kb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "parent" : {
                                "limit_size_in_bytes" : 8143876915,
                                "limit_size" : "7.5gb",
                                "estimated_size_in_bytes" : 894393656,
                                "estimated_size" : "852.9mb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            }
                        },
                        "script" : {
                            "compilations" : 0,
                            "cache_evictions" : 0,
                            "compilation_limit_triggered" : 0
                        },
                        "discovery" : {
                            "cluster_state_queue" : {
                                "total" : 0,
                                "pending" : 0,
                                "committed" : 0
                            },
                            "published_cluster_states" : {
                                "full_states" : 1,
                                "incompatible_diffs" : 0,
                                "compatible_diffs" : 1654
                            }
                        },
                        "ingest" : {
                            "total" : {
                                "count" : 0,
                                "time_in_millis" : 0,
                                "current" : 0,
                                "failed" : 0
                            },
                            "pipelines" : { }
                        },
                        "adaptive_selection" : {
                            "8dqNgo-KRYGIzJ2xsgbfEw" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1200061,
                                "avg_response_time_ns" : 2868875,
                                "rank" : "2.9"
                            },
                            "3g-J2dUQS_m2su_yvnqRUA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 539218,
                                "avg_response_time_ns" : 2301522,
                                "rank" : "2.3"
                            },
                            "Gh_8fOzBTmaNnbQmAyd-rA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1194916,
                                "avg_response_time_ns" : 2677770,
                                "rank" : "2.7"
                            }
                        },
                        "script_cache" : {
                            "sum" : {
                                "compilations" : 0,
                                "cache_evictions" : 0,
                                "compilation_limit_triggered" : 0
                            }
                        },
                        "indexing_pressure" : {
                            "memory" : {
                                "current" : {
                                    "combined_coordinating_and_primary_in_bytes" : 0,
                                    "coordinating_in_bytes" : 0,
                                    "primary_in_bytes" : 0,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 0
                                },
                                "total" : {
                                    "combined_coordinating_and_primary_in_bytes" : 2515084,
                                    "coordinating_in_bytes" : 1517909,
                                    "primary_in_bytes" : 1583647,
                                    "replica_in_bytes" : 54564,
                                    "all_in_bytes" : 2569648,
                                    "coordinating_rejections" : 0,
                                    "primary_rejections" : 0,
                                    "replica_rejections" : 0
                                }
                            }
                        },
                        "shard_indexing_pressure" : {
                            "stats" : { },
                            "total_rejections_breakup_shadow_mode" : {
                                "node_limits" : 0,
                                "no_successful_request_limits" : 0,
                                "throughput_degradation_limits" : 0
                            },
                            "enabled" : 'true',
                            "enforced" : 'false'
                        }
                    },
                    "Gh_8fOzBTmaNnbQmAyd-rA" : {
                        "timestamp" : 1639459994257,
                        "name" : "84c74d57fefcca221479642d7916c1f2",
                        "roles" : [ "data", "ingest", "master", "remote_cluster_client" ],
                        "indices" : {
                            "docs" : {
                                "count" : 429,
                                "deleted" : 3
                            },
                            "store" : {
                                "size_in_bytes" : 199274
                            },
                            "indexing" : {
                                "index_total" : 3203,
                                "index_time_in_millis" : 2154,
                            "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 27,
                                "time_in_millis" : 150,
                                "exists_total" : 27,
                                "exists_time_in_millis" : 150,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 68,
                                "query_time_in_millis" : 68,
                                "query_current" : 0,
                                "fetch_total" : 68,
                                "fetch_time_in_millis" : 93,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 65,
                                "total_docs" : 10,
                                "total_size_in_bytes" : 54027,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 377487360
                            },
                            "refresh" : {
                                "total" : 113,
                                "total_time_in_millis" : 471,
                                "external_total" : 89,
                                "external_total_time_in_millis" : 468,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 17,
                                "periodic" : 0,
                                "total_time_in_millis" : 78
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 23,
                                "total_time_in_millis" : 15
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 6,
                                "memory_in_bytes" : 29344,
                                "terms_memory_in_bytes" : 16704,
                                "stored_fields_memory_in_bytes" : 2992,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 2112,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 7536,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 48,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 220,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 220,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "os" : {
                            "timestamp" : 1639459994259,
                            "cpu" : {
                                "percent" : 3,
                                "load_average" : {
                                    "1m" : 0.2,
                                    "5m" : 0.19,
                                    "15m" : 0.16
                                }
                            },
                            "mem" : {
                                "total_in_bytes" : 16480382976,
                                "free_in_bytes" : 470929408,
                                "used_in_bytes" : 16009453568,
                                "free_percent" : 3,
                                "used_percent" : 97
                            },
                            "swap" : {
                                "total_in_bytes" : 2147479552,
                                "free_in_bytes" : 2146955264,
                                "used_in_bytes" : 524288
                            },
                            "cgroup" : {
                                "cpuacct" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "usage_nanos" : 28451058701403
                                },
                                "cpu" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "cfs_period_micros" : 100000,
                                    "cfs_quota_micros" : -1,
                                    "stat" : {
                                        "number_of_elapsed_periods" : 0,
                                        "number_of_times_throttled" : 0,
                                        "time_throttled_nanos" : 0
                                    }
                                },
                                "memory" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "limit_in_bytes" : "9223372036854771712",
                                    "usage_in_bytes" : "11654156288"
                                }
                            }
                        },
                        "process" : {
                            "timestamp" : 1639459994259,
                            "open_file_descriptors" : 1720,
                            "max_file_descriptors" : 1024000,
                            "cpu" : {
                                "percent" : 0,
                                "total_in_millis" : 6645250
                            },
                            "mem" : {
                                "total_virtual_in_bytes" : 12330627072
                            }
                        },
                        "jvm" : {
                            "timestamp" : 1639459994261,
                            "uptime_in_millis" : 476882987,
                            "mem" : {
                                "heap_used_in_bytes" : 894191304,
                                "heap_used_percent" : 10,
                                "heap_committed_in_bytes" : 8572502016,
                                "heap_max_in_bytes" : 8572502016,
                                "non_heap_used_in_bytes" : 333612432,
                                "non_heap_committed_in_bytes" : 421584896,
                                "pools" : {
                                    "young" : {
                                        "used_in_bytes" : 65466784,
                                        "max_in_bytes" : 139591680,
                                        "peak_used_in_bytes" : 139591680,
                                        "peak_max_in_bytes" : 139591680
                                    },
                                    "survivor" : {
                                        "used_in_bytes" : 1788432,
                                        "max_in_bytes" : 17432576,
                                        "peak_used_in_bytes" : 17432576,
                                        "peak_max_in_bytes" : 17432576
                                    },
                                    "old" : {
                                        "used_in_bytes" : 826950920,
                                        "max_in_bytes" : 8415477760,
                                        "peak_used_in_bytes" : 826950920,
                                        "peak_max_in_bytes" : 8415477760
                                    }
                                }
                            },
                            "threads" : {
                                "count" : 149,
                                "peak_count" : 149
                            },
                            "gc" : {
                                "collectors" : {
                                    "young" : {
                                        "collection_count" : 15182,
                                        "collection_time_in_millis" : 844088
                                    },
                                    "old" : {
                                        "collection_count" : 4,
                                        "collection_time_in_millis" : 328
                                    }
                                }
                            },
                            "buffer_pools" : {
                                "mapped" : {
                                    "count" : 12,
                                    "used_in_bytes" : 181134,
                                    "total_capacity_in_bytes" : 181134
                                },
                                "direct" : {
                                    "count" : 124,
                                    "used_in_bytes" : 5723993,
                                    "total_capacity_in_bytes" : 5723991
                                }
                            },
                            "classes" : {
                                "current_loaded_count" : 38081,
                                "total_loaded_count" : 38081,
                                "total_unloaded_count" : 0
                            }
                        },
                        "thread_pool" : {
                            "ad-threadpool" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "analyze" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_started" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_store" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "flush" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 16
                            },
                            "force_merge" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 8
                            },
                            "generic" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 17,
                                "completed" : 129829
                            },
                            "get" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 11
                            },
                            "listener" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "management" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 1,
                                "rejected" : 0,
                                "largest" : 5,
                                "completed" : 545404
                            },
                            "open_distro_job_scheduler" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "refresh" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1422509
                            },
                            "search" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 171
                            },
                            "search_throttled" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "snapshot" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1
                            },
                            "snapshot_segments" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 32
                            },
                            "snapshot_shards" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 394
                            },
                            "sql-worker" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "ultrawarm_migration" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "warmer" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "write" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 64
                            }
                        },
                        "fs" : {
                            "timestamp" : 1639459994261,
                            "total" : {
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528399360,
                                "available_in_bytes" : 10511622144
                            },
                            "data" : [ {
                                "type" : "ext4",
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528399360,
                                "available_in_bytes" : 10511622144
                            } ],
                            "io_stats" : {
                                "devices" : [ {
                                    "device_name" : "dm-0",
                                    "operations" : 48867,
                                    "read_operations" : 2,
                                    "write_operations" : 48865,
                                    "read_kilobytes" : 8,
                                    "write_kilobytes" : 197588
                                } ],
                                "total" : {
                                    "operations" : 48867,
                                    "read_operations" : 2,
                                    "write_operations" : 48865,
                                    "read_kilobytes" : 8,
                                    "write_kilobytes" : 197588
                                }
                            }
                        },
                        "http" : {
                            "current_open" : 0,
                            "total_opened" : 0
                        },
                        "breakers" : {
                            "request" : {
                                "limit_size_in_bytes" : 5143501209,
                                "limit_size" : "4.7gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "fielddata" : {
                                "limit_size_in_bytes" : 3429000806,
                                "limit_size" : "3.1gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.03,
                                "tripped" : 0
                            },
                            "in_flight_requests" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 2.0,
                                "tripped" : 0
                            },
                            "accounting" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 29344,
                                "estimated_size" : "28.6kb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "parent" : {
                                "limit_size_in_bytes" : 8143876915,
                                "limit_size" : "7.5gb",
                                "estimated_size_in_bytes" : 894260408,
                                "estimated_size" : "852.8mb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            }
                        },
                        "script" : {
                            "compilations" : 0,
                            "cache_evictions" : 0,
                            "compilation_limit_triggered" : 0
                        },
                        "discovery" : {
                            "cluster_state_queue" : {
                                "total" : 0,
                                "pending" : 0,
                                "committed" : 0
                            },
                            "published_cluster_states" : {
                                "full_states" : 2,
                                "incompatible_diffs" : 0,
                                "compatible_diffs" : 1655
                            }
                        },
                        "ingest" : {
                            "total" : {
                                "count" : 0,
                                "time_in_millis" : 0,
                                "current" : 0,
                                "failed" : 0
                            },
                            "pipelines" : { }
                        },
                        "adaptive_selection" : {
                            "8dqNgo-KRYGIzJ2xsgbfEw" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 9660684,
                                "avg_response_time_ns" : 2687566,
                                "rank" : "2.7"
                            },
                            "3g-J2dUQS_m2su_yvnqRUA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 761374,
                                "avg_response_time_ns" : 2167628,
                                "rank" : "2.2"
                            },
                            "Gh_8fOzBTmaNnbQmAyd-rA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1392209,
                                "avg_response_time_ns" : 1800499,
                                "rank" : "1.8"
                            }
                        },
                        "script_cache" : {
                            "sum" : {
                                "compilations" : 0,
                                "cache_evictions" : 0,
                                "compilation_limit_triggered" : 0
                            }
                        },
                        "indexing_pressure" : {
                            "memory" : {
                                "current" : {
                                    "combined_coordinating_and_primary_in_bytes" : 0,
                                    "coordinating_in_bytes" : 0,
                                    "primary_in_bytes" : 0,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 0
                                },
                                "total" : {
                                    "combined_coordinating_and_primary_in_bytes" : 2381025,
                                    "coordinating_in_bytes" : 1271939,
                                    "primary_in_bytes" : 1596269,
                                    "replica_in_bytes" : 54564,
                                    "all_in_bytes" : 2435589,
                                    "coordinating_rejections" : 0,
                                    "primary_rejections" : 0,
                                    "replica_rejections" : 0
                                }
                            }
                        },
                        "shard_indexing_pressure" : {
                            "stats" : { },
                            "total_rejections_breakup_shadow_mode" : {
                                "node_limits" : 0,
                                "no_successful_request_limits" : 0,
                                "throughput_degradation_limits" : 0
                            },
                            "enabled" : 'true',
                            "enforced" : 'false'
                        }
                    },
                    "8dqNgo-KRYGIzJ2xsgbfEw" : {
                        "timestamp" : 1639459994260,
                        "name" : "c6051c73530bb3b94b30d92e2877867e",
                        "roles" : [ "data", "ingest", "master", "remote_cluster_client" ],
                        "indices" : {
                            "docs" : {
                                "count" : 196,
                                "deleted" : 4
                            },
                            "store" : {
                                "size_in_bytes" : 153617
                            },
                            "indexing" : {
                                "index_total" : 1653,
                                "index_time_in_millis" : 1526,
                                "index_current" : 0,
                                "index_failed" : 18,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 11,
                                "time_in_millis" : 93,
                                "exists_total" : 11,
                                "exists_time_in_millis" : 93,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 134,
                                "query_time_in_millis" : 118,
                                "query_current" : 0,
                                "fetch_total" : 68,
                                "fetch_time_in_millis" : 108,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 83,
                                "total_docs" : 10,
                                "total_size_in_bytes" : 53955,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 230686720
                            },
                            "refresh" : {
                                "total" : 92,
                                "total_time_in_millis" : 731,
                                "external_total" : 69,
                                "external_total_time_in_millis" : 687,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 11,
                                "periodic" : 0,
                                "total_time_in_millis" : 16
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 20,
                                "total_time_in_millis" : 6
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 9,
                                "memory_in_bytes" : 39140,
                                "terms_memory_in_bytes" : 26976,
                                "stored_fields_memory_in_bytes" : 4456,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 3392,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 4316,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 48,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 220,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 220,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 1078,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 1
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "os" : {
                            "timestamp" : 1639459994261,
                            "cpu" : {
                                "percent" : 2,
                                "load_average" : {
                                    "1m" : 0.07,
                                    "5m" : 0.19,
                                    "15m" : 0.23
                                }
                            },
                            "mem" : {
                                "total_in_bytes" : 16480382976,
                                "free_in_bytes" : 416960512,
                                "used_in_bytes" : 16063422464,
                                "free_percent" : 3,
                                "used_percent" : 97
                            },
                            "swap" : {
                                "total_in_bytes" : 2147479552,
                                "free_in_bytes" : 2147217408,
                                "used_in_bytes" : 262144
                            },
                            "cgroup" : {
                                "cpuacct" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "usage_nanos" : 31461420075888
                                },
                                "cpu" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "cfs_period_micros" : 100000,
                                    "cfs_quota_micros" : -1,
                                    "stat" : {
                                        "number_of_elapsed_periods" : 0,
                                        "number_of_times_throttled" : 0,
                                        "time_throttled_nanos" : 0
                                    }
                                },
                                "memory" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "limit_in_bytes" : "9223372036854771712",
                                    "usage_in_bytes" : "11988336640"
                                }
                            }
                        },
                        "process" : {
                            "timestamp" : 1639459994261,
                            "open_file_descriptors" : 1722,
                            "max_file_descriptors" : 1024000,
                            "cpu" : {
                                "percent" : 0,
                                "total_in_millis" : 7539800
                            },
                            "mem" : {
                                "total_virtual_in_bytes" : 12402905088
                            }
                        },
                        "jvm" : {
                            "timestamp" : 1639459994262,
                            "uptime_in_millis" : 476871234,
                            "mem" : {
                                "heap_used_in_bytes" : 1063432744,
                                "heap_used_percent" : 12,
                                "heap_committed_in_bytes" : 8572502016,
                                "heap_max_in_bytes" : 8572502016,
                                "non_heap_used_in_bytes" : 345156048,
                                "non_heap_committed_in_bytes" : 428797952,
                                "pools" : {
                                    "young" : {
                                        "used_in_bytes" : 138802096,
                                        "max_in_bytes" : 139591680,
                                        "peak_used_in_bytes" : 139591680,
                                        "peak_max_in_bytes" : 139591680
                                    },
                                    "survivor" : {
                                        "used_in_bytes" : 2165352,
                                        "max_in_bytes" : 17432576,
                                        "peak_used_in_bytes" : 17432576,
                                        "peak_max_in_bytes" : 17432576
                                    },
                                    "old" : {
                                        "used_in_bytes" : 922488776,
                                        "max_in_bytes" : 8415477760,
                                        "peak_used_in_bytes" : 922488776,
                                        "peak_max_in_bytes" : 8415477760
                                    }
                                }
                            },
                            "threads" : {
                                "count" : 146,
                                "peak_count" : 147
                            },
                            "gc" : {
                                "collectors" : {
                                    "young" : {
                                        "collection_count" : 15931,
                                        "collection_time_in_millis" : 873377
                                    },
                                    "old" : {
                                        "collection_count" : 4,
                                        "collection_time_in_millis" : 361
                                    }
                                }
                            },
                            "buffer_pools" : {
                                "mapped" : {
                                    "count" : 16,
                                    "used_in_bytes" : 131826,
                                    "total_capacity_in_bytes" : 131826
                                },
                                "direct" : {
                                    "count" : 124,
                                    "used_in_bytes" : 5819744,
                                    "total_capacity_in_bytes" : 5819742
                                }
                            },
                            "classes" : {
                                "current_loaded_count" : 38718,
                                "total_loaded_count" : 38719,
                                "total_unloaded_count" : 1
                            }
                        },
                        "thread_pool" : {
                            "ad-threadpool" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "analyze" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_started" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_store" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "flush" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 10
                            },
                            "force_merge" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 8
                            },
                            "generic" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 157065
                            },
                            "get" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 11
                            },
                            "listener" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "management" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 1,
                                "rejected" : 0,
                                "largest" : 5,
                                "completed" : 593330
                            },
                            "open_distro_job_scheduler" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "refresh" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1899220
                            },
                            "search" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 250
                            },
                            "search_throttled" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "snapshot" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1337
                            },
                            "snapshot_segments" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 50
                            },
                            "snapshot_shards" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 399
                            },
                            "sql-worker" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "ultrawarm_migration" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "warmer" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1
                            },
                            "write" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 78
                            }
                        },
                        "fs" : {
                            "timestamp" : 1639459994262,
                            "total" : {
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528395264,
                                "available_in_bytes" : 10511618048
                            },
                            "least_usage_estimate" : {
                                "total_in_bytes" : 10566782976,
                                "available_in_bytes" : 10511618048,
                                "used_disk_percent" : 0.5220598182558973
                            },
                            "most_usage_estimate" : {
                                "total_in_bytes" : 10566782976,
                                "available_in_bytes" : 10511618048,
                                "used_disk_percent" : 0.5220598182558973
                            },
                            "data" : [ {
                                "type" : "ext4",
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528395264,
                                "available_in_bytes" : 10511618048
                            } ],
                            "io_stats" : {
                                "devices" : [ {
                                    "device_name" : "dm-0",
                                    "operations" : 48493,
                                    "read_operations" : 1,
                                    "write_operations" : 48492,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 195160
                                } ],
                                "total" : {
                                    "operations" : 48493,
                                    "read_operations" : 1,
                                    "write_operations" : 48492,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 195160
                                }
                            }
                        },
                        "http" : {
                            "current_open" : 0,
                            "total_opened" : 0
                        },
                        "breakers" : {
                            "request" : {
                                "limit_size_in_bytes" : 5143501209,
                                "limit_size" : "4.7gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "fielddata" : {
                                "limit_size_in_bytes" : 3429000806,
                                "limit_size" : "3.1gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.03,
                                "tripped" : 0
                            },
                            "in_flight_requests" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 1416,
                                "estimated_size" : "1.3kb",
                                "overhead" : 2.0,
                                "tripped" : 0
                            },
                            "accounting" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 38948,
                                "estimated_size" : "38kb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "parent" : {
                                "limit_size_in_bytes" : 8143876915,
                                "limit_size" : "7.5gb",
                                "estimated_size_in_bytes" : 1063504312,
                                "estimated_size" : "1014.2mb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            }
                        },
                        "script" : {
                            "compilations" : 0,
                            "cache_evictions" : 0,
                            "compilation_limit_triggered" : 0
                        },
                        "discovery" : {
                            "cluster_state_queue" : {
                                "total" : 0,
                                "pending" : 0,
                                "committed" : 0
                            },
                            "published_cluster_states" : {
                                "full_states" : 2,
                                "incompatible_diffs" : 0,
                                "compatible_diffs" : 1655
                            }
                        },
                        "ingest" : {
                            "total" : {
                                "count" : 0,
                                "time_in_millis" : 0,
                                "current" : 0,
                                "failed" : 0
                            },
                            "pipelines" : { }
                        },
                        "adaptive_selection" : {
                            "8dqNgo-KRYGIzJ2xsgbfEw" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1085299,
                                "avg_response_time_ns" : 8921949,
                                "rank" : "8.9"
                            },
                            "3g-J2dUQS_m2su_yvnqRUA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 560096,
                                "avg_response_time_ns" : 2817684,
                                "rank" : "2.8"
                            },
                            "Gh_8fOzBTmaNnbQmAyd-rA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1285417,
                                "avg_response_time_ns" : 2861637,
                                "rank" : "2.9"
                            }
                        },
                        "script_cache" : {
                            "sum" : {
                                "compilations" : 0,
                                "cache_evictions" : 0,
                                "compilation_limit_triggered" : 0
                            }
                        },
                        "indexing_pressure" : {
                            "memory" : {
                                "current" : {
                                    "combined_coordinating_and_primary_in_bytes" : 0,
                                    "coordinating_in_bytes" : 0,
                                    "primary_in_bytes" : 0,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 0
                                },
                                "total" : {
                                    "combined_coordinating_and_primary_in_bytes" : 1862276,
                                    "coordinating_in_bytes" : 1268361,
                                    "primary_in_bytes" : 878293,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 1862276,
                                    "coordinating_rejections" : 0,
                                    "primary_rejections" : 0,
                                    "replica_rejections" : 0
                                }
                            }
                        },
                        "shard_indexing_pressure" : {
                            "stats" : { },
                            "total_rejections_breakup_shadow_mode" : {
                                "node_limits" : 0,
                                "no_successful_request_limits" : 0,
                                "throughput_degradation_limits" : 0
                            },
                            "enabled" : 'true',
                            "enforced" : 'false'
                        }
                    }
                }
            }
        except elasticsearch.TransportError:
            logging.getLogger(__name__).exception("Could not retrieve node stats.")
            return {}
        return stats["nodes"].values()


class TransformStats(TelemetryDevice):
    internal = False
    command = "transform-stats"
    human_name = "Transform Stats"
    help = "Regularly samples transform stats"

    """
    Gathers Transform stats
    """

    def __init__(self, telemetry_params, clients, metrics_store):
        """
        :param telemetry_params: The configuration object for telemetry_params.
        :param clients: A dict of clients to all clusters.
        :param metrics_store: The configured metrics store we write to.
        """
        super().__init__()

        self.telemetry_params = telemetry_params
        self.clients = clients
        self.sample_interval = telemetry_params.get("transform-stats-sample-interval", 1)
        if self.sample_interval <= 0:
            raise exceptions.SystemSetupError(
                f"The telemetry parameter 'transform-stats-sample-interval' must be greater than zero "
                f"but was [{self.sample_interval}].")
        self.specified_cluster_names = self.clients.keys()
        self.transforms_per_cluster = self.telemetry_params.get("transform-stats-transforms", False)
        if self.transforms_per_cluster:
            for cluster_name in self.transforms_per_cluster.keys():
                if cluster_name not in clients:
                    raise exceptions.SystemSetupError(
                        f"The telemetry parameter 'transform-stats-transforms' must be a JSON Object with keys "
                        f"matching the cluster names [{','.join(sorted(clients.keys()))}] specified in --target-hosts "
                        f"but it had [{cluster_name}].")
            self.specified_cluster_names = self.transforms_per_cluster.keys()

        self.metrics_store = metrics_store
        self.samplers = []

    def on_benchmark_start(self):
        for cluster_name in self.specified_cluster_names:
            recorder = TransformStatsRecorder(cluster_name, self.clients[cluster_name], self.metrics_store,
                                              self.sample_interval,
                                              self.transforms_per_cluster[
                                                  cluster_name] if self.transforms_per_cluster else None)
            sampler = SamplerThread(recorder)
            self.samplers.append(sampler)
            sampler.setDaemon(True)
            # we don't require starting recorders precisely at the same time
            sampler.start()

    def on_benchmark_stop(self):
        if self.samplers:
            for sampler in self.samplers:
                sampler.finish()
                # record the final stats
                sampler.recorder.record_final()


class TransformStatsRecorder:
    """
    Collects and pushes Transform stats for the specified cluster to the metric store.
    """

    def __init__(self, cluster_name, client, metrics_store, sample_interval, transforms=None):
        """
        :param cluster_name: The cluster_name that the client connects to, as specified in target.hosts.
        :param client: The OpenSearch client for this cluster.
        :param metrics_store: The configured metrics store we write to.
        :param sample_interval: integer controlling the interval, in seconds, between collecting samples.
        :param transforms: optional list of transforms to filter results from.
        """

        self.cluster_name = cluster_name
        self.client = client
        self.metrics_store = metrics_store
        self.sample_interval = sample_interval
        self.transforms = transforms
        self.logger = logging.getLogger(__name__)

        self.logger.info("transform stats recorder")

    def __str__(self):
        return "transform stats"

    def record(self):
        """
        Collect Transform stats for transforms (optionally) specified in telemetry parameters and push to metrics store.
        """

        self._record()

    def record_final(self):
        """
        Collect final Transform stats for transforms (optionally) specified in telemetry parameters and push to metrics store.
        """

        self._record("total_")

    def _record(self, prefix=""):
        # ES returns all stats values in bytes or ms via "human: 'false'"

        # pylint: disable=import-outside-toplevel
        import elasticsearch

        try:
            stats = self.client.transform.get_transform_stats("_all")

        except elasticsearch.TransportError:
            msg = f"A transport error occurred while collecting transform stats on " \
                  f"cluster [{self.cluster_name}]"
            self.logger.exception(msg)
            raise exceptions.BenchmarkError(msg)

        for transform in stats["transforms"]:
            try:
                if self.transforms and transform["id"] not in self.transforms:
                    # Skip metrics for transform not part of user supplied whitelist (transform-stats-transforms)
                    # in telemetry params.
                    continue
                self.record_stats_per_transform(transform["id"], transform["stats"], prefix)

            except KeyError:
                self.logger.warning(
                    "The 'transform' key does not contain a 'transform' or 'stats' key "
                    "Maybe the output format has changed. Skipping."
                )

    def record_stats_per_transform(self, transform_id, stats, prefix=""):
        """
        :param transform_id: The transform id.
        :param stats: A dict with returned transform stats for the transform.
        :param prefix: A prefix for the counters/values, e.g. for total runtimes
        """

        meta_data = {
            "transform_id": transform_id
        }

        self.metrics_store.put_value_cluster_level(prefix + "transform_pages_processed",
                                                   stats.get("pages_processed", 0),
                                                   meta_data=meta_data)
        self.metrics_store.put_value_cluster_level(prefix + "transform_documents_processed",
                                                   stats.get("documents_processed", 0),
                                                   meta_data=meta_data)
        self.metrics_store.put_value_cluster_level(prefix + "transform_documents_indexed",
                                                   stats.get("documents_indexed", 0),
                                                   meta_data=meta_data)
        self.metrics_store.put_value_cluster_level(prefix + "transform_index_total", stats.get("index_total", 0),
                                                   meta_data=meta_data)
        self.metrics_store.put_value_cluster_level(prefix + "transform_index_failures", stats.get("index_failures", 0),
                                                   meta_data=meta_data)
        self.metrics_store.put_value_cluster_level(prefix + "transform_search_total", stats.get("search_total", 0),
                                                   meta_data=meta_data)
        self.metrics_store.put_value_cluster_level(prefix + "transform_search_failures",
                                                   stats.get("search_failures", 0),
                                                   meta_data=meta_data)
        self.metrics_store.put_value_cluster_level(prefix + "transform_processing_total",
                                                   stats.get("processing_total", 0),
                                                   meta_data=meta_data)

        self.metrics_store.put_value_cluster_level(prefix + "transform_search_time",
                                                   stats.get("search_time_in_ms", 0),
                                                   "ms", meta_data=meta_data)
        self.metrics_store.put_value_cluster_level(prefix + "transform_index_time",
                                                   stats.get("index_time_in_ms", 0), "ms",
                                                   meta_data=meta_data)
        self.metrics_store.put_value_cluster_level(prefix + "transform_processing_time",
                                                   stats.get("processing_time_in_ms", 0), "ms",
                                                   meta_data=meta_data)

        documents_processed = stats.get("documents_processed", 0)
        processing_time = stats.get("search_time_in_ms", 0)
        processing_time += stats.get("processing_time_in_ms", 0)
        processing_time += stats.get("index_time_in_ms", 0)

        if processing_time > 0:
            throughput = documents_processed / processing_time * 1000
            self.metrics_store.put_value_cluster_level(prefix + "transform_throughput", throughput,
                                                       "docs/s", meta_data=meta_data)


class SearchableSnapshotsStats(TelemetryDevice):
    internal = False
    command = "searchable-snapshots-stats"
    human_name = "Searchable Snapshots Stats"
    help = "Regularly samples searchable snapshots stats"

    """
    Gathers searchable snapshots stats on a cluster level
    """
    def __init__(self, telemetry_params, clients, metrics_store):
        """
        :param telemetry_params: The configuration object for telemetry_params.
            May optionally specify:
            ``searchable-stats-indices``: str with index/index-pattern or list of indices or index-patterns
            that stats should be collected from.
            Specifying this will implicitly use the level=indices parameter in the API call, as opposed to
            the level=cluster (default).

            Not all clusters need to be specified, but any name used must be be present in target.hosts.
            Alternatively, the index or index pattern can be specified as a string in case only one cluster is involved.

            Examples:

            --telemetry-params="searchable-snapshots-stats-indices:opensearchlogs-2020-01-01"

            --telemetry-params="searchable-snapshots-stats-indices:opensearchlogs*"

            --telemetry-params=./telemetry-params.json

            where telemetry-params.json is:

            {
              "searchable-snapshots-stats-indices": {
                "default": ["leader-opensearchlogs-*"],
                "follower": ["follower-opensearchlogs-*"]
              }
            }


            ``searchable-snapshots-stats-sample-interval``: positive integer controlling the sampling interval.
            Default: 1 second.
        :param clients: A dict of clients to all clusters.
        :param metrics_store: The configured metrics store we write to.
        """
        super().__init__()

        self.telemetry_params = telemetry_params
        self.clients = clients
        self.sample_interval = telemetry_params.get("searchable-snapshots-stats-sample-interval", 1)
        if self.sample_interval <= 0:
            raise exceptions.SystemSetupError(
                f"The telemetry parameter 'searchable-snapshots-stats-sample-interval' must be greater than zero "
                f"but was {self.sample_interval}.")
        self.specified_cluster_names = self.clients.keys()
        indices_per_cluster = self.telemetry_params.get("searchable-snapshots-stats-indices", None)
        # allow the user to specify either an index pattern as string or as a JSON object
        if isinstance(indices_per_cluster, str):
            self.indices_per_cluster = {opts.TargetHosts.DEFAULT: [indices_per_cluster]}
        else:
            self.indices_per_cluster = indices_per_cluster

        if self.indices_per_cluster:
            for cluster_name in self.indices_per_cluster.keys():
                if cluster_name not in clients:
                    raise exceptions.SystemSetupError(
                        f"The telemetry parameter 'searchable-snapshots-stats-indices' must be a JSON Object "
                        f"with keys matching the cluster names [{','.join(sorted(clients.keys()))}] specified in "
                        f"--target-hosts but it had [{cluster_name}].")
            self.specified_cluster_names = self.indices_per_cluster.keys()

        self.metrics_store = metrics_store
        self.samplers = []

    def on_benchmark_start(self):
        for cluster_name in self.specified_cluster_names:
            recorder = SearchableSnapshotsStatsRecorder(
                cluster_name, self.clients[cluster_name], self.metrics_store, self.sample_interval,
                self.indices_per_cluster[cluster_name] if self.indices_per_cluster else None)
            sampler = SamplerThread(recorder)
            self.samplers.append(sampler)
            sampler.setDaemon(True)
            # we don't require starting recorders precisely at the same time
            sampler.start()

    def on_benchmark_stop(self):
        if self.samplers:
            for sampler in self.samplers:
                sampler.finish()


class SearchableSnapshotsStatsRecorder:
    """
    Collects and pushes searchable snapshots stats for the specified cluster to the metric store.
    """

    def __init__(self, cluster_name, client, metrics_store, sample_interval, indices=None):
        """
        :param cluster_name: The cluster_name that the client connects to, as specified in target.hosts.
        :param client: The OpenSearch client for this cluster.
        :param metrics_store: The configured metrics store we write to.
        :param sample_interval: integer controlling the interval, in seconds, between collecting samples.
        :param indices: optional list of indices to filter results from.
        """

        self.cluster_name = cluster_name
        self.client = client
        self.metrics_store = metrics_store
        self.sample_interval = sample_interval
        self.indices = indices
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return "searchable snapshots stats"

    def record(self):
        """
        Collect searchable snapshots stats for indexes (optionally) specified in telemetry parameters
        and push to metrics store.
        """
        # pylint: disable=import-outside-toplevel
        import elasticsearch

        try:
            stats_api_endpoint = "/_searchable_snapshots/stats"
            level = "indices" if self.indices else "cluster"
            # we don't use the existing client support (searchable_snapshots.stats())
            # as the API is deliberately undocumented and might change:
            stats = self.client.transport.perform_request("GET", stats_api_endpoint, params={"level": level})
        except elasticsearch.NotFoundError as e:
            if "No searchable snapshots indices found" in e.info.get("error").get("reason"):
                self.logger.info(
                    "Unable to find valid indices while collecting searchable snapshots stats "
                    "on cluster [%s]", self.cluster_name)
                # allow collection, indices might be mounted later on
                return
        except elasticsearch.TransportError:
            raise exceptions.BenchmarkError(
                f"A transport error occurred while collecting searchable snapshots stats on cluster "
                f"[{self.cluster_name}]") from None

        total_stats = stats.get("total", [])
        for lucene_file_stats in total_stats:
            self._push_stats(level="cluster", stats=lucene_file_stats)

        if self.indices:
            for idx, idx_stats in stats.get("indices", {}).items():
                if not self._match_list_or_pattern(idx):
                    continue

                for lucene_file_stats in idx_stats.get("total", []):
                    self._push_stats(level="index", stats=lucene_file_stats, index=idx)

    def _push_stats(self, level, stats, index=None):
        doc = {
            "name": "searchable-snapshots-stats",
            # be lenient as the API is still WiP
            "lucene_file_type": stats.get("file_ext"),
            "stats": stats,
        }

        if index:
            doc["index"] = index

        meta_data = {
            "cluster": self.cluster_name,
            "level": level
        }

        self.metrics_store.put_doc(doc, level=MetaInfoScope.cluster, meta_data=meta_data)

    # TODO Consider moving under the utils package for broader/future use?
    # at the moment it's only useful here as this stats API is undocumented and we don't wan't to use
    # a specific opensearch-py stats method that could support index filtering in a standard way
    def _match_list_or_pattern(self, idx):
        """
        Match idx from self.indices

        :param idx: String that may include shell style wildcards (https://docs.python.org/3/library/fnmatch.html)
        :return: Boolean if idx matches anything from self.indices
        """
        for index_param in self.indices:
            if fnmatch.fnmatch(idx, index_param):
                return True
        return False

class StartupTime(InternalTelemetryDevice):
    def __init__(self, stopwatch=time.StopWatch):
        super().__init__()
        self.timer = stopwatch()

    def on_pre_node_start(self, node_name):
        self.timer.start()

    def attach_to_node(self, node):
        self.timer.stop()

    def store_system_metrics(self, node, metrics_store):
        metrics_store.put_value_node_level(node.node_name, "node_startup_time", self.timer.total_time(), "s")


class DiskIo(InternalTelemetryDevice):
    """
    Gathers disk I/O stats.
    """
    def __init__(self, node_count_on_host):
        super().__init__()
        self.node_count_on_host = node_count_on_host
        self.read_bytes = None
        self.write_bytes = None

    def attach_to_node(self, node):
        os_process = sysstats.setup_process_stats(node.pid)
        process_start = sysstats.process_io_counters(os_process)
        if process_start:
            self.read_bytes = process_start.read_bytes
            self.write_bytes = process_start.write_bytes
            self.logger.info("Using more accurate process-based I/O counters.")
        else:
            # noinspection PyBroadException
            try:
                disk_start = sysstats.disk_io_counters()
                self.read_bytes = disk_start.read_bytes
                self.write_bytes = disk_start.write_bytes
                self.logger.warning("Process I/O counters are not supported on this platform. Falling back to less "
                                    "accurate disk I/O counters.")
            except BaseException:
                self.logger.exception("Could not determine I/O stats at benchmark start.")

    def detach_from_node(self, node, running):
        if running:
            # Be aware the semantics of write counts etc. are different for disk and process statistics.
            # Thus we're conservative and only publish I/O bytes now.
            # noinspection PyBroadException
            try:
                os_process = sysstats.setup_process_stats(node.pid)
                process_end = sysstats.process_io_counters(os_process)
                # we have process-based disk counters, no need to worry how many nodes are on this host
                if process_end:
                    self.read_bytes = process_end.read_bytes - self.read_bytes
                    self.write_bytes = process_end.write_bytes - self.write_bytes
                else:
                    disk_end = sysstats.disk_io_counters()
                    if self.node_count_on_host > 1:
                        self.logger.info("There are [%d] nodes on this host and Benchmark fell back to disk I/O counters. "
                                         "Attributing [1/%d] of total I/O to [%s].",
                                         self.node_count_on_host, self.node_count_on_host, node.node_name)

                    self.read_bytes = (disk_end.read_bytes - self.read_bytes) // self.node_count_on_host
                    self.write_bytes = (disk_end.write_bytes - self.write_bytes) // self.node_count_on_host
            # Catching RuntimeException is not sufficient: psutil might raise AccessDenied (derived from Exception)
            except BaseException:
                self.logger.exception("Could not determine I/O stats at benchmark end.")
                # reset all counters so we don't attempt to write inconsistent numbers to the metrics store later on
                self.read_bytes = None
                self.write_bytes = None

    def store_system_metrics(self, node, metrics_store):
        if self.write_bytes is not None:
            metrics_store.put_value_node_level(node.node_name, "disk_io_write_bytes", self.write_bytes, "byte")
        if self.read_bytes is not None:
            metrics_store.put_value_node_level(node.node_name, "disk_io_read_bytes", self.read_bytes, "byte")


def store_node_attribute_metadata(metrics_store, nodes_info):
    # push up all node level attributes to cluster level iff the values are identical for all nodes
    pseudo_cluster_attributes = {}
    for node in nodes_info:
        if "attributes" in node:
            for k, v in node["attributes"].items():
                attribute_key = "attribute_%s" % str(k)
                metrics_store.add_meta_info(metrics.MetaInfoScope.node, node["name"], attribute_key, v)
                if attribute_key not in pseudo_cluster_attributes:
                    pseudo_cluster_attributes[attribute_key] = set()
                pseudo_cluster_attributes[attribute_key].add(v)

    for k, v in pseudo_cluster_attributes.items():
        if len(v) == 1:
            metrics_store.add_meta_info(metrics.MetaInfoScope.cluster, None, k, next(iter(v)))


def store_plugin_metadata(metrics_store, nodes_info):
    # push up all plugins to cluster level iff all nodes have the same ones
    all_nodes_plugins = []
    all_same = False

    for node in nodes_info:
        plugins = [p["name"] for p in extract_value(node, ["plugins"], fallback=[]) if "name" in p]
        if not all_nodes_plugins:
            all_nodes_plugins = plugins.copy()
            all_same = True
        else:
            # order does not matter so we do a set comparison
            all_same = all_same and set(all_nodes_plugins) == set(plugins)

        if plugins:
            metrics_store.add_meta_info(metrics.MetaInfoScope.node, node["name"], "plugins", plugins)

    if all_same and all_nodes_plugins:
        metrics_store.add_meta_info(metrics.MetaInfoScope.cluster, None, "plugins", all_nodes_plugins)


def extract_value(node, path, fallback="unknown"):
    value = node
    try:
        for k in path:
            value = value[k]
    except KeyError:
        value = fallback
    return value


class ClusterEnvironmentInfo(InternalTelemetryDevice):
    """
    Gathers static environment information on a cluster level (e.g. version numbers).
    """
    def __init__(self, client, metrics_store):
        super().__init__()
        self.metrics_store = metrics_store
        self.client = client

    def on_benchmark_start(self):
        # noinspection PyBroadException
        try:
            client_info = {
                "name" : "790a811a99da41e60c11490b37aad552",
                "cluster_name" : "397869430111:mensor-metrics",
                "cluster_uuid" : "E-DOkPUaQEeMRcnO29oR2w",
                "version" : {
                    "number" : "7.7.0",
                    "build_flavor" : "oss",
                    "build_type" : "tar",
                    "build_hash" : "unknown",
                    "build_date" : "2021-05-21T16:27:33.851846Z",
                    "build_snapshot" : 'false',
                    "lucene_version" : "8.5.1",
                    "minimum_wire_compatibility_version" : "6.8.0",
                    "minimum_index_compatibility_version" : "6.0.0-beta1"
                },
                "tagline" : "You Know, for Search"
            }
        except BaseException:
            self.logger.exception("Could not retrieve cluster version info")
            return
        revision = client_info["version"]["build_hash"]
        distribution_version = client_info["version"]["number"]
        # older versions (pre 6.3.0) don't expose a build_flavor property because the only (implicit) flavor was "oss".
        distribution_flavor = client_info["version"].get("build_flavor", "oss")
        self.metrics_store.add_meta_info(metrics.MetaInfoScope.cluster, None, "source_revision", revision)
        self.metrics_store.add_meta_info(metrics.MetaInfoScope.cluster, None, "distribution_version", distribution_version)
        self.metrics_store.add_meta_info(metrics.MetaInfoScope.cluster, None, "distribution_flavor", distribution_flavor)

        info = self.client.nodes.info(node_id="_all")
        nodes_info = info["nodes"].values()
        for node in nodes_info:
            node_name = node["name"]
            # while we could determine this for bare-metal nodes that are
            # provisioned by Benchmark, there are other cases (Docker, externally
            # provisioned clusters) where it's not that easy.
            self.metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "jvm_vendor", extract_value(node, ["jvm", "vm_vendor"]))
            self.metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "jvm_version", extract_value(node, ["jvm", "version"]))

        store_plugin_metadata(self.metrics_store, nodes_info)
        store_node_attribute_metadata(self.metrics_store, nodes_info)


def add_metadata_for_node(metrics_store, node_name, host_name):
    """
    Gathers static environment information like OS or CPU details for benchmark-provisioned nodes.
    """
    metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "os_name", sysstats.os_name())
    metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "os_version", sysstats.os_version())
    metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "cpu_logical_cores", sysstats.logical_cpu_cores())
    metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "cpu_physical_cores", sysstats.physical_cpu_cores())
    metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "cpu_model", sysstats.cpu_model())
    metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "node_name", node_name)
    metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "host_name", host_name)


class ExternalEnvironmentInfo(InternalTelemetryDevice):
    """
    Gathers static environment information for externally provisioned clusters.
    """
    def __init__(self, client, metrics_store):
        super().__init__()
        self.metrics_store = metrics_store
        self.client = client

    # noinspection PyBroadException
    def on_benchmark_start(self):
        try:
            node_stas_value = {
                "_nodes" : {
                    "total" : 3,
                    "successful" : 3,
                    "failed" : 0
                },
                "cluster_name" : "397869430111:juno-mcm",
                "nodes" : {
                    "3g-J2dUQS_m2su_yvnqRUA" : {
                        "timestamp" : 1639459994258,
                        "name" : "29f760e1a8adb54ed33d8eab83248fa0",
                        "roles" : [ "data", "ingest", "master", "remote_cluster_client" ],
                        "indices" : {
                            "docs" : {
                                "count" : 404,
                                "deleted" : 3
                            },
                            "store" : {
                                "size_in_bytes" : 185622
                            },
                            "indexing" : {
                                "index_total" : 3184,
                                "index_time_in_millis" : 2130,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 18,
                                "time_in_millis" : 69,
                                "exists_total" : 18,
                                "exists_time_in_millis" : 69,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 67,
                                "query_time_in_millis" : 34,
                                "query_current" : 0,
                                "fetch_total" : 1,
                                "fetch_time_in_millis" : 2,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 73,
                                "total_docs" : 10,
                                "total_size_in_bytes" : 53955,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 377487360
                            },
                            "refresh" : {
                                "total" : 106,
                                "total_time_in_millis" : 441,
                                "external_total" : 87,
                                "external_total_time_in_millis" : 418,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 16,
                                "periodic" : 0,
                                "total_time_in_millis" : 16
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 20,
                                "total_time_in_millis" : 4
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 5,
                                "memory_in_bytes" : 27524,
                                "terms_memory_in_bytes" : 15920,
                                "stored_fields_memory_in_bytes" : 2504,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 2112,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 6988,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 0,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 220,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 220,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 1078,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 1
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "os" : {
                            "timestamp" : 1639459994258,
                            "cpu" : {
                                "percent" : 3,
                                "load_average" : {
                                    "1m" : 0.04,
                                    "5m" : 0.07,
                                    "15m" : 0.11
                                }
                            },
                            "mem" : {
                                "total_in_bytes" : 16480382976,
                                "free_in_bytes" : 412884992,
                                "used_in_bytes" : 16067497984,
                                "free_percent" : 3,
                                "used_percent" : 97
                            },
                            "swap" : {
                                "total_in_bytes" : 2147479552,
                                "free_in_bytes" : 2146955264,
                                "used_in_bytes" : 524288
                            },
                            "cgroup" : {
                                "cpuacct" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "usage_nanos" : 27963478780746
                                },
                                "cpu" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "cfs_period_micros" : 100000,
                                    "cfs_quota_micros" : -1,
                                    "stat" : {
                                        "number_of_elapsed_periods" : 0,
                                        "number_of_times_throttled" : 0,
                                        "time_throttled_nanos" : 0
                                    }
                                },
                                "memory" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "limit_in_bytes" : "9223372036854771712",
                                    "usage_in_bytes" : "11649228800"
                                }
                            }
                        },
                        "process" : {
                            "timestamp" : 1639459994259,
                            "open_file_descriptors" : 1724,
                            "max_file_descriptors" : 1024000,
                            "cpu" : {
                                "percent" : 0,
                                "total_in_millis" : 6614350
                            },
                            "mem" : {
                                "total_virtual_in_bytes" : 12349054976
                            }
                        },
                        "jvm" : {
                            "timestamp" : 1639459994260,
                            "uptime_in_millis" : 476870141,
                            "mem" : {
                                "heap_used_in_bytes" : 894324928,
                                "heap_used_percent" : 10,
                                "heap_committed_in_bytes" : 8572502016,
                                "heap_max_in_bytes" : 8572502016,
                                "non_heap_used_in_bytes" : 333200872,
                                "non_heap_committed_in_bytes" : 395907072,
                                "pools" : {
                                    "young" : {
                                        "used_in_bytes" : 52322208,
                                        "max_in_bytes" : 139591680,
                                        "peak_used_in_bytes" : 139591680,
                                        "peak_max_in_bytes" : 139591680
                                    },
                                    "survivor" : {
                                        "used_in_bytes" : 2067384,
                                        "max_in_bytes" : 17432576,
                                        "peak_used_in_bytes" : 17432576,
                                        "peak_max_in_bytes" : 17432576
                                    },
                                    "old" : {
                                        "used_in_bytes" : 839935336,
                                        "max_in_bytes" : 8415477760,
                                        "peak_used_in_bytes" : 839935336,
                                        "peak_max_in_bytes" : 8415477760
                                    }
                                }
                            },
                            "threads" : {
                                "count" : 145,
                                "peak_count" : 145
                            },
                            "gc" : {
                                "collectors" : {
                                    "young" : {
                                        "collection_count" : 15161,
                                        "collection_time_in_millis" : 816018
                                    },
                                    "old" : {
                                        "collection_count" : 4,
                                        "collection_time_in_millis" : 332
                                    }
                                }
                            },
                            "buffer_pools" : {
                                "mapped" : {
                                    "count" : 11,
                                    "used_in_bytes" : 162531,
                                    "total_capacity_in_bytes" : 162531
                                },
                                "direct" : {
                                    "count" : 124,
                                    "used_in_bytes" : 5763821,
                                    "total_capacity_in_bytes" : 5763819
                                }
                            },
                            "classes" : {
                                "current_loaded_count" : 37833,
                                "total_loaded_count" : 37837,
                                "total_unloaded_count" : 4
                            }
                        },
                        "thread_pool" : {
                            "ad-threadpool" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "analyze" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_started" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_store" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "flush" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 16
                            },
                            "force_merge" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 8
                            },
                            "generic" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 15,
                                "completed" : 127991
                            },
                            "get" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 10
                            },
                            "listener" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "management" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 1,
                                "rejected" : 0,
                                "largest" : 5,
                                "completed" : 545500
                            },
                            "open_distro_job_scheduler" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "refresh" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1424007
                            },
                            "search" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 117
                            },
                            "search_throttled" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "snapshot" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1
                            },
                            "snapshot_segments" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 28
                            },
                            "snapshot_shards" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 262
                            },
                            "sql-worker" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "ultrawarm_migration" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "warmer" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "write" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 63
                            }
                        },
                        "fs" : {
                            "timestamp" : 1639459994260,
                            "total" : {
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528403456,
                                "available_in_bytes" : 10511626240
                            },
                            "data" : [ {
                                "type" : "ext4",
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528403456,
                                "available_in_bytes" : 10511626240
                            } ],
                            "io_stats" : {
                                "devices" : [ {
                                    "device_name" : "dm-0",
                                    "operations" : 46888,
                                    "read_operations" : 1,
                                    "write_operations" : 46887,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 189620
                                } ],
                                "total" : {
                                    "operations" : 46888,
                                    "read_operations" : 1,
                                    "write_operations" : 46887,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 189620
                                }
                            }
                        },
                        "http" : {
                            "current_open" : 0,
                            "total_opened" : 0
                        },
                        "breakers" : {
                            "request" : {
                                "limit_size_in_bytes" : 5143501209,
                                "limit_size" : "4.7gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "fielddata" : {
                                "limit_size_in_bytes" : 3429000806,
                                "limit_size" : "3.1gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.03,
                                "tripped" : 0
                            },
                            "in_flight_requests" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 1416,
                                "estimated_size" : "1.3kb",
                                "overhead" : 2.0,
                                "tripped" : 0
                            },
                            "accounting" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 27524,
                                "estimated_size" : "26.8kb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "parent" : {
                                "limit_size_in_bytes" : 8143876915,
                                "limit_size" : "7.5gb",
                                "estimated_size_in_bytes" : 894393656,
                                "estimated_size" : "852.9mb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            }
                        },
                        "script" : {
                            "compilations" : 0,
                            "cache_evictions" : 0,
                            "compilation_limit_triggered" : 0
                        },
                        "discovery" : {
                            "cluster_state_queue" : {
                                "total" : 0,
                                "pending" : 0,
                                "committed" : 0
                            },
                            "published_cluster_states" : {
                                "full_states" : 1,
                                "incompatible_diffs" : 0,
                                "compatible_diffs" : 1654
                            }
                        },
                        "ingest" : {
                            "total" : {
                                "count" : 0,
                                "time_in_millis" : 0,
                                "current" : 0,
                                "failed" : 0
                            },
                            "pipelines" : { }
                        },
                        "adaptive_selection" : {
                            "8dqNgo-KRYGIzJ2xsgbfEw" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1200061,
                                "avg_response_time_ns" : 2868875,
                                "rank" : "2.9"
                            },
                            "3g-J2dUQS_m2su_yvnqRUA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 539218,
                                "avg_response_time_ns" : 2301522,
                                "rank" : "2.3"
                            },
                            "Gh_8fOzBTmaNnbQmAyd-rA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1194916,
                                "avg_response_time_ns" : 2677770,
                                "rank" : "2.7"
                            }
                        },
                        "script_cache" : {
                            "sum" : {
                                "compilations" : 0,
                                "cache_evictions" : 0,
                                "compilation_limit_triggered" : 0
                            }
                        },
                        "indexing_pressure" : {
                            "memory" : {
                                "current" : {
                                    "combined_coordinating_and_primary_in_bytes" : 0,
                                    "coordinating_in_bytes" : 0,
                                    "primary_in_bytes" : 0,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 0
                                },
                                "total" : {
                                    "combined_coordinating_and_primary_in_bytes" : 2515084,
                                    "coordinating_in_bytes" : 1517909,
                                    "primary_in_bytes" : 1583647,
                                    "replica_in_bytes" : 54564,
                                    "all_in_bytes" : 2569648,
                                    "coordinating_rejections" : 0,
                                    "primary_rejections" : 0,
                                    "replica_rejections" : 0
                                }
                            }
                        },
                        "shard_indexing_pressure" : {
                            "stats" : { },
                            "total_rejections_breakup_shadow_mode" : {
                                "node_limits" : 0,
                                "no_successful_request_limits" : 0,
                                "throughput_degradation_limits" : 0
                            },
                            "enabled" : 'true',
                            "enforced" : 'false'
                        }
                    },
                    "Gh_8fOzBTmaNnbQmAyd-rA" : {
                        "timestamp" : 1639459994257,
                        "name" : "84c74d57fefcca221479642d7916c1f2",
                        "roles" : [ "data", "ingest", "master", "remote_cluster_client" ],
                        "indices" : {
                            "docs" : {
                                "count" : 429,
                                "deleted" : 3
                            },
                            "store" : {
                                "size_in_bytes" : 199274
                            },
                            "indexing" : {
                                "index_total" : 3203,
                                "index_time_in_millis" : 2154,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 27,
                                "time_in_millis" : 150,
                                "exists_total" : 27,
                                "exists_time_in_millis" : 150,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 68,
                                "query_time_in_millis" : 68,
                                "query_current" : 0,
                                "fetch_total" : 68,
                                "fetch_time_in_millis" : 93,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 65,
                                "total_docs" : 10,
                                "total_size_in_bytes" : 54027,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 377487360
                            },
                            "refresh" : {
                                "total" : 113,
                                "total_time_in_millis" : 471,
                                "external_total" : 89,
                                "external_total_time_in_millis" : 468,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 17,
                                "periodic" : 0,
                                "total_time_in_millis" : 78
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 23,
                                "total_time_in_millis" : 15
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 6,
                                "memory_in_bytes" : 29344,
                                "terms_memory_in_bytes" : 16704,
                                "stored_fields_memory_in_bytes" : 2992,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 2112,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 7536,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 48,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 220,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 220,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "os" : {
                            "timestamp" : 1639459994259,
                            "cpu" : {
                                "percent" : 3,
                                "load_average" : {
                                    "1m" : 0.2,
                                    "5m" : 0.19,
                                    "15m" : 0.16
                                }
                            },
                            "mem" : {
                                "total_in_bytes" : 16480382976,
                                "free_in_bytes" : 470929408,
                                "used_in_bytes" : 16009453568,
                                "free_percent" : 3,
                                "used_percent" : 97
                            },
                            "swap" : {
                                "total_in_bytes" : 2147479552,
                                "free_in_bytes" : 2146955264,
                                "used_in_bytes" : 524288
                            },
                            "cgroup" : {
                                "cpuacct" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "usage_nanos" : 28451058701403
                                },
                                "cpu" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "cfs_period_micros" : 100000,
                                    "cfs_quota_micros" : -1,
                                    "stat" : {
                                        "number_of_elapsed_periods" : 0,
                                        "number_of_times_throttled" : 0,
                                        "time_throttled_nanos" : 0
                                    }
                                },
                                "memory" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "limit_in_bytes" : "9223372036854771712",
                                    "usage_in_bytes" : "11654156288"
                                }
                            }
                        },
                        "process" : {
                            "timestamp" : 1639459994259,
                            "open_file_descriptors" : 1720,
                            "max_file_descriptors" : 1024000,
                            "cpu" : {
                                "percent" : 0,
                                "total_in_millis" : 6645250
                            },
                            "mem" : {
                                "total_virtual_in_bytes" : 12330627072
                            }
                        },
                        "jvm" : {
                            "timestamp" : 1639459994261,
                            "uptime_in_millis" : 476882987,
                            "mem" : {
                                "heap_used_in_bytes" : 894191304,
                                "heap_used_percent" : 10,
                                "heap_committed_in_bytes" : 8572502016,
                                "heap_max_in_bytes" : 8572502016,
                                "non_heap_used_in_bytes" : 333612432,
                                "non_heap_committed_in_bytes" : 421584896,
                                "pools" : {
                                    "young" : {
                                        "used_in_bytes" : 65466784,
                                        "max_in_bytes" : 139591680,
                                        "peak_used_in_bytes" : 139591680,
                                        "peak_max_in_bytes" : 139591680
                                    },
                                    "survivor" : {
                                        "used_in_bytes" : 1788432,
                                        "max_in_bytes" : 17432576,
                                        "peak_used_in_bytes" : 17432576,
                                        "peak_max_in_bytes" : 17432576
                                    },
                                    "old" : {
                                        "used_in_bytes" : 826950920,
                                        "max_in_bytes" : 8415477760,
                                        "peak_used_in_bytes" : 826950920,
                                        "peak_max_in_bytes" : 8415477760
                                    }
                                }
                            },
                            "threads" : {
                                "count" : 149,
                                "peak_count" : 149
                            },
                            "gc" : {
                                "collectors" : {
                                    "young" : {
                                        "collection_count" : 15182,
                                        "collection_time_in_millis" : 844088
                                    },
                                    "old" : {
                                        "collection_count" : 4,
                                        "collection_time_in_millis" : 328
                                    }
                                }
                            },
                            "buffer_pools" : {
                                "mapped" : {
                                    "count" : 12,
                                    "used_in_bytes" : 181134,
                                    "total_capacity_in_bytes" : 181134
                                },
                                "direct" : {
                                    "count" : 124,
                                    "used_in_bytes" : 5723993,
                                    "total_capacity_in_bytes" : 5723991
                                }
                            },
                            "classes" : {
                                "current_loaded_count" : 38081,
                                "total_loaded_count" : 38081,
                                "total_unloaded_count" : 0
                            }
                        },
                        "thread_pool" : {
                            "ad-threadpool" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "analyze" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_started" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_store" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "flush" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 16
                            },
                            "force_merge" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 8
                            },
                            "generic" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 17,
                                "completed" : 129829
                            },
                            "get" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 11
                            },
                            "listener" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "management" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 1,
                                "rejected" : 0,
                                "largest" : 5,
                                "completed" : 545404
                            },
                            "open_distro_job_scheduler" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "refresh" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1422509
                            },
                            "search" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 171
                            },
                            "search_throttled" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "snapshot" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1
                            },
                            "snapshot_segments" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 32
                            },
                            "snapshot_shards" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 394
                            },
                            "sql-worker" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "ultrawarm_migration" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "warmer" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "write" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 64
                            }
                        },
                        "fs" : {
                            "timestamp" : 1639459994261,
                            "total" : {
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528399360,
                                "available_in_bytes" : 10511622144
                            },
                            "data" : [ {
                                "type" : "ext4",
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528399360,
                                "available_in_bytes" : 10511622144
                            } ],
                            "io_stats" : {
                                "devices" : [ {
                                    "device_name" : "dm-0",
                                    "operations" : 48867,
                                    "read_operations" : 2,
                                    "write_operations" : 48865,
                                    "read_kilobytes" : 8,
                                    "write_kilobytes" : 197588
                                } ],
                                "total" : {
                                    "operations" : 48867,
                                    "read_operations" : 2,
                                    "write_operations" : 48865,
                                    "read_kilobytes" : 8,
                                    "write_kilobytes" : 197588
                                }
                            }
                        },
                        "http" : {
                            "current_open" : 0,
                            "total_opened" : 0
                        },
                        "breakers" : {
                            "request" : {
                                "limit_size_in_bytes" : 5143501209,
                                "limit_size" : "4.7gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "fielddata" : {
                                "limit_size_in_bytes" : 3429000806,
                                "limit_size" : "3.1gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.03,
                                "tripped" : 0
                            },
                            "in_flight_requests" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 2.0,
                                "tripped" : 0
                            },
                            "accounting" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 29344,
                                "estimated_size" : "28.6kb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "parent" : {
                                "limit_size_in_bytes" : 8143876915,
                                "limit_size" : "7.5gb",
                                "estimated_size_in_bytes" : 894260408,
                                "estimated_size" : "852.8mb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            }
                        },
                        "script" : {
                            "compilations" : 0,
                            "cache_evictions" : 0,
                            "compilation_limit_triggered" : 0
                        },
                        "discovery" : {
                            "cluster_state_queue" : {
                                "total" : 0,
                                "pending" : 0,
                                "committed" : 0
                            },
                            "published_cluster_states" : {
                                "full_states" : 2,
                                "incompatible_diffs" : 0,
                                "compatible_diffs" : 1655
                            }
                        },
                        "ingest" : {
                            "total" : {
                                "count" : 0,
                                "time_in_millis" : 0,
                                "current" : 0,
                                "failed" : 0
                            },
                            "pipelines" : { }
                        },
                        "adaptive_selection" : {
                            "8dqNgo-KRYGIzJ2xsgbfEw" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 9660684,
                                "avg_response_time_ns" : 2687566,
                                "rank" : "2.7"
                            },
                            "3g-J2dUQS_m2su_yvnqRUA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 761374,
                                "avg_response_time_ns" : 2167628,
                                "rank" : "2.2"
                            },
                            "Gh_8fOzBTmaNnbQmAyd-rA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1392209,
                                "avg_response_time_ns" : 1800499,
                                "rank" : "1.8"
                            }
                        },
                        "script_cache" : {
                            "sum" : {
                                "compilations" : 0,
                                "cache_evictions" : 0,
                                "compilation_limit_triggered" : 0
                            }
                        },
                        "indexing_pressure" : {
                            "memory" : {
                                "current" : {
                                    "combined_coordinating_and_primary_in_bytes" : 0,
                                    "coordinating_in_bytes" : 0,
                                    "primary_in_bytes" : 0,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 0
                                },
                                "total" : {
                                    "combined_coordinating_and_primary_in_bytes" : 2381025,
                                    "coordinating_in_bytes" : 1271939,
                                    "primary_in_bytes" : 1596269,
                                    "replica_in_bytes" : 54564,
                                    "all_in_bytes" : 2435589,
                                    "coordinating_rejections" : 0,
                                    "primary_rejections" : 0,
                                    "replica_rejections" : 0
                                }
                            }
                        },
                        "shard_indexing_pressure" : {
                            "stats" : { },
                            "total_rejections_breakup_shadow_mode" : {
                                "node_limits" : 0,
                                "no_successful_request_limits" : 0,
                                "throughput_degradation_limits" : 0
                            },
                            "enabled" : 'true',
                            "enforced" : 'false'
                        }
                    },
                    "8dqNgo-KRYGIzJ2xsgbfEw" : {
                        "timestamp" : 1639459994260,
                        "name" : "c6051c73530bb3b94b30d92e2877867e",
                        "roles" : [ "data", "ingest", "master", "remote_cluster_client" ],
                        "indices" : {
                            "docs" : {
                                "count" : 196,
                                "deleted" : 4
                            },
                            "store" : {
                                "size_in_bytes" : 153617
                            },
                            "indexing" : {
                                "index_total" : 1653,
                                "index_time_in_millis" : 1526,
                                "index_current" : 0,
                                "index_failed" : 18,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 11,
                                "time_in_millis" : 93,
                                "exists_total" : 11,
                                "exists_time_in_millis" : 93,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 134,
                                "query_time_in_millis" : 118,
                                "query_current" : 0,
                                "fetch_total" : 68,
                                "fetch_time_in_millis" : 108,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 83,
                                "total_docs" : 10,
                                "total_size_in_bytes" : 53955,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 230686720
                            },
                            "refresh" : {
                                "total" : 92,
                                "total_time_in_millis" : 731,
                                "external_total" : 69,
                                "external_total_time_in_millis" : 687,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 11,
                                "periodic" : 0,
                                "total_time_in_millis" : 16
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 20,
                                "total_time_in_millis" : 6
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 9,
                                "memory_in_bytes" : 39140,
                                "terms_memory_in_bytes" : 26976,
                                "stored_fields_memory_in_bytes" : 4456,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 3392,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 4316,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 48,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 220,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 220,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 1078,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 1
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "os" : {
                            "timestamp" : 1639459994261,
                            "cpu" : {
                                "percent" : 2,
                                "load_average" : {
                                    "1m" : 0.07,
                                    "5m" : 0.19,
                                    "15m" : 0.23
                                }
                            },
                            "mem" : {
                                "total_in_bytes" : 16480382976,
                                "free_in_bytes" : 416960512,
                                "used_in_bytes" : 16063422464,
                                "free_percent" : 3,
                                "used_percent" : 97
                            },
                            "swap" : {
                                "total_in_bytes" : 2147479552,
                                "free_in_bytes" : 2147217408,
                                "used_in_bytes" : 262144
                            },
                            "cgroup" : {
                                "cpuacct" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "usage_nanos" : 31461420075888
                                },
                                "cpu" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "cfs_period_micros" : 100000,
                                    "cfs_quota_micros" : -1,
                                    "stat" : {
                                        "number_of_elapsed_periods" : 0,
                                        "number_of_times_throttled" : 0,
                                        "time_throttled_nanos" : 0
                                    }
                                },
                                "memory" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "limit_in_bytes" : "9223372036854771712",
                                    "usage_in_bytes" : "11988336640"
                                }
                            }
                        },
                        "process" : {
                            "timestamp" : 1639459994261,
                            "open_file_descriptors" : 1722,
                            "max_file_descriptors" : 1024000,
                            "cpu" : {
                                "percent" : 0,
                                "total_in_millis" : 7539800
                            },
                            "mem" : {
                                "total_virtual_in_bytes" : 12402905088
                            }
                        },
                        "jvm" : {
                            "timestamp" : 1639459994262,
                            "uptime_in_millis" : 476871234,
                            "mem" : {
                                "heap_used_in_bytes" : 1063432744,
                                "heap_used_percent" : 12,
                                "heap_committed_in_bytes" : 8572502016,
                                "heap_max_in_bytes" : 8572502016,
                                "non_heap_used_in_bytes" : 345156048,
                                "non_heap_committed_in_bytes" : 428797952,
                                "pools" : {
                                    "young" : {
                                        "used_in_bytes" : 138802096,
                                        "max_in_bytes" : 139591680,
                                        "peak_used_in_bytes" : 139591680,
                                        "peak_max_in_bytes" : 139591680
                                    },
                                    "survivor" : {
                                        "used_in_bytes" : 2165352,
                                        "max_in_bytes" : 17432576,
                                        "peak_used_in_bytes" : 17432576,
                                        "peak_max_in_bytes" : 17432576
                                    },
                                    "old" : {
                                        "used_in_bytes" : 922488776,
                                        "max_in_bytes" : 8415477760,
                                        "peak_used_in_bytes" : 922488776,
                                        "peak_max_in_bytes" : 8415477760
                                    }
                                }
                            },
                            "threads" : {
                                "count" : 146,
                                "peak_count" : 147
                            },
                            "gc" : {
                                "collectors" : {
                                    "young" : {
                                        "collection_count" : 15931,
                                        "collection_time_in_millis" : 873377
                                    },
                                    "old" : {
                                        "collection_count" : 4,
                                        "collection_time_in_millis" : 361
                                    }
                                }
                            },
                            "buffer_pools" : {
                                "mapped" : {
                                    "count" : 16,
                                    "used_in_bytes" : 131826,
                                    "total_capacity_in_bytes" : 131826
                                },
                                "direct" : {
                                    "count" : 124,
                                    "used_in_bytes" : 5819744,
                                    "total_capacity_in_bytes" : 5819742
                                }
                            },
                            "classes" : {
                                "current_loaded_count" : 38718,
                                "total_loaded_count" : 38719,
                                "total_unloaded_count" : 1
                            }
                        },
                        "thread_pool" : {
                            "ad-threadpool" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "analyze" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_started" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_store" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "flush" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 10
                            },
                            "force_merge" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 8
                            },
                            "generic" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 157065
                            },
                            "get" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 11
                            },
                            "listener" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "management" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 1,
                                "rejected" : 0,
                                "largest" : 5,
                                "completed" : 593330
                            },
                            "open_distro_job_scheduler" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "refresh" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1899220
                            },
                            "search" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 250
                            },
                            "search_throttled" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "snapshot" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1337
                            },
                            "snapshot_segments" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 50
                            },
                            "snapshot_shards" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 399
                            },
                            "sql-worker" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "ultrawarm_migration" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "warmer" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1
                            },
                            "write" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 78
                            }
                        },
                        "fs" : {
                            "timestamp" : 1639459994262,
                            "total" : {
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528395264,
                                "available_in_bytes" : 10511618048
                            },
                            "least_usage_estimate" : {
                                "total_in_bytes" : 10566782976,
                                "available_in_bytes" : 10511618048,
                                "used_disk_percent" : 0.5220598182558973
                            },
                            "most_usage_estimate" : {
                                "total_in_bytes" : 10566782976,
                                "available_in_bytes" : 10511618048,
                                "used_disk_percent" : 0.5220598182558973
                            },
                            "data" : [ {
                                "type" : "ext4",
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528395264,
                                "available_in_bytes" : 10511618048
                            } ],
                            "io_stats" : {
                                "devices" : [ {
                                    "device_name" : "dm-0",
                                    "operations" : 48493,
                                    "read_operations" : 1,
                                    "write_operations" : 48492,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 195160
                                } ],
                                "total" : {
                                    "operations" : 48493,
                                    "read_operations" : 1,
                                    "write_operations" : 48492,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 195160
                                }
                            }
                        },
                        "http" : {
                            "current_open" : 0,
                            "total_opened" : 0
                        },
                        "breakers" : {
                            "request" : {
                                "limit_size_in_bytes" : 5143501209,
                                "limit_size" : "4.7gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "fielddata" : {
                                "limit_size_in_bytes" : 3429000806,
                                "limit_size" : "3.1gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.03,
                                "tripped" : 0
                            },
                            "in_flight_requests" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 1416,
                                "estimated_size" : "1.3kb",
                                "overhead" : 2.0,
                                "tripped" : 0
                            },
                            "accounting" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 38948,
                                "estimated_size" : "38kb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "parent" : {
                                "limit_size_in_bytes" : 8143876915,
                                "limit_size" : "7.5gb",
                                "estimated_size_in_bytes" : 1063504312,
                                "estimated_size" : "1014.2mb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            }
                        },
                        "script" : {
                            "compilations" : 0,
                            "cache_evictions" : 0,
                            "compilation_limit_triggered" : 0
                        },
                        "discovery" : {
                            "cluster_state_queue" : {
                                "total" : 0,
                                "pending" : 0,
                                "committed" : 0
                            },
                            "published_cluster_states" : {
                                "full_states" : 2,
                                "incompatible_diffs" : 0,
                                "compatible_diffs" : 1655
                            }
                        },
                        "ingest" : {
                            "total" : {
                                "count" : 0,
                                "time_in_millis" : 0,
                                "current" : 0,
                                "failed" : 0
                            },
                            "pipelines" : { }
                        },
                        "adaptive_selection" : {
                            "8dqNgo-KRYGIzJ2xsgbfEw" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1085299,
                                "avg_response_time_ns" : 8921949,
                                "rank" : "8.9"
                            },
                            "3g-J2dUQS_m2su_yvnqRUA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 560096,
                                "avg_response_time_ns" : 2817684,
                                "rank" : "2.8"
                            },
                            "Gh_8fOzBTmaNnbQmAyd-rA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1285417,
                                "avg_response_time_ns" : 2861637,
                                "rank" : "2.9"
                            }
                        },
                        "script_cache" : {
                            "sum" : {
                                "compilations" : 0,
                                "cache_evictions" : 0,
                                "compilation_limit_triggered" : 0
                            }
                        },
                        "indexing_pressure" : {
                            "memory" : {
                                "current" : {
                                    "combined_coordinating_and_primary_in_bytes" : 0,
                                    "coordinating_in_bytes" : 0,
                                    "primary_in_bytes" : 0,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 0
                                },
                                "total" : {
                                    "combined_coordinating_and_primary_in_bytes" : 1862276,
                                    "coordinating_in_bytes" : 1268361,
                                    "primary_in_bytes" : 878293,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 1862276,
                                    "coordinating_rejections" : 0,
                                    "primary_rejections" : 0,
                                    "replica_rejections" : 0
                                }
                            }
                        },
                        "shard_indexing_pressure" : {
                            "stats" : { },
                            "total_rejections_breakup_shadow_mode" : {
                                "node_limits" : 0,
                                "no_successful_request_limits" : 0,
                                "throughput_degradation_limits" : 0
                            },
                            "enabled" : 'true',
                            "enforced" : 'false'
                        }
                    }
                }
            }
            nodes_stats = node_stas_value["nodes"].values()
        except BaseException:
            self.logger.exception("Could not retrieve nodes stats")
            nodes_stats = []
        try:
            nodes_info = self.client.nodes.info(node_id="_all")["nodes"].values()
        except BaseException:
            self.logger.exception("Could not retrieve nodes info")
            nodes_info = []

        for node in nodes_stats:
            node_name = node["name"]
            host = node.get("host", "unknown")
            self.metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "node_name", node_name)
            self.metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, "host_name", host)

        for node in nodes_info:
            node_name = node["name"]
            self.store_node_info(node_name, "os_name", node, ["os", "name"])
            self.store_node_info(node_name, "os_version", node, ["os", "version"])
            self.store_node_info(node_name, "cpu_logical_cores", node, ["os", "available_processors"])
            self.store_node_info(node_name, "jvm_vendor", node, ["jvm", "vm_vendor"])
            self.store_node_info(node_name, "jvm_version", node, ["jvm", "version"])

        store_plugin_metadata(self.metrics_store, nodes_info)
        store_node_attribute_metadata(self.metrics_store, nodes_info)

    def store_node_info(self, node_name, metric_key, node, path):
        self.metrics_store.add_meta_info(metrics.MetaInfoScope.node, node_name, metric_key, extract_value(node, path))


class JvmStatsSummary(InternalTelemetryDevice):
    """
    Gathers a summary of various JVM statistics during the whole test execution.
    """
    def __init__(self, client, metrics_store):
        super().__init__()
        self.metrics_store = metrics_store
        self.client = client
        self.jvm_stats_per_node = {}

    def on_benchmark_start(self):
        self.logger.info("JvmStatsSummary on benchmark start")
        self.jvm_stats_per_node = self.jvm_stats()

    def on_benchmark_stop(self):
        jvm_stats_at_end = self.jvm_stats()
        total_old_gen_collection_time = 0
        total_old_gen_collection_count = 0
        total_young_gen_collection_time = 0
        total_young_gen_collection_count = 0

        for node_name, jvm_stats_end in jvm_stats_at_end.items():
            if node_name in self.jvm_stats_per_node:
                jvm_stats_start = self.jvm_stats_per_node[node_name]
                young_gc_time = max(jvm_stats_end["young_gc_time"] - jvm_stats_start["young_gc_time"], 0)
                young_gc_count = max(jvm_stats_end["young_gc_count"] - jvm_stats_start["young_gc_count"], 0)
                old_gc_time = max(jvm_stats_end["old_gc_time"] - jvm_stats_start["old_gc_time"], 0)
                old_gc_count = max(jvm_stats_end["old_gc_count"] - jvm_stats_start["old_gc_count"], 0)

                total_young_gen_collection_time += young_gc_time
                total_young_gen_collection_count += young_gc_count
                total_old_gen_collection_time += old_gc_time
                total_old_gen_collection_count += old_gc_count

                self.metrics_store.put_value_node_level(node_name, "node_young_gen_gc_time", young_gc_time, "ms")
                self.metrics_store.put_value_node_level(node_name, "node_young_gen_gc_count", young_gc_count)
                self.metrics_store.put_value_node_level(node_name, "node_old_gen_gc_time", old_gc_time, "ms")
                self.metrics_store.put_value_node_level(node_name, "node_old_gen_gc_count", old_gc_count)

                all_pool_stats = {
                    "name": "jvm_memory_pool_stats"
                }
                for pool_name, pool_stats in jvm_stats_end["pools"].items():
                    all_pool_stats[pool_name] = {
                        "peak_usage": pool_stats["peak"],
                        "unit": "byte"
                    }
                self.metrics_store.put_doc(all_pool_stats, level=MetaInfoScope.node, node_name=node_name)

            else:
                self.logger.warning("Cannot determine JVM stats for [%s] (not in the cluster at the start of the benchmark).", node_name)

        self.metrics_store.put_value_cluster_level("node_total_young_gen_gc_time", total_young_gen_collection_time, "ms")
        self.metrics_store.put_value_cluster_level("node_total_young_gen_gc_count", total_young_gen_collection_count)
        self.metrics_store.put_value_cluster_level("node_total_old_gen_gc_time", total_old_gen_collection_time, "ms")
        self.metrics_store.put_value_cluster_level("node_total_old_gen_gc_count", total_old_gen_collection_count)

        self.jvm_stats_per_node = None

    def jvm_stats(self):
        self.logger.debug("Gathering JVM stats")
        jvm_stats = {}
        # pylint: disable=import-outside-toplevel
        import elasticsearch
        try:
            stats = {
                "_nodes" : {
                    "total" : 3,
                    "successful" : 3,
                    "failed" : 0
                },
                "cluster_name" : "397869430111:juno-mcm",
                "nodes" : {
                    "3g-J2dUQS_m2su_yvnqRUA" : {
                        "timestamp" : 1639459994258,
                        "name" : "29f760e1a8adb54ed33d8eab83248fa0",
                        "roles" : [ "data", "ingest", "master", "remote_cluster_client" ],
                        "indices" : {
                            "docs" : {
                                "count" : 404,
                                "deleted" : 3
                            },
                            "store" : {
                                "size_in_bytes" : 185622
                            },
                            "indexing" : {
                                "index_total" : 3184,
                                "index_time_in_millis" : 2130,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 18,
                                "time_in_millis" : 69,
                                "exists_total" : 18,
                                "exists_time_in_millis" : 69,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 67,
                                "query_time_in_millis" : 34,
                                "query_current" : 0,
                                "fetch_total" : 1,
                                "fetch_time_in_millis" : 2,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 73,
                                "total_docs" : 10,
                                "total_size_in_bytes" : 53955,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 377487360
                            },
                            "refresh" : {
                                "total" : 106,
                                "total_time_in_millis" : 441,
                                "external_total" : 87,
                                "external_total_time_in_millis" : 418,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 16,
                                "periodic" : 0,
                                "total_time_in_millis" : 16
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 20,
                                "total_time_in_millis" : 4
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 5,
                                "memory_in_bytes" : 27524,
                                "terms_memory_in_bytes" : 15920,
                                "stored_fields_memory_in_bytes" : 2504,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 2112,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 6988,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 0,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 220,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 220,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 1078,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 1
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "os" : {
                            "timestamp" : 1639459994258,
                            "cpu" : {
                                "percent" : 3,
                                "load_average" : {
                                    "1m" : 0.04,
                                    "5m" : 0.07,
                                    "15m" : 0.11
                                }
                            },
                            "mem" : {
                                "total_in_bytes" : 16480382976,
                                "free_in_bytes" : 412884992,
                                "used_in_bytes" : 16067497984,
                                "free_percent" : 3,
                                "used_percent" : 97
                            },
                            "swap" : {
                                "total_in_bytes" : 2147479552,
                                "free_in_bytes" : 2146955264,
                                "used_in_bytes" : 524288
                            },
                            "cgroup" : {
                                "cpuacct" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "usage_nanos" : 27963478780746
                                },
                                "cpu" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "cfs_period_micros" : 100000,
                                    "cfs_quota_micros" : -1,
                                    "stat" : {
                                        "number_of_elapsed_periods" : 0,
                                        "number_of_times_throttled" : 0,
                                        "time_throttled_nanos" : 0
                                    }
                                },
                                "memory" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "limit_in_bytes" : "9223372036854771712",
                                    "usage_in_bytes" : "11649228800"
                                }
                            }
                        },
                        "process" : {
                            "timestamp" : 1639459994259,
                            "open_file_descriptors" : 1724,
                            "max_file_descriptors" : 1024000,
                            "cpu" : {
                                "percent" : 0,
                                "total_in_millis" : 6614350
                            },
                            "mem" : {
                                "total_virtual_in_bytes" : 12349054976
                            }
                        },
                        "jvm" : {
                            "timestamp" : 1639459994260,
                            "uptime_in_millis" : 476870141,
                            "mem" : {
                                "heap_used_in_bytes" : 894324928,
                                "heap_used_percent" : 10,
                                "heap_committed_in_bytes" : 8572502016,
                                "heap_max_in_bytes" : 8572502016,
                                "non_heap_used_in_bytes" : 333200872,
                                "non_heap_committed_in_bytes" : 395907072,
                                "pools" : {
                                    "young" : {
                                        "used_in_bytes" : 52322208,
                                        "max_in_bytes" : 139591680,
                                        "peak_used_in_bytes" : 139591680,
                                        "peak_max_in_bytes" : 139591680
                                    },
                                    "survivor" : {
                                        "used_in_bytes" : 2067384,
                                        "max_in_bytes" : 17432576,
                                        "peak_used_in_bytes" : 17432576,
                                        "peak_max_in_bytes" : 17432576
                                    },
                                    "old" : {
                                        "used_in_bytes" : 839935336,
                                        "max_in_bytes" : 8415477760,
                                        "peak_used_in_bytes" : 839935336,
                                        "peak_max_in_bytes" : 8415477760
                                    }
                                }
                            },
                            "threads" : {
                                "count" : 145,
                                "peak_count" : 145
                            },
                            "gc" : {
                                "collectors" : {
                                    "young" : {
                                        "collection_count" : 15161,
                                        "collection_time_in_millis" : 816018
                                    },
                                    "old" : {
                                        "collection_count" : 4,
                                        "collection_time_in_millis" : 332
                                    }
                                }
                            },
                            "buffer_pools" : {
                                "mapped" : {
                                    "count" : 11,
                                    "used_in_bytes" : 162531,
                                    "total_capacity_in_bytes" : 162531
                                },
                                "direct" : {
                                    "count" : 124,
                                    "used_in_bytes" : 5763821,
                                    "total_capacity_in_bytes" : 5763819
                                }
                            },
                            "classes" : {
                                "current_loaded_count" : 37833,
                                "total_loaded_count" : 37837,
                                "total_unloaded_count" : 4
                            }
                        },
                        "thread_pool" : {
                            "ad-threadpool" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "analyze" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_started" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_store" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "flush" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 16
                            },
                            "force_merge" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 8
                            },
                            "generic" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 15,
                                "completed" : 127991
                            },
                            "get" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 10
                            },
                            "listener" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "management" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 1,
                                "rejected" : 0,
                                "largest" : 5,
                                "completed" : 545500
                            },
                            "open_distro_job_scheduler" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "refresh" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1424007
                            },
                            "search" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 117
                            },
                            "search_throttled" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "snapshot" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1
                            },
                            "snapshot_segments" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 28
                            },
                            "snapshot_shards" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 262
                            },
                            "sql-worker" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "ultrawarm_migration" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "warmer" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "write" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 63
                            }
                        },
                        "fs" : {
                            "timestamp" : 1639459994260,
                            "total" : {
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528403456,
                                "available_in_bytes" : 10511626240
                            },
                            "data" : [ {
                                "type" : "ext4",
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528403456,
                                "available_in_bytes" : 10511626240
                            } ],
                            "io_stats" : {
                                "devices" : [ {
                                    "device_name" : "dm-0",
                                    "operations" : 46888,
                                    "read_operations" : 1,
                                    "write_operations" : 46887,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 189620
                                } ],
                                "total" : {
                                    "operations" : 46888,
                                    "read_operations" : 1,
                                    "write_operations" : 46887,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 189620
                                }
                            }
                        },
                        "http" : {
                            "current_open" : 0,
                            "total_opened" : 0
                        },
                        "breakers" : {
                            "request" : {
                                "limit_size_in_bytes" : 5143501209,
                                "limit_size" : "4.7gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "fielddata" : {
                                "limit_size_in_bytes" : 3429000806,
                                "limit_size" : "3.1gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.03,
                                "tripped" : 0
                            },
                            "in_flight_requests" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 1416,
                                "estimated_size" : "1.3kb",
                                "overhead" : 2.0,
                                "tripped" : 0
                            },
                            "accounting" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 27524,
                                "estimated_size" : "26.8kb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "parent" : {
                                "limit_size_in_bytes" : 8143876915,
                                "limit_size" : "7.5gb",
                                "estimated_size_in_bytes" : 894393656,
                                "estimated_size" : "852.9mb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            }
                        },
                        "script" : {
                            "compilations" : 0,
                            "cache_evictions" : 0,
                            "compilation_limit_triggered" : 0
                        },
                        "discovery" : {
                            "cluster_state_queue" : {
                                "total" : 0,
                                "pending" : 0,
                                "committed" : 0
                            },
                            "published_cluster_states" : {
                                "full_states" : 1,
                                "incompatible_diffs" : 0,
                                "compatible_diffs" : 1654
                            }
                        },
                        "ingest" : {
                            "total" : {
                                "count" : 0,
                                "time_in_millis" : 0,
                                "current" : 0,
                                "failed" : 0
                            },
                            "pipelines" : { }
                        },
                        "adaptive_selection" : {
                            "8dqNgo-KRYGIzJ2xsgbfEw" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1200061,
                                "avg_response_time_ns" : 2868875,
                                "rank" : "2.9"
                            },
                            "3g-J2dUQS_m2su_yvnqRUA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 539218,
                                "avg_response_time_ns" : 2301522,
                                "rank" : "2.3"
                            },
                            "Gh_8fOzBTmaNnbQmAyd-rA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1194916,
                                "avg_response_time_ns" : 2677770,
                                "rank" : "2.7"
                            }
                        },
                        "script_cache" : {
                            "sum" : {
                                "compilations" : 0,
                                "cache_evictions" : 0,
                                "compilation_limit_triggered" : 0
                            }
                        },
                        "indexing_pressure" : {
                            "memory" : {
                                "current" : {
                                    "combined_coordinating_and_primary_in_bytes" : 0,
                                    "coordinating_in_bytes" : 0,
                                    "primary_in_bytes" : 0,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 0
                                },
                                "total" : {
                                    "combined_coordinating_and_primary_in_bytes" : 2515084,
                                    "coordinating_in_bytes" : 1517909,
                                    "primary_in_bytes" : 1583647,
                                    "replica_in_bytes" : 54564,
                                    "all_in_bytes" : 2569648,
                                    "coordinating_rejections" : 0,
                                    "primary_rejections" : 0,
                                    "replica_rejections" : 0
                                }
                            }
                        },
                        "shard_indexing_pressure" : {
                            "stats" : { },
                            "total_rejections_breakup_shadow_mode" : {
                                "node_limits" : 0,
                                "no_successful_request_limits" : 0,
                                "throughput_degradation_limits" : 0
                            },
                            "enabled" : 'true',
                            "enforced" : 'false'
                        }
                    },
                    "Gh_8fOzBTmaNnbQmAyd-rA" : {
                        "timestamp" : 1639459994257,
                        "name" : "84c74d57fefcca221479642d7916c1f2",
                        "roles" : [ "data", "ingest", "master", "remote_cluster_client" ],
                        "indices" : {
                            "docs" : {
                                "count" : 429,
                                "deleted" : 3
                            },
                            "store" : {
                                "size_in_bytes" : 199274
                            },
                            "indexing" : {
                                "index_total" : 3203,
                                "index_time_in_millis" : 2154,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 27,
                                "time_in_millis" : 150,
                                "exists_total" : 27,
                                "exists_time_in_millis" : 150,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 68,
                                "query_time_in_millis" : 68,
                                "query_current" : 0,
                                "fetch_total" : 68,
                                "fetch_time_in_millis" : 93,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 65,
                                "total_docs" : 10,
                                "total_size_in_bytes" : 54027,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 377487360
                            },
                            "refresh" : {
                                "total" : 113,
                                "total_time_in_millis" : 471,
                                "external_total" : 89,
                                "external_total_time_in_millis" : 468,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 17,
                                "periodic" : 0,
                                "total_time_in_millis" : 78
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 23,
                                "total_time_in_millis" : 15
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 6,
                                "memory_in_bytes" : 29344,
                                "terms_memory_in_bytes" : 16704,
                                "stored_fields_memory_in_bytes" : 2992,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 2112,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 7536,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 48,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 220,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 220,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "os" : {
                            "timestamp" : 1639459994259,
                            "cpu" : {
                                "percent" : 3,
                                "load_average" : {
                                    "1m" : 0.2,
                                    "5m" : 0.19,
                                    "15m" : 0.16
                                }
                            },
                            "mem" : {
                                "total_in_bytes" : 16480382976,
                                "free_in_bytes" : 470929408,
                                "used_in_bytes" : 16009453568,
                                "free_percent" : 3,
                                "used_percent" : 97
                            },
                            "swap" : {
                                "total_in_bytes" : 2147479552,
                                "free_in_bytes" : 2146955264,
                                "used_in_bytes" : 524288
                            },
                            "cgroup" : {
                                "cpuacct" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "usage_nanos" : 28451058701403
                                },
                                "cpu" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "cfs_period_micros" : 100000,
                                    "cfs_quota_micros" : -1,
                                    "stat" : {
                                        "number_of_elapsed_periods" : 0,
                                        "number_of_times_throttled" : 0,
                                        "time_throttled_nanos" : 0
                                    }
                                },
                                "memory" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "limit_in_bytes" : "9223372036854771712",
                                    "usage_in_bytes" : "11654156288"
                                }
                            }
                        },
                        "process" : {
                            "timestamp" : 1639459994259,
                            "open_file_descriptors" : 1720,
                            "max_file_descriptors" : 1024000,
                            "cpu" : {
                                "percent" : 0,
                                "total_in_millis" : 6645250
                            },
                            "mem" : {
                                "total_virtual_in_bytes" : 12330627072
                            }
                        },
                        "jvm" : {
                            "timestamp" : 1639459994261,
                            "uptime_in_millis" : 476882987,
                            "mem" : {
                                "heap_used_in_bytes" : 894191304,
                                "heap_used_percent" : 10,
                                "heap_committed_in_bytes" : 8572502016,
                                "heap_max_in_bytes" : 8572502016,
                                "non_heap_used_in_bytes" : 333612432,
                                "non_heap_committed_in_bytes" : 421584896,
                                "pools" : {
                                    "young" : {
                                        "used_in_bytes" : 65466784,
                                        "max_in_bytes" : 139591680,
                                        "peak_used_in_bytes" : 139591680,
                                        "peak_max_in_bytes" : 139591680
                                    },
                                    "survivor" : {
                                        "used_in_bytes" : 1788432,
                                        "max_in_bytes" : 17432576,
                                        "peak_used_in_bytes" : 17432576,
                                        "peak_max_in_bytes" : 17432576
                                    },
                                    "old" : {
                                        "used_in_bytes" : 826950920,
                                        "max_in_bytes" : 8415477760,
                                        "peak_used_in_bytes" : 826950920,
                                        "peak_max_in_bytes" : 8415477760
                                    }
                                }
                            },
                            "threads" : {
                                "count" : 149,
                                "peak_count" : 149
                            },
                            "gc" : {
                                "collectors" : {
                                    "young" : {
                                        "collection_count" : 15182,
                                        "collection_time_in_millis" : 844088
                                    },
                                    "old" : {
                                        "collection_count" : 4,
                                        "collection_time_in_millis" : 328
                                    }
                                }
                            },
                            "buffer_pools" : {
                                "mapped" : {
                                    "count" : 12,
                                    "used_in_bytes" : 181134,
                                    "total_capacity_in_bytes" : 181134
                                },
                                "direct" : {
                                    "count" : 124,
                                    "used_in_bytes" : 5723993,
                                    "total_capacity_in_bytes" : 5723991
                                }
                            },
                            "classes" : {
                                "current_loaded_count" : 38081,
                                "total_loaded_count" : 38081,
                                "total_unloaded_count" : 0
                            }
                        },
                        "thread_pool" : {
                            "ad-threadpool" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "analyze" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_started" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_store" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "flush" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 16
                            },
                            "force_merge" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 8
                            },
                            "generic" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 17,
                                "completed" : 129829
                            },
                            "get" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 11
                            },
                            "listener" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "management" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 1,
                                "rejected" : 0,
                                "largest" : 5,
                                "completed" : 545404
                            },
                            "open_distro_job_scheduler" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "refresh" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1422509
                            },
                            "search" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 171
                            },
                            "search_throttled" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "snapshot" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1
                            },
                            "snapshot_segments" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 32
                            },
                            "snapshot_shards" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 394
                            },
                            "sql-worker" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "ultrawarm_migration" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "warmer" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "write" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 64
                            }
                        },
                        "fs" : {
                            "timestamp" : 1639459994261,
                            "total" : {
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528399360,
                                "available_in_bytes" : 10511622144
                            },
                            "data" : [ {
                                "type" : "ext4",
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528399360,
                                "available_in_bytes" : 10511622144
                            } ],
                            "io_stats" : {
                                "devices" : [ {
                                    "device_name" : "dm-0",
                                    "operations" : 48867,
                                    "read_operations" : 2,
                                    "write_operations" : 48865,
                                    "read_kilobytes" : 8,
                                    "write_kilobytes" : 197588
                                } ],
                                "total" : {
                                    "operations" : 48867,
                                    "read_operations" : 2,
                                    "write_operations" : 48865,
                                    "read_kilobytes" : 8,
                                    "write_kilobytes" : 197588
                                }
                            }
                        },
                        "http" : {
                            "current_open" : 0,
                            "total_opened" : 0
                        },
                        "breakers" : {
                            "request" : {
                                "limit_size_in_bytes" : 5143501209,
                                "limit_size" : "4.7gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "fielddata" : {
                                "limit_size_in_bytes" : 3429000806,
                                "limit_size" : "3.1gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.03,
                                "tripped" : 0
                            },
                            "in_flight_requests" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 2.0,
                                "tripped" : 0
                            },
                            "accounting" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 29344,
                                "estimated_size" : "28.6kb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "parent" : {
                                "limit_size_in_bytes" : 8143876915,
                                "limit_size" : "7.5gb",
                                "estimated_size_in_bytes" : 894260408,
                                "estimated_size" : "852.8mb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            }
                        },
                        "script" : {
                            "compilations" : 0,
                            "cache_evictions" : 0,
                            "compilation_limit_triggered" : 0
                        },
                        "discovery" : {
                            "cluster_state_queue" : {
                                "total" : 0,
                                "pending" : 0,
                                "committed" : 0
                            },
                            "published_cluster_states" : {
                                "full_states" : 2,
                                "incompatible_diffs" : 0,
                                "compatible_diffs" : 1655
                            }
                        },
                        "ingest" : {
                            "total" : {
                                "count" : 0,
                                "time_in_millis" : 0,
                                "current" : 0,
                                "failed" : 0
                            },
                            "pipelines" : { }
                        },
                        "adaptive_selection" : {
                            "8dqNgo-KRYGIzJ2xsgbfEw" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 9660684,
                                "avg_response_time_ns" : 2687566,
                                "rank" : "2.7"
                            },
                            "3g-J2dUQS_m2su_yvnqRUA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 761374,
                                "avg_response_time_ns" : 2167628,
                                "rank" : "2.2"
                            },
                            "Gh_8fOzBTmaNnbQmAyd-rA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1392209,
                                "avg_response_time_ns" : 1800499,
                                "rank" : "1.8"
                            }
                        },
                        "script_cache" : {
                            "sum" : {
                                "compilations" : 0,
                                "cache_evictions" : 0,
                                "compilation_limit_triggered" : 0
                            }
                        },
                        "indexing_pressure" : {
                            "memory" : {
                                "current" : {
                                    "combined_coordinating_and_primary_in_bytes" : 0,
                                    "coordinating_in_bytes" : 0,
                                    "primary_in_bytes" : 0,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 0
                                },
                                "total" : {
                                    "combined_coordinating_and_primary_in_bytes" : 2381025,
                                    "coordinating_in_bytes" : 1271939,
                                    "primary_in_bytes" : 1596269,
                                    "replica_in_bytes" : 54564,
                                    "all_in_bytes" : 2435589,
                                    "coordinating_rejections" : 0,
                                    "primary_rejections" : 0,
                                    "replica_rejections" : 0
                                }
                            }
                        },
                        "shard_indexing_pressure" : {
                            "stats" : { },
                            "total_rejections_breakup_shadow_mode" : {
                                "node_limits" : 0,
                                "no_successful_request_limits" : 0,
                                "throughput_degradation_limits" : 0
                            },
                            "enabled" : 'true',
                            "enforced" : 'false'
                        }
                    },
                    "8dqNgo-KRYGIzJ2xsgbfEw" : {
                        "timestamp" : 1639459994260,
                        "name" : "c6051c73530bb3b94b30d92e2877867e",
                        "roles" : [ "data", "ingest", "master", "remote_cluster_client" ],
                        "indices" : {
                            "docs" : {
                                "count" : 196,
                                "deleted" : 4
                            },
                            "store" : {
                                "size_in_bytes" : 153617
                            },
                            "indexing" : {
                                "index_total" : 1653,
                                "index_time_in_millis" : 1526,
                                "index_current" : 0,
                                "index_failed" : 18,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 11,
                                "time_in_millis" : 93,
                                "exists_total" : 11,
                                "exists_time_in_millis" : 93,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 134,
                                "query_time_in_millis" : 118,
                                "query_current" : 0,
                                "fetch_total" : 68,
                                "fetch_time_in_millis" : 108,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 83,
                                "total_docs" : 10,
                                "total_size_in_bytes" : 53955,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 230686720
                            },
                            "refresh" : {
                                "total" : 92,
                                "total_time_in_millis" : 731,
                                "external_total" : 69,
                                "external_total_time_in_millis" : 687,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 11,
                                "periodic" : 0,
                                "total_time_in_millis" : 16
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 20,
                                "total_time_in_millis" : 6
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 9,
                                "memory_in_bytes" : 39140,
                                "terms_memory_in_bytes" : 26976,
                                "stored_fields_memory_in_bytes" : 4456,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 3392,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 4316,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 48,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 220,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 220,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 1078,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 1
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "os" : {
                            "timestamp" : 1639459994261,
                            "cpu" : {
                                "percent" : 2,
                                "load_average" : {
                                    "1m" : 0.07,
                                    "5m" : 0.19,
                                    "15m" : 0.23
                                }
                            },
                            "mem" : {
                                "total_in_bytes" : 16480382976,
                                "free_in_bytes" : 416960512,
                                "used_in_bytes" : 16063422464,
                                "free_percent" : 3,
                                "used_percent" : 97
                            },
                            "swap" : {
                                "total_in_bytes" : 2147479552,
                                "free_in_bytes" : 2147217408,
                                "used_in_bytes" : 262144
                            },
                            "cgroup" : {
                                "cpuacct" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "usage_nanos" : 31461420075888
                                },
                                "cpu" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "cfs_period_micros" : 100000,
                                    "cfs_quota_micros" : -1,
                                    "stat" : {
                                        "number_of_elapsed_periods" : 0,
                                        "number_of_times_throttled" : 0,
                                        "time_throttled_nanos" : 0
                                    }
                                },
                                "memory" : {
                                    "control_group" : "/system.slice/apollo-init.service",
                                    "limit_in_bytes" : "9223372036854771712",
                                    "usage_in_bytes" : "11988336640"
                                }
                            }
                        },
                        "process" : {
                            "timestamp" : 1639459994261,
                            "open_file_descriptors" : 1722,
                            "max_file_descriptors" : 1024000,
                            "cpu" : {
                                "percent" : 0,
                                "total_in_millis" : 7539800
                            },
                            "mem" : {
                                "total_virtual_in_bytes" : 12402905088
                            }
                        },
                        "jvm" : {
                            "timestamp" : 1639459994262,
                            "uptime_in_millis" : 476871234,
                            "mem" : {
                                "heap_used_in_bytes" : 1063432744,
                                "heap_used_percent" : 12,
                                "heap_committed_in_bytes" : 8572502016,
                                "heap_max_in_bytes" : 8572502016,
                                "non_heap_used_in_bytes" : 345156048,
                                "non_heap_committed_in_bytes" : 428797952,
                                "pools" : {
                                    "young" : {
                                        "used_in_bytes" : 138802096,
                                        "max_in_bytes" : 139591680,
                                        "peak_used_in_bytes" : 139591680,
                                        "peak_max_in_bytes" : 139591680
                                    },
                                    "survivor" : {
                                        "used_in_bytes" : 2165352,
                                        "max_in_bytes" : 17432576,
                                        "peak_used_in_bytes" : 17432576,
                                        "peak_max_in_bytes" : 17432576
                                    },
                                    "old" : {
                                        "used_in_bytes" : 922488776,
                                        "max_in_bytes" : 8415477760,
                                        "peak_used_in_bytes" : 922488776,
                                        "peak_max_in_bytes" : 8415477760
                                    }
                                }
                            },
                            "threads" : {
                                "count" : 146,
                                "peak_count" : 147
                            },
                            "gc" : {
                                "collectors" : {
                                    "young" : {
                                        "collection_count" : 15931,
                                        "collection_time_in_millis" : 873377
                                    },
                                    "old" : {
                                        "collection_count" : 4,
                                        "collection_time_in_millis" : 361
                                    }
                                }
                            },
                            "buffer_pools" : {
                                "mapped" : {
                                    "count" : 16,
                                    "used_in_bytes" : 131826,
                                    "total_capacity_in_bytes" : 131826
                                },
                                "direct" : {
                                    "count" : 124,
                                    "used_in_bytes" : 5819744,
                                    "total_capacity_in_bytes" : 5819742
                                }
                            },
                            "classes" : {
                                "current_loaded_count" : 38718,
                                "total_loaded_count" : 38719,
                                "total_unloaded_count" : 1
                            }
                        },
                        "thread_pool" : {
                            "ad-threadpool" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "analyze" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_started" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "fetch_shard_store" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 2
                            },
                            "flush" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 10
                            },
                            "force_merge" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 8
                            },
                            "generic" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 157065
                            },
                            "get" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 11
                            },
                            "listener" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "management" : {
                                "threads" : 5,
                                "queue" : 0,
                                "active" : 1,
                                "rejected" : 0,
                                "largest" : 5,
                                "completed" : 593330
                            },
                            "open_distro_job_scheduler" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "refresh" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1899220
                            },
                            "search" : {
                                "threads" : 4,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 4,
                                "completed" : 250
                            },
                            "search_throttled" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "snapshot" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1337
                            },
                            "snapshot_segments" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 50
                            },
                            "snapshot_shards" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 399
                            },
                            "sql-worker" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "ultrawarm_migration" : {
                                "threads" : 0,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 0,
                                "completed" : 0
                            },
                            "warmer" : {
                                "threads" : 1,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 1,
                                "completed" : 1
                            },
                            "write" : {
                                "threads" : 2,
                                "queue" : 0,
                                "active" : 0,
                                "rejected" : 0,
                                "largest" : 2,
                                "completed" : 78
                            }
                        },
                        "fs" : {
                            "timestamp" : 1639459994262,
                            "total" : {
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528395264,
                                "available_in_bytes" : 10511618048
                            },
                            "least_usage_estimate" : {
                                "total_in_bytes" : 10566782976,
                                "available_in_bytes" : 10511618048,
                                "used_disk_percent" : 0.5220598182558973
                            },
                            "most_usage_estimate" : {
                                "total_in_bytes" : 10566782976,
                                "available_in_bytes" : 10511618048,
                                "used_disk_percent" : 0.5220598182558973
                            },
                            "data" : [ {
                                "type" : "ext4",
                                "total_in_bytes" : 10566782976,
                                "free_in_bytes" : 10528395264,
                                "available_in_bytes" : 10511618048
                            } ],
                            "io_stats" : {
                                "devices" : [ {
                                    "device_name" : "dm-0",
                                    "operations" : 48493,
                                    "read_operations" : 1,
                                    "write_operations" : 48492,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 195160
                                } ],
                                "total" : {
                                    "operations" : 48493,
                                    "read_operations" : 1,
                                    "write_operations" : 48492,
                                    "read_kilobytes" : 4,
                                    "write_kilobytes" : 195160
                                }
                            }
                        },
                        "http" : {
                            "current_open" : 0,
                            "total_opened" : 0
                        },
                        "breakers" : {
                            "request" : {
                                "limit_size_in_bytes" : 5143501209,
                                "limit_size" : "4.7gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "fielddata" : {
                                "limit_size_in_bytes" : 3429000806,
                                "limit_size" : "3.1gb",
                                "estimated_size_in_bytes" : 0,
                                "estimated_size" : "0b",
                                "overhead" : 1.03,
                                "tripped" : 0
                            },
                            "in_flight_requests" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 1416,
                                "estimated_size" : "1.3kb",
                                "overhead" : 2.0,
                                "tripped" : 0
                            },
                            "accounting" : {
                                "limit_size_in_bytes" : 8572502016,
                                "limit_size" : "7.9gb",
                                "estimated_size_in_bytes" : 38948,
                                "estimated_size" : "38kb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            },
                            "parent" : {
                                "limit_size_in_bytes" : 8143876915,
                                "limit_size" : "7.5gb",
                                "estimated_size_in_bytes" : 1063504312,
                                "estimated_size" : "1014.2mb",
                                "overhead" : 1.0,
                                "tripped" : 0
                            }
                        },
                        "script" : {
                            "compilations" : 0,
                            "cache_evictions" : 0,
                            "compilation_limit_triggered" : 0
                        },
                        "discovery" : {
                            "cluster_state_queue" : {
                                "total" : 0,
                                "pending" : 0,
                                "committed" : 0
                            },
                            "published_cluster_states" : {
                                "full_states" : 2,
                                "incompatible_diffs" : 0,
                                "compatible_diffs" : 1655
                            }
                        },
                        "ingest" : {
                            "total" : {
                                "count" : 0,
                                "time_in_millis" : 0,
                                "current" : 0,
                                "failed" : 0
                            },
                            "pipelines" : { }
                        },
                        "adaptive_selection" : {
                            "8dqNgo-KRYGIzJ2xsgbfEw" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1085299,
                                "avg_response_time_ns" : 8921949,
                                "rank" : "8.9"
                            },
                            "3g-J2dUQS_m2su_yvnqRUA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 560096,
                                "avg_response_time_ns" : 2817684,
                                "rank" : "2.8"
                            },
                            "Gh_8fOzBTmaNnbQmAyd-rA" : {
                                "outgoing_searches" : 0,
                                "avg_queue_size" : 0,
                                "avg_service_time_ns" : 1285417,
                                "avg_response_time_ns" : 2861637,
                                "rank" : "2.9"
                            }
                        },
                        "script_cache" : {
                            "sum" : {
                                "compilations" : 0,
                                "cache_evictions" : 0,
                                "compilation_limit_triggered" : 0
                            }
                        },
                        "indexing_pressure" : {
                            "memory" : {
                                "current" : {
                                    "combined_coordinating_and_primary_in_bytes" : 0,
                                    "coordinating_in_bytes" : 0,
                                    "primary_in_bytes" : 0,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 0
                                },
                                "total" : {
                                    "combined_coordinating_and_primary_in_bytes" : 1862276,
                                    "coordinating_in_bytes" : 1268361,
                                    "primary_in_bytes" : 878293,
                                    "replica_in_bytes" : 0,
                                    "all_in_bytes" : 1862276,
                                    "coordinating_rejections" : 0,
                                    "primary_rejections" : 0,
                                    "replica_rejections" : 0
                                }
                            }
                        },
                        "shard_indexing_pressure" : {
                            "stats" : { },
                            "total_rejections_breakup_shadow_mode" : {
                                "node_limits" : 0,
                                "no_successful_request_limits" : 0,
                                "throughput_degradation_limits" : 0
                            },
                            "enabled" : 'true',
                            "enforced" : 'false'
                        }
                    }
                }
            }
        except elasticsearch.TransportError:
            self.logger.exception("Could not retrieve GC times.")
            return jvm_stats
        nodes = stats["nodes"]
        for node in nodes.values():
            node_name = node["name"]
            gc = node["jvm"]["gc"]["collectors"]
            old_gen_collection_time = gc["old"]["collection_time_in_millis"]
            old_gen_collection_count = gc["old"]["collection_count"]
            young_gen_collection_time = gc["young"]["collection_time_in_millis"]
            young_gen_collection_count = gc["young"]["collection_count"]
            jvm_stats[node_name] = {
                "young_gc_time": young_gen_collection_time,
                "young_gc_count": young_gen_collection_count,
                "old_gc_time": old_gen_collection_time,
                "old_gc_count": old_gen_collection_count,
                "pools": {}
            }
            pool_usage = node["jvm"]["mem"]["pools"]
            for pool_name, pool_stats in pool_usage.items():
                jvm_stats[node_name]["pools"][pool_name] = {
                    "peak": pool_stats["peak_used_in_bytes"]
                }
        return jvm_stats


class IndexStats(InternalTelemetryDevice):
    """
    Gathers statistics via the OpenSearch index stats API
    """
    def __init__(self, client, metrics_store):
        super().__init__()
        self.client = client
        self.metrics_store = metrics_store
        self.first_time = True

    def on_benchmark_start(self):
        # we only determine this value at the start of the benchmark. This is actually only useful for
        # the pipeline "benchmark-only" where we don't have control over the cluster and the user might not have restarted
        # the cluster so we can at least tell them.
        if self.first_time:
            for t in self.index_times(self.index_stats(), per_shard_stats=False):
                n = t["name"]
                v = t["value"]
                if t["value"] > 0:
                    console.warn("%s is %d ms indicating that the cluster is not in a defined clean state. Recorded index time "
                                 "metrics may be misleading." % (n, v), logger=self.logger)
            self.first_time = False

    def on_benchmark_stop(self):
        self.logger.info("Gathering indices stats for all primaries on benchmark stop.")
        index_stats = self.index_stats()
        # import json
        # self.logger.debug("Returned indices stats:\n%s", json.dumps(index_stats, indent=2))
        if "_all" not in index_stats or "primaries" not in index_stats["_all"]:
            return
        p = index_stats["_all"]["primaries"]
        # actually this is add_count
        self.add_metrics(self.extract_value(p, ["segments", "count"]), "segments_count")
        self.add_metrics(self.extract_value(p, ["segments", "memory_in_bytes"]), "segments_memory_in_bytes", "byte")

        for t in self.index_times(index_stats):
            self.metrics_store.put_doc(doc=t, level=metrics.MetaInfoScope.cluster)

        for ct in self.index_counts(index_stats):
            self.metrics_store.put_doc(doc=ct, level=metrics.MetaInfoScope.cluster)

        self.add_metrics(self.extract_value(p, ["segments", "doc_values_memory_in_bytes"]), "segments_doc_values_memory_in_bytes", "byte")
        self.add_metrics(self.extract_value(p, ["segments", "stored_fields_memory_in_bytes"]), "segments_stored_fields_memory_in_bytes",
                                            "byte")
        self.add_metrics(self.extract_value(p, ["segments", "terms_memory_in_bytes"]), "segments_terms_memory_in_bytes", "byte")
        self.add_metrics(self.extract_value(p, ["segments", "norms_memory_in_bytes"]), "segments_norms_memory_in_bytes", "byte")
        self.add_metrics(self.extract_value(p, ["segments", "points_memory_in_bytes"]), "segments_points_memory_in_bytes", "byte")
        self.add_metrics(self.extract_value(index_stats, ["_all", "total", "store", "size_in_bytes"]), "store_size_in_bytes", "byte")
        self.add_metrics(self.extract_value(index_stats, ["_all", "total", "translog", "size_in_bytes"]), "translog_size_in_bytes", "byte")

    def index_stats(self):
        # noinspection PyBroadException
        try:
            return {
                "_shards" : {
                    "total" : 12,
                    "successful" : 12,
                    "failed" : 0
                },
                "_all" : {
                    "primaries" : {
                        "docs" : {
                            "count" : 1010,
                            "deleted" : 4
                        },
                        "store" : {
                            "size_in_bytes" : 434703
                        },
                        "indexing" : {
                            "index_total" : 1014,
                            "index_time_in_millis" : 633,
                            "index_current" : 0,
                            "index_failed" : 18,
                            "delete_total" : 0,
                            "delete_time_in_millis" : 0,
                            "delete_current" : 0,
                            "noop_update_total" : 0,
                            "is_throttled" : 'false',
                            "throttle_time_in_millis" : 0
                        },
                        "get" : {
                            "total" : 9,
                            "time_in_millis" : 92,
                            "exists_total" : 9,
                            "exists_time_in_millis" : 92,
                            "missing_total" : 0,
                            "missing_time_in_millis" : 0,
                            "current" : 0
                        },
                        "search" : {
                            "open_contexts" : 0,
                            "query_total" : 135,
                            "query_time_in_millis" : 142,
                            "query_current" : 0,
                            "fetch_total" : 69,
                            "fetch_time_in_millis" : 122,
                            "fetch_current" : 0,
                            "scroll_total" : 0,
                            "scroll_time_in_millis" : 0,
                            "scroll_current" : 0,
                            "suggest_total" : 0,
                            "suggest_time_in_millis" : 0,
                            "suggest_current" : 0
                        },
                        "merges" : {
                            "current" : 0,
                            "current_docs" : 0,
                            "current_size_in_bytes" : 0,
                            "total" : 1,
                            "total_time_in_millis" : 83,
                            "total_docs" : 10,
                            "total_size_in_bytes" : 53955,
                            "total_stopped_time_in_millis" : 0,
                            "total_throttled_time_in_millis" : 0,
                            "total_auto_throttle_in_bytes" : 167772160
                        },
                        "refresh" : {
                            "total" : 80,
                            "total_time_in_millis" : 516,
                            "external_total" : 57,
                            "external_total_time_in_millis" : 500,
                            "listeners" : 0
                        },
                        "flush" : {
                            "total" : 8,
                            "periodic" : 0,
                            "total_time_in_millis" : 11
                        },
                        "warmer" : {
                            "current" : 0,
                            "total" : 30,
                            "total_time_in_millis" : 9
                        },
                        "query_cache" : {
                            "memory_size_in_bytes" : 0,
                            "total_count" : 0,
                            "hit_count" : 0,
                            "miss_count" : 0,
                            "cache_size" : 0,
                            "cache_count" : 0,
                            "evictions" : 0
                        },
                        "fielddata" : {
                            "memory_size_in_bytes" : 0,
                            "evictions" : 0
                        },
                        "completion" : {
                            "size_in_bytes" : 0
                        },
                        "segments" : {
                            "count" : 13,
                            "memory_in_bytes" : 81004,
                            "terms_memory_in_bytes" : 50272,
                            "stored_fields_memory_in_bytes" : 6536,
                            "term_vectors_memory_in_bytes" : 0,
                            "norms_memory_in_bytes" : 6336,
                            "points_memory_in_bytes" : 0,
                            "doc_values_memory_in_bytes" : 17860,
                            "index_writer_memory_in_bytes" : 0,
                            "version_map_memory_in_bytes" : 0,
                            "fixed_bit_set_memory_in_bytes" : 48,
                            "max_unsafe_auto_id_timestamp" : -1,
                            "file_sizes" : { }
                        },
                        "translog" : {
                            "operations" : 0,
                            "size_in_bytes" : 440,
                            "uncommitted_operations" : 0,
                            "uncommitted_size_in_bytes" : 440,
                            "earliest_last_modified_age" : 0
                        },
                        "request_cache" : {
                            "memory_size_in_bytes" : 1078,
                            "evictions" : 0,
                            "hit_count" : 0,
                            "miss_count" : 1
                        },
                        "recovery" : {
                            "current_as_source" : 0,
                            "current_as_target" : 0,
                            "throttle_time_in_millis" : 0
                        }
                    },
                    "total" : {
                        "docs" : {
                            "count" : 1029,
                            "deleted" : 10
                        },
                        "store" : {
                            "size_in_bytes" : 538513
                        },
                        "indexing" : {
                            "index_total" : 1040,
                            "index_time_in_millis" : 765,
                            "index_current" : 0,
                            "index_failed" : 18,
                            "delete_total" : 0,
                            "delete_time_in_millis" : 0,
                            "delete_current" : 0,
                            "noop_update_total" : 0,
                            "is_throttled" : 'false',
                            "throttle_time_in_millis" : 0
                        },
                        "get" : {
                            "total" : 56,
                            "time_in_millis" : 312,
                            "exists_total" : 56,
                            "exists_time_in_millis" : 312,
                            "missing_total" : 0,
                            "missing_time_in_millis" : 0,
                            "current" : 0
                        },
                        "search" : {
                            "open_contexts" : 0,
                            "query_total" : 269,
                            "query_time_in_millis" : 220,
                            "query_current" : 0,
                            "fetch_total" : 137,
                            "fetch_time_in_millis" : 203,
                            "fetch_current" : 0,
                            "scroll_total" : 0,
                            "scroll_time_in_millis" : 0,
                            "scroll_current" : 0,
                            "suggest_total" : 0,
                            "suggest_time_in_millis" : 0,
                            "suggest_current" : 0
                        },
                        "merges" : {
                            "current" : 0,
                            "current_docs" : 0,
                            "current_size_in_bytes" : 0,
                            "total" : 3,
                            "total_time_in_millis" : 221,
                            "total_docs" : 30,
                            "total_size_in_bytes" : 161937,
                            "total_stopped_time_in_millis" : 0,
                            "total_throttled_time_in_millis" : 0,
                            "total_auto_throttle_in_bytes" : 251658240
                        },
                        "refresh" : {
                            "total" : 160,
                            "total_time_in_millis" : 1012,
                            "external_total" : 129,
                            "external_total_time_in_millis" : 1026,
                            "listeners" : 0
                        },
                        "flush" : {
                            "total" : 14,
                            "periodic" : 0,
                            "total_time_in_millis" : 110
                        },
                        "warmer" : {
                            "current" : 0,
                            "total" : 63,
                            "total_time_in_millis" : 25
                        },
                        "query_cache" : {
                            "memory_size_in_bytes" : 0,
                            "total_count" : 0,
                            "hit_count" : 0,
                            "miss_count" : 0,
                            "cache_size" : 0,
                            "cache_count" : 0,
                            "evictions" : 0
                        },
                        "fielddata" : {
                            "memory_size_in_bytes" : 0,
                            "evictions" : 0
                        },
                        "completion" : {
                            "size_in_bytes" : 0
                        },
                        "segments" : {
                            "count" : 20,
                            "memory_in_bytes" : 96008,
                            "terms_memory_in_bytes" : 59600,
                            "stored_fields_memory_in_bytes" : 9952,
                            "term_vectors_memory_in_bytes" : 0,
                            "norms_memory_in_bytes" : 7616,
                            "points_memory_in_bytes" : 0,
                            "doc_values_memory_in_bytes" : 18840,
                            "index_writer_memory_in_bytes" : 0,
                            "version_map_memory_in_bytes" : 0,
                            "fixed_bit_set_memory_in_bytes" : 96,
                            "max_unsafe_auto_id_timestamp" : -1,
                            "file_sizes" : { }
                        },
                        "translog" : {
                            "operations" : 0,
                            "size_in_bytes" : 660,
                            "uncommitted_operations" : 0,
                            "uncommitted_size_in_bytes" : 660,
                            "earliest_last_modified_age" : 0
                        },
                        "request_cache" : {
                            "memory_size_in_bytes" : 2156,
                            "evictions" : 0,
                            "hit_count" : 0,
                            "miss_count" : 2
                        },
                        "recovery" : {
                            "current_as_source" : 0,
                            "current_as_target" : 0,
                            "throttle_time_in_millis" : 0
                        }
                    }
                },
                "indices" : {
                    "geonames" : {
                        "uuid" : "2N1IW908QUe1LA9BmKC4lA",
                        "primaries" : {
                            "docs" : {
                                "count" : 1000,
                                "deleted" : 0
                            },
                            "store" : {
                                "size_in_bytes" : 377926
                            },
                            "indexing" : {
                                "index_total" : 1000,
                                "index_time_in_millis" : 560,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 0,
                                "time_in_millis" : 0,
                                "exists_total" : 0,
                                "exists_time_in_millis" : 0,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 0,
                                "query_time_in_millis" : 0,
                                "query_current" : 0,
                                "fetch_total" : 0,
                                "fetch_time_in_millis" : 0,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 0,
                                "total_time_in_millis" : 0,
                                "total_docs" : 0,
                                "total_size_in_bytes" : 0,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 104857600
                            },
                            "refresh" : {
                                "total" : 21,
                                "total_time_in_millis" : 45,
                                "external_total" : 16,
                                "external_total_time_in_millis" : 39,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 5,
                                "periodic" : 0,
                                "total_time_in_millis" : 0
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 11,
                                "total_time_in_millis" : 0
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 8,
                                "memory_in_bytes" : 71464,
                                "terms_memory_in_bytes" : 44672,
                                "stored_fields_memory_in_bytes" : 4096,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 5632,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 17064,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 0,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 275,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 275,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "total" : {
                            "docs" : {
                                "count" : 1000,
                                "deleted" : 0
                            },
                            "store" : {
                                "size_in_bytes" : 377926
                            },
                            "indexing" : {
                                "index_total" : 1000,
                                "index_time_in_millis" : 560,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 0,
                                "time_in_millis" : 0,
                                "exists_total" : 0,
                                "exists_time_in_millis" : 0,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 0,
                                "query_time_in_millis" : 0,
                                "query_current" : 0,
                                "fetch_total" : 0,
                                "fetch_time_in_millis" : 0,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 0,
                                "total_time_in_millis" : 0,
                                "total_docs" : 0,
                                "total_size_in_bytes" : 0,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 104857600
                            },
                            "refresh" : {
                                "total" : 21,
                                "total_time_in_millis" : 45,
                                "external_total" : 16,
                                "external_total_time_in_millis" : 39,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 5,
                                "periodic" : 0,
                                "total_time_in_millis" : 0
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 11,
                                "total_time_in_millis" : 0
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 8,
                                "memory_in_bytes" : 71464,
                                "terms_memory_in_bytes" : 44672,
                                "stored_fields_memory_in_bytes" : 4096,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 5632,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 17064,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 0,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 275,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 275,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        }
                    },
                    ".kibana_-1077723136_mensor_1" : {
                        "uuid" : "llFUZhZZQESrUizqpGjd1Q",
                        "primaries" : {
                            "docs" : {
                                "count" : 1,
                                "deleted" : 0
                            },
                            "store" : {
                                "size_in_bytes" : 3963
                            },
                            "indexing" : {
                                "index_total" : 1,
                                "index_time_in_millis" : 11,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 2,
                                "time_in_millis" : 2,
                                "exists_total" : 2,
                                "exists_time_in_millis" : 2,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 68,
                                "query_time_in_millis" : 68,
                                "query_current" : 0,
                                "fetch_total" : 68,
                                "fetch_time_in_millis" : 93,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 0,
                                "total_time_in_millis" : 0,
                                "total_docs" : 0,
                                "total_size_in_bytes" : 0,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 20971520
                            },
                            "refresh" : {
                                "total" : 9,
                                "total_time_in_millis" : 5,
                                "external_total" : 4,
                                "external_total_time_in_millis" : 6,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 1,
                                "periodic" : 0,
                                "total_time_in_millis" : 11
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 3,
                                "total_time_in_millis" : 5
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 1,
                                "memory_in_bytes" : 1348,
                                "terms_memory_in_bytes" : 784,
                                "stored_fields_memory_in_bytes" : 488,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 0,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 76,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 48,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 55,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 55,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "total" : {
                            "docs" : {
                                "count" : 2,
                                "deleted" : 0
                            },
                            "store" : {
                                "size_in_bytes" : 7926
                            },
                            "indexing" : {
                                "index_total" : 1,
                                "index_time_in_millis" : 11,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 6,
                                "time_in_millis" : 5,
                                "exists_total" : 6,
                                "exists_time_in_millis" : 5,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 135,
                                "query_time_in_millis" : 112,
                                "query_current" : 0,
                                "fetch_total" : 135,
                                "fetch_time_in_millis" : 172,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 0,
                                "total_time_in_millis" : 0,
                                "total_docs" : 0,
                                "total_size_in_bytes" : 0,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 41943040
                            },
                            "refresh" : {
                                "total" : 13,
                                "total_time_in_millis" : 11,
                                "external_total" : 6,
                                "external_total_time_in_millis" : 8,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 3,
                                "periodic" : 0,
                                "total_time_in_millis" : 27
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 4,
                                "total_time_in_millis" : 7
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 2,
                                "memory_in_bytes" : 2696,
                                "terms_memory_in_bytes" : 1568,
                                "stored_fields_memory_in_bytes" : 976,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 0,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 152,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 96,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 110,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 110,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        }
                    },
                    ".kibana_1" : {
                        "uuid" : "9JvZ81ZQSl2IaqxcWaev2g",
                        "primaries" : {
                            "docs" : {
                                "count" : 0,
                                "deleted" : 0
                            },
                            "store" : {
                                "size_in_bytes" : 208
                            },
                            "indexing" : {
                                "index_total" : 0,
                                "index_time_in_millis" : 0,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 0,
                                "time_in_millis" : 0,
                                "exists_total" : 0,
                                "exists_time_in_millis" : 0,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 67,
                                "query_time_in_millis" : 74,
                                "query_current" : 0,
                                "fetch_total" : 1,
                                "fetch_time_in_millis" : 29,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 0,
                                "total_time_in_millis" : 0,
                                "total_docs" : 0,
                                "total_size_in_bytes" : 0,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 20971520
                            },
                            "refresh" : {
                                "total" : 7,
                                "total_time_in_millis" : 0,
                                "external_total" : 3,
                                "external_total_time_in_millis" : 1,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 1,
                                "periodic" : 0,
                                "total_time_in_millis" : 0
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 1
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 0,
                                "memory_in_bytes" : 0,
                                "terms_memory_in_bytes" : 0,
                                "stored_fields_memory_in_bytes" : 0,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 0,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 0,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 0,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 55,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 55,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 1078,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 1
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "total" : {
                            "docs" : {
                                "count" : 0,
                                "deleted" : 0
                            },
                            "store" : {
                                "size_in_bytes" : 416
                            },
                            "indexing" : {
                                "index_total" : 0,
                                "index_time_in_millis" : 0,
                                "index_current" : 0,
                                "index_failed" : 0,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 0,
                                "time_in_millis" : 0,
                                "exists_total" : 0,
                                "exists_time_in_millis" : 0,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 134,
                                "query_time_in_millis" : 108,
                                "query_current" : 0,
                                "fetch_total" : 2,
                                "fetch_time_in_millis" : 31,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 0,
                                "total_time_in_millis" : 0,
                                "total_docs" : 0,
                                "total_size_in_bytes" : 0,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 41943040
                            },
                            "refresh" : {
                                "total" : 10,
                                "total_time_in_millis" : 0,
                                "external_total" : 5,
                                "external_total_time_in_millis" : 2,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 2,
                                "periodic" : 0,
                                "total_time_in_millis" : 0
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 2,
                                "total_time_in_millis" : 2
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 0,
                                "memory_in_bytes" : 0,
                                "terms_memory_in_bytes" : 0,
                                "stored_fields_memory_in_bytes" : 0,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 0,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 0,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 0,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 110,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 110,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 2156,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 2
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        }
                    },
                    ".opendistro_security" : {
                        "uuid" : "wuNhDVnhQi20mLiizs6Y7w",
                        "primaries" : {
                            "docs" : {
                                "count" : 9,
                                "deleted" : 4
                            },
                            "store" : {
                                "size_in_bytes" : 52606
                            },
                            "indexing" : {
                                "index_total" : 13,
                                "index_time_in_millis" : 62,
                                "index_current" : 0,
                                "index_failed" : 18,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 7,
                                "time_in_millis" : 90,
                                "exists_total" : 7,
                                "exists_time_in_millis" : 90,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 0,
                                "query_time_in_millis" : 0,
                                "query_current" : 0,
                                "fetch_total" : 0,
                                "fetch_time_in_millis" : 0,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 1,
                                "total_time_in_millis" : 83,
                                "total_docs" : 10,
                                "total_size_in_bytes" : 53955,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 20971520
                            },
                            "refresh" : {
                                "total" : 43,
                                "total_time_in_millis" : 466,
                                "external_total" : 34,
                                "external_total_time_in_millis" : 454,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 1,
                                "periodic" : 0,
                                "total_time_in_millis" : 0
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 15,
                                "total_time_in_millis" : 3
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 4,
                                "memory_in_bytes" : 8192,
                                "terms_memory_in_bytes" : 4816,
                                "stored_fields_memory_in_bytes" : 1952,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 704,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 720,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 0,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 55,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 55,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        },
                        "total" : {
                            "docs" : {
                                "count" : 27,
                                "deleted" : 10
                            },
                            "store" : {
                                "size_in_bytes" : 152245
                            },
                            "indexing" : {
                                "index_total" : 39,
                                "index_time_in_millis" : 194,
                                "index_current" : 0,
                                "index_failed" : 18,
                                "delete_total" : 0,
                                "delete_time_in_millis" : 0,
                                "delete_current" : 0,
                                "noop_update_total" : 0,
                                "is_throttled" : 'false',
                                "throttle_time_in_millis" : 0
                            },
                            "get" : {
                                "total" : 50,
                                "time_in_millis" : 307,
                                "exists_total" : 50,
                                "exists_time_in_millis" : 307,
                                "missing_total" : 0,
                                "missing_time_in_millis" : 0,
                                "current" : 0
                            },
                            "search" : {
                                "open_contexts" : 0,
                                "query_total" : 0,
                                "query_time_in_millis" : 0,
                                "query_current" : 0,
                                "fetch_total" : 0,
                                "fetch_time_in_millis" : 0,
                                "fetch_current" : 0,
                                "scroll_total" : 0,
                                "scroll_time_in_millis" : 0,
                                "scroll_current" : 0,
                                "suggest_total" : 0,
                                "suggest_time_in_millis" : 0,
                                "suggest_current" : 0
                            },
                            "merges" : {
                                "current" : 0,
                                "current_docs" : 0,
                                "current_size_in_bytes" : 0,
                                "total" : 3,
                                "total_time_in_millis" : 221,
                                "total_docs" : 30,
                                "total_size_in_bytes" : 161937,
                                "total_stopped_time_in_millis" : 0,
                                "total_throttled_time_in_millis" : 0,
                                "total_auto_throttle_in_bytes" : 62914560
                            },
                            "refresh" : {
                                "total" : 116,
                                "total_time_in_millis" : 956,
                                "external_total" : 102,
                                "external_total_time_in_millis" : 977,
                                "listeners" : 0
                            },
                            "flush" : {
                                "total" : 4,
                                "periodic" : 0,
                                "total_time_in_millis" : 83
                            },
                            "warmer" : {
                                "current" : 0,
                                "total" : 46,
                                "total_time_in_millis" : 16
                            },
                            "query_cache" : {
                                "memory_size_in_bytes" : 0,
                                "total_count" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0,
                                "cache_size" : 0,
                                "cache_count" : 0,
                                "evictions" : 0
                            },
                            "fielddata" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0
                            },
                            "completion" : {
                                "size_in_bytes" : 0
                            },
                            "segments" : {
                                "count" : 10,
                                "memory_in_bytes" : 21848,
                                "terms_memory_in_bytes" : 13360,
                                "stored_fields_memory_in_bytes" : 4880,
                                "term_vectors_memory_in_bytes" : 0,
                                "norms_memory_in_bytes" : 1984,
                                "points_memory_in_bytes" : 0,
                                "doc_values_memory_in_bytes" : 1624,
                                "index_writer_memory_in_bytes" : 0,
                                "version_map_memory_in_bytes" : 0,
                                "fixed_bit_set_memory_in_bytes" : 0,
                                "max_unsafe_auto_id_timestamp" : -1,
                                "file_sizes" : { }
                            },
                            "translog" : {
                                "operations" : 0,
                                "size_in_bytes" : 165,
                                "uncommitted_operations" : 0,
                                "uncommitted_size_in_bytes" : 165,
                                "earliest_last_modified_age" : 0
                            },
                            "request_cache" : {
                                "memory_size_in_bytes" : 0,
                                "evictions" : 0,
                                "hit_count" : 0,
                                "miss_count" : 0
                            },
                            "recovery" : {
                                "current_as_source" : 0,
                                "current_as_target" : 0,
                                "throttle_time_in_millis" : 0
                            }
                        }
                    }
                }
            }

        except BaseException:
            self.logger.exception("Could not retrieve index stats.")
            return {}

    def index_times(self, stats, per_shard_stats=True):
        times = []
        self.index_time(times, stats, "merges_total_time", ["merges", "total_time_in_millis"], per_shard_stats)
        self.index_time(times, stats, "merges_total_throttled_time", ["merges", "total_throttled_time_in_millis"], per_shard_stats)
        self.index_time(times, stats, "indexing_total_time", ["indexing", "index_time_in_millis"], per_shard_stats)
        self.index_time(times, stats, "indexing_throttle_time", ["indexing", "throttle_time_in_millis"], per_shard_stats)
        self.index_time(times, stats, "refresh_total_time", ["refresh", "total_time_in_millis"], per_shard_stats)
        self.index_time(times, stats, "flush_total_time", ["flush", "total_time_in_millis"], per_shard_stats)
        return times

    def index_time(self, values, stats, name, path, per_shard_stats):
        primary_total_stats = self.extract_value(stats, ["_all", "primaries"], default_value={})
        value = self.extract_value(primary_total_stats, path)
        if value is not None:
            doc = {
                "name": name,
                "value": value,
                "unit": "ms",
            }
            if per_shard_stats:
                doc["per-shard"] = self.primary_shard_stats(stats, path)
            values.append(doc)

    def index_counts(self, stats):
        counts = []
        self.index_count(counts, stats, "merges_total_count", ["merges", "total"])
        self.index_count(counts, stats, "refresh_total_count", ["refresh", "total"])
        self.index_count(counts, stats, "flush_total_count", ["flush", "total"])
        return counts

    def index_count(self, values, stats, name, path):
        primary_total_stats = self.extract_value(stats, ["_all", "primaries"], default_value={})
        value = self.extract_value(primary_total_stats, path)
        if value is not None:
            doc = {
                "name": name,
                "value": value
            }
            values.append(doc)

    def primary_shard_stats(self, stats, path):
        shard_stats = []
        try:
            for shards in stats["indices"].values():
                for shard in shards["shards"].values():
                    for shard_metrics in shard:
                        if shard_metrics["routing"]["primary"]:
                            shard_stats.append(self.extract_value(shard_metrics, path, default_value=0))
        except KeyError:
            self.logger.warning("Could not determine primary shard stats at path [%s].", ",".join(path))
        return shard_stats

    def add_metrics(self, value, metric_key, unit=None):
        if value is not None:
            if unit:
                self.metrics_store.put_value_cluster_level(metric_key, value, unit)
            else:
                self.metrics_store.put_value_cluster_level(metric_key, value)

    def extract_value(self, primaries, path, default_value=None):
        value = primaries
        try:
            for k in path:
                value = value[k]
            return value
        except KeyError:
            self.logger.warning("Could not determine value at path [%s]. Returning default value [%s]", ",".join(path), str(default_value))
            return default_value


class MlBucketProcessingTime(InternalTelemetryDevice):
    def __init__(self, client, metrics_store):
        super().__init__()
        self.client = client
        self.metrics_store = metrics_store

    def on_benchmark_stop(self):
        # pylint: disable=import-outside-toplevel
        import elasticsearch
        try:
            # results = self.client.search(index=".ml-anomalies-*", body={
            #     "size": 0,
            #     "query": {
            #         "bool": {
            #             "must": [
            #                 {"term": {"result_type": "bucket"}}
            #             ]
            #         }
            #     },
            #     "aggs": {
            #         "jobs": {
            #             "terms": {
            #                 "field": "job_id"
            #             },
            #             "aggs": {
            #                 "min_pt": {
            #                     "min": {"field": "processing_time_ms"}
            #                 },
            #                 "max_pt": {
            #                     "max": {"field": "processing_time_ms"}
            #                 },
            #                 "mean_pt": {
            #                     "avg": {"field": "processing_time_ms"}
            #                 },
            #                 "median_pt": {
            #                     "percentiles": {"field": "processing_time_ms", "percents": [50]}
            #                 }
            #             }
            #         }
            #     }
            # })
           results = ''
        except elasticsearch.TransportError:
            self.logger.exception("Could not retrieve ML bucket processing time.")
            return
        try:
            results = ''
            # for job in results["aggregations"]["jobs"]["buckets"]:
            #     ml_job_stats = collections.OrderedDict()
            #     ml_job_stats["name"] = "ml_processing_time"
            #     ml_job_stats["job"] = job["key"]
            #     ml_job_stats["min"] = job["min_pt"]["value"]
            #     ml_job_stats["mean"] = job["mean_pt"]["value"]
            #     ml_job_stats["median"] = job["median_pt"]["values"]["50.0"]
            #     ml_job_stats["max"] = job["max_pt"]["value"]
            #     ml_job_stats["unit"] = "ms"
            #     self.metrics_store.put_doc(doc=dict(ml_job_stats), level=MetaInfoScope.cluster)
        except KeyError:
            # no ML running
            pass


class IndexSize(InternalTelemetryDevice):
    """
    Measures the final size of the index
    """
    def __init__(self, data_paths):
        super().__init__()
        self.data_paths = data_paths
        self.attached = False
        self.index_size_bytes = None

    def attach_to_node(self, node):
        self.attached = True

    def detach_from_node(self, node, running):
        # we need to gather the file size after the node has terminated so we can be sure that it has written all its buffers.
        if not running and self.attached and self.data_paths:
            self.attached = False
            index_size_bytes = 0
            for data_path in self.data_paths:
                index_size_bytes += io.get_size(data_path)
            self.index_size_bytes = index_size_bytes

    def store_system_metrics(self, node, metrics_store):
        if self.index_size_bytes:
            metrics_store.put_value_node_level(node.node_name, "final_index_size_bytes", self.index_size_bytes, "byte")
