import logging
import os
import subprocess

from osbenchmark import time, telemetry
from osbenchmark.builder import java_resolver, cluster
from osbenchmark.builder.launchers.launcher import Launcher
from osbenchmark.exceptions import LaunchError, ExecutorError
from osbenchmark.utils import opts, io


class ProcessLauncher(Launcher):
    PROCESS_WAIT_TIMEOUT_SECONDS = 90.0

    def __init__(self, pci, executor, clock=time.Clock):
        super().__init__(executor)
        self.logger = logging.getLogger(__name__)
        self.pci = pci
        self._clock = clock
        self.pass_env_vars = opts.csv_to_list(self.pci.variables["system"]["env"]["passenv"])

    def start(self, host, node_configurations):
        node_count_on_host = len(node_configurations)
        return [self._start_node(host, node_configuration, node_count_on_host) for node_configuration in node_configurations]

    def _start_node(self, host, node_configuration, node_count_on_host):
        host_name = node_configuration.ip
        node_name = node_configuration.node_name
        binary_path = node_configuration.binary_path
        data_paths = node_configuration.data_paths
        node_telemetry_dir = os.path.join(node_configuration.node_root_path, "telemetry")

        java_major_version, java_home = java_resolver.java_home(node_configuration.provision_config_instance_runtime_jdks,
                                                                self.pci.variables["system"]["runtime"]["jdk"],
                                                                node_configuration.provision_config_instance_provides_bundled_jdk)
        self.logger.info("Java major version: %s", java_major_version)
        self.logger.info("Java home: %s", java_home)

        self.logger.info("Starting node [%s].", node_name)

        enabled_devices = self.pci.variables["telemetry"]["devices"]
        telemetry_params = self.pci.variables["telemetry"]["params"]
        # TODO non-stats telemetry devices? Easy solution is don't support on external hosts. Otherwise need a shell_executor
        # TODO doesn't seem MVP to support these telem devices on external hosts
        # TODO could just branch on self.pci.provider == Provider.LOCAL
        
        node_telemetry = [
            telemetry.FlightRecorder(telemetry_params, node_telemetry_dir, java_major_version),
            telemetry.JitCompiler(node_telemetry_dir),
            telemetry.Gc(telemetry_params, node_telemetry_dir, java_major_version),
            telemetry.Heapdump(node_telemetry_dir),
            telemetry.DiskIo(node_count_on_host),
            telemetry.IndexSize(data_paths),
            telemetry.StartupTime(),
        ]

        t = telemetry.Telemetry(enabled_devices, devices=node_telemetry)
        env = self._prepare_env(host, node_name, java_home, t)
        t.on_pre_node_start(node_name)
        node_pid = self._start_process(host, binary_path, env)
        self.logger.info("Successfully started node [%s] with PID [%s].", node_name, node_pid)
        node = cluster.Node(node_pid, binary_path, host_name, node_name, t)

        self.logger.info("Attaching telemetry devices to node [%s].", node_name)
        t.attach_to_node(node)

        return node

    def _prepare_env(self, host, node_name, java_home, t):
        env = self._get_env(host)
        if java_home:
            self._set_env(env, "PATH", os.path.join(java_home, "bin"), separator=":", prepend=True)
            # This property is the higher priority starting in ES 7.12.0, and is the only supported java home in >=8.0
            env["OPENSEARCH_JAVA_HOME"] = java_home
            # TODO remove this when ES <8.0 becomes unsupported by Benchmark
            env["JAVA_HOME"] = java_home
            self.logger.info("JAVA HOME: %s", env["JAVA_HOME"])
        if not env.get("OPENSEARCH_JAVA_OPTS"):
            env["OPENSEARCH_JAVA_OPTS"] = "-XX:+ExitOnOutOfMemoryError"

        # we just blindly trust telemetry here...
        for v in t.instrument_candidate_java_opts():
            self._set_env(env, "OPENSEARCH_JAVA_OPTS", v)

        self.logger.debug("env for [%s]: %s", node_name, str(env))
        return env

    def _get_env(self, host):
        env_vars = self.shell_executor.execute(host, "printenv", output=True)
        env_vars_as_dict = dict(env_var.split("=") for env_var in env_vars)

        pass_env_keys = set(self.pass_env_vars).intersection(set(env_vars_as_dict.keys()))
        return {key: env_vars_as_dict[key] for key in pass_env_keys}

    def _set_env(self, env, k, v, separator=' ', prepend=False):
        if v is not None:
            if k not in env:
                env[k] = v
            elif prepend:
                env[k] = v + separator + env[k]
            else:
                env[k] = env[k] + separator + v

    def _start_process(self, host, binary_path, env):
        if int(self.shell_executor.execute(host, "echo $EUID")[0]) == 0:
            raise LaunchError("Cannot launch OpenSearch as root. Please run Benchmark as a non-root user.")

        cmd = [io.escape_path(os.path.join(binary_path, "bin", "opensearch"))]
        cmd.extend(["-d", "-p", "pid"])
        try:
            self.shell_executor.execute(host, " ".join(cmd), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, detach=True)
        except ExecutorError as e:
            raise LaunchError("Exception starting OpenSearch", e)

        return self._wait_for_pidfile(host, io.escape_path(os.path.join(binary_path, "pid")))

    def _wait_for_pidfile(self, host, pidfilename):
        stop_watch = self._clock.stop_watch()
        stop_watch.start()
        while stop_watch.split_time() < self.PROCESS_WAIT_TIMEOUT_SECONDS:
            try:
                return int(self.shell_executor.execute(host, "cat " + pidfilename, output=True)[0])
            except (ExecutorError, IndexError):
                time.sleep(0.5)

        msg = "pid file not available after {} seconds!".format(self.PROCESS_WAIT_TIMEOUT_SECONDS)
        self.logger.error(msg)
        raise LaunchError(msg)

    def stop(self, host, nodes, metrics_store):
        self.logger.info("Shutting down [%d] nodes on this host.", len(nodes))
        stopped_nodes = []
        for node in nodes:
            node_name = node.node_name
            if metrics_store:
                telemetry.add_metadata_for_node(metrics_store, node_name, node.host_name)

            if self._is_process_running(host, node.pid):
                node.telemetry.detach_from_node(node, running=True)

                stop_watch = self._clock.stop_watch()
                stop_watch.start()

                # SIGTERM to kill gracefully
                if self._terminate_process(host, node.pid):
                    time.sleep(10)

                    if self._is_process_running(host, node.pid):
                        # SIGKILL to force kill
                        if self._kill_process(host, node.pid):
                            stopped_nodes.append(node)
                    else:
                        stopped_nodes.append(node)

                self.logger.info("Done shutting down node [%s] in [%.1f] s.", node_name, stop_watch.split_time())

                node.telemetry.detach_from_node(node, running=False)

            # store system metrics in any case (telemetry devices may derive system metrics while the node is running)
            if metrics_store:
                node.telemetry.store_system_metrics(node, metrics_store)
        return stopped_nodes

    def _is_process_running(self, host, pid):
        try:
            self.shell_executor.execute(host, "ps aux | awk '$2 == " + str(pid) + "'", output=True, shell=True)[0]
            return True
        except IndexError:
            self.logger.warning("No process found with PID [%s] on host [%s].", pid, host)
            return False

    def _terminate_process(self, host, pid):
        try:
            self.shell_executor.execute(host, "kill " + str(pid))
            return True
        except ExecutorError:
            self.logger.warning("No process found with PID [%s] on host [%s].", pid, host)
            return False

    def _kill_process(self, host, pid):
        try:
            self.shell_executor.execute(host, "kill -9 " + str(pid))
            return True
        except ExecutorError:
            self.logger.warning("No process found with PID [%s] on host [%s].", pid, host)
            return False
