# pylint: disable=protected-access

from unittest import TestCase, mock
from unittest.mock import Mock

from osbenchmark import telemetry
from osbenchmark.builder import cluster
from osbenchmark.builder.launchers.process_launcher import ProcessLauncher
from osbenchmark.builder.provision_config import ProvisionConfigInstance
from osbenchmark.builder.provisioner import NodeConfiguration
from osbenchmark.exceptions import ExecutorError
from tests.builder.launchers.launcher_test_helpers import TestClock, IterationBasedStopWatch


class ProcessLauncherTests(TestCase):
    def setUp(self):
        self.shell_executor = Mock()
        self.metrics_store = Mock()

        self.variables = {
            "system": {
                "runtime": {
                    "jdk": None
                },
                "env": {
                    "passenv": "PATH"
                }
            },
            "telemetry": {
                "devices": [],
                "params": None
            }
        }
        self.pci = ProvisionConfigInstance("fake_pci", "/path/to/root", ["/path/to/config"], variables=self.variables)

        self.proc_launcher = ProcessLauncher(self.pci, self.shell_executor, self.metrics_store,
                                             clock=TestClock(IterationBasedStopWatch(1)))
        self.host = None
        self.path = "fake"

    @mock.patch('osbenchmark.builder.java_resolver.java_home', return_value=(12, "/java_home/"))
    @mock.patch('osbenchmark.utils.jvm.supports_option', return_value=True)
    @mock.patch('osbenchmark.telemetry.Telemetry')
    def test_daemon_start_stop(self, telemetry, supports, java_home):
        telemetry.attach_to_node.return_value = None

        node_configs = []
        for node in range(2):
            node_configs.append(NodeConfiguration(build_type="tar",
                                                  provision_config_instance_runtime_jdks="12,11",
                                                  provision_config_instance_provides_bundled_jdk=True,
                                                  ip="127.0.0.1",
                                                  node_name="testnode-{}".format(node),
                                                  node_root_path="/tmp",
                                                  binary_path="/tmp",
                                                  data_paths="/tmp"))

        # (printenv, echo $EUID, start opensearch process, check pidfile, get pid) x 2
        self.proc_launcher.shell_executor.execute.side_effect = [["PATH=" + self.path], ["12"], None, ["5555555"], ["5555555"],
                                                                 ["PATH=" + self.path], ["12"], None, ["6666666"], ["6666666"]]
        nodes = self.proc_launcher.start(self.host, node_configs)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].pid, 5555555)

        # (check if process running, terminate process, check if process running) x 2
        self.proc_launcher.shell_executor.execute.side_effect = [["fake process exists"], None, [],
                                                                 ["second fake process exists"], None, []]
        stopped_nodes = self.proc_launcher.stop(self.host, nodes)
        # all nodes should be stopped
        self.assertEqual(nodes, stopped_nodes)

    def test_daemon_stop_with_already_terminated_process(self):
        nodes = [
            cluster.Node(pid=-1,
                         binary_path="/bin",
                         host_name="localhost",
                         node_name="benchmark-0",
                         telemetry=telemetry.Telemetry())
        ]

        self.proc_launcher.shell_executor.execute.return_value = []

        stopped_nodes = self.proc_launcher.stop(self.host, nodes)
        # no nodes should have been stopped (they were already stopped)
        self.assertEqual([], stopped_nodes)

    def test_env_options_order(self):
        node_telemetry = [
            telemetry.FlightRecorder(telemetry_params={}, log_root="/tmp/telemetry", java_major_version=8)
        ]
        t = telemetry.Telemetry(["jfr"], devices=node_telemetry)

        self.proc_launcher.shell_executor.execute.return_value = ["PATH=" + self.path]
        env = self.proc_launcher._prepare_env(self.host, node_name="node0", java_home="/java_home", t=t)

        self.assertEqual("/java_home/bin:" + self.path, env["PATH"])
        self.assertEqual("-XX:+ExitOnOutOfMemoryError -XX:+UnlockDiagnosticVMOptions -XX:+DebugNonSafepoints "
                         "-XX:+UnlockCommercialFeatures -XX:+FlightRecorder "
                         "-XX:FlightRecorderOptions=disk=true,maxage=0s,maxsize=0,dumponexit=true,dumponexitpath=/tmp/telemetry/profile.jfr " # pylint: disable=line-too-long
                         "-XX:StartFlightRecording=defaultrecording=true", env["OPENSEARCH_JAVA_OPTS"])

    def test_bundled_jdk_not_in_path(self):
        self.proc_launcher.shell_executor.execute.return_value = ["PATH=" + self.path, "JAVA_HOME=/path/to/java"]

        t = telemetry.Telemetry()
        # no JAVA_HOME -> use the bundled JDK
        env = self.proc_launcher._prepare_env(self.host, node_name="node0", java_home=None, t=t)

        # unmodified
        self.assertEqual(self.path, env["PATH"])
        self.assertIsNone(env.get("JAVA_HOME"))

    def test_pass_env_vars(self):
        java_home = "/path/to/java"
        foo1 = "BAR1"

        self.proc_launcher.pci.variables["system"]["env"]["passenv"] = "JAVA_HOME,FOO1"
        self.proc_launcher.shell_executor.execute.return_value = ["PATH=" + self.path, "JAVA_HOME=" + java_home, "FOO1=" + foo1]

        t = telemetry.Telemetry()
        # no JAVA_HOME -> use the bundled JDK
        env = self.proc_launcher._prepare_env(self.host, node_name="node0", java_home=None, t=t)

        # unmodified
        self.assertEqual(java_home, env["JAVA_HOME"])
        self.assertEqual(foo1, env["FOO1"])
        self.assertEqual(env["OPENSEARCH_JAVA_OPTS"], "-XX:+ExitOnOutOfMemoryError")

    def test_pass_java_opts(self):
        self.proc_launcher.pci.variables["system"]["env"]["passenv"] = "OPENSEARCH_JAVA_OPTS"

        opensearch_java_opts = "-XX:-someJunk"
        self.proc_launcher.shell_executor.execute.return_value = ["PATH=" + self.path, "OPENSEARCH_JAVA_OPTS=" + opensearch_java_opts]

        t = telemetry.Telemetry()
        # no JAVA_HOME -> use the bundled JDK
        env = self.proc_launcher._prepare_env(self.host, node_name="node0", java_home=None, t=t)

        # unmodified
        self.assertEqual(opensearch_java_opts, env["OPENSEARCH_JAVA_OPTS"])

    def test_pid_file_not_found(self):
        self.proc_launcher.shell_executor.execute.side_effect = ExecutorError("fake")

        is_file_ready = self.proc_launcher._is_pid_file_available(self.host, "fake")
        self.assertEqual(is_file_ready, False)

    def test_pid_file_empty(self):
        self.proc_launcher.shell_executor.execute.return_value = []

        is_file_ready = self.proc_launcher._is_pid_file_available(self.host, "fake")
        self.assertEqual(is_file_ready, False)

    def test_pid_file_ready(self):
        self.proc_launcher.shell_executor.execute.return_value = ["1234"]

        is_file_ready = self.proc_launcher._is_pid_file_available(self.host, "testpidfile")
        self.assertEqual(is_file_ready, True)
