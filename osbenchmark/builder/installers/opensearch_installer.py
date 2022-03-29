import logging
import os
import uuid

from osbenchmark.builder.installers.installer import Installer
from osbenchmark.builder.models.node import Node
from osbenchmark.builder.provision_config import BootstrapHookHandler


class OpenSearchInstaller(Installer):
    OPENSEARCH_BINARY_KEY = "opensearch"

    def __init__(self, provision_config_instance, executor, hook_handler_class=BootstrapHookHandler):
        self.logger = logging.getLogger(__name__)
        super().__init__(executor, self.logger)
        self.provision_config_instance = provision_config_instance
        self.hook_handler = hook_handler_class(self.provision_config_instance)
        if self.hook_handler.can_load():
            self.hook_handler.load()

    def install(self, host, binaries, all_node_ips):
        node = self._create_node()
        self._prepare_node(host, node, binaries[OpenSearchInstaller.OPENSEARCH_BINARY_KEY], all_node_ips)

        return node

    def _create_node(self):
        node_name = str(uuid.uuid4())
        node_port = int(self.provision_config_instance.variables["node"]["port"])
        node_root_dir = os.path.join(self.provision_config_instance.variables["test_execution_root"], node_name)
        node_binary_path = os.path.join(node_root_dir, "install")

        return Node(name=node_name,
                    port=node_port,
                    pid=None,
                    root_dir=node_root_dir,
                    binary_path=node_binary_path,
                    data_paths=None,
                    telemetry=None)

    def _prepare_node(self, host, node, binary, all_node_ips):
        node_log_dir = os.path.join(node.root_dir, "logs", "server")
        node_heap_dump_dir = os.path.join(node.root_dir, "heapdump")

        self._prepare_directories(host, node, node_log_dir, node_heap_dump_dir)
        self._extract_opensearch(host, node, binary)
        self._update_node_binary_path(node)
        self._set_node_data_paths(node)
        # we need to immediately delete the prebundled config files as plugins may copy their configuration during installation.
        self._delete_prebundled_config_files(host, node)
        self._prepare_config_files(host, node, node_log_dir, node_heap_dump_dir, all_node_ips)

    def _prepare_directories(self, host, node, node_log_dir, node_heap_dump_dir):
        directories_to_create = [node.binary_path, node_log_dir, node_heap_dump_dir]
        for directory_to_create in directories_to_create:
            self._create_directory(host, directory_to_create)

    def _extract_opensearch(self, host, node, binary):
        self.logger.info("Unzipping %s to %s", binary, node.binary_path)
        self.executor.execute(host, "tar -xzvf {} --directory {}".format(binary, node.binary_path))

    def _update_node_binary_path(self, node):
        node.binary_path = os.path.join(node.binary_path, "opensearch*")

    def _set_node_data_paths(self, node):
        node.data_paths = [os.path.join(node.binary_path, "data")]

    def _delete_prebundled_config_files(self, host, node):
        config_path = os.path.join(node.binary_path, "config")
        self.logger.info("Deleting pre-bundled OpenSearch configuration at [%s]", config_path)
        self._delete_path(host, config_path)

    def _prepare_config_files(self, host, node, log_dir, heap_dump_dir, all_node_ips):
        config_vars = self._get_config_vars(host, node, log_dir, heap_dump_dir, all_node_ips)
        self._apply_configs(host, node, self.provision_config_instance.config_paths, config_vars)

    def _get_config_vars(self, host, node, log_dir, heap_dump_dir, all_node_ips):
        provisioner_defaults = {
            "cluster_name": self.provision_config_instance.variables["cluster_name"],
            "node_name": node.name,
            "data_paths": node.data_paths[0],
            "log_path": log_dir,
            "heap_dump_path": heap_dump_dir,
            # this is the node's IP address as specified by the user when invoking Benchmark
            "node_ip": host.address,
            # this is the IP address that the node will be bound to. Benchmark will bind to the node's IP address (but not to 0.0.0.0). The
            "network_host": host.address,
            "http_port": str(node.port),
            "transport_port": str(node.port + 100),
            "all_node_ips": "[\"%s\"]" % "\",\"".join(all_node_ips),
            # at the moment we are strict and enforce that all nodes are master eligible nodes
            "minimum_master_nodes": len(all_node_ips),
            "install_root_path": node.binary_path
        }
        config_vars = {}
        config_vars.update(self.provision_config_instance.variables)
        config_vars.update(provisioner_defaults)
        return config_vars

    def invoke_install_hook(self, phase, variables, env):
        self.hook_handler.invoke(phase.name, variables=variables, env=env)

    def cleanup(self, host):
        self._cleanup(host, self.provision_config_instance.variables["preserve_install"])