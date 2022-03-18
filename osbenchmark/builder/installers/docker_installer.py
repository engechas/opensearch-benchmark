import logging
import os
import uuid

import jinja2
from jinja2 import select_autoescape

from osbenchmark.builder.installers.installer import Installer
from osbenchmark.exceptions import InvalidSyntax, SystemSetupError


class DockerInstaller(Installer):
    def __init__(self, pci, node_name, ip, http_port, node_root_dir, distribution_version, benchmark_root):
        self.logger = logging.getLogger(__name__)
        self.pci = pci

        self.node_name = node_name
        self.node_ip = ip
        self.http_port = http_port
        self.node_root_dir = node_root_dir
        self.node_log_dir = os.path.join(node_root_dir, "logs", "server")
        self.heap_dump_dir = os.path.join(node_root_dir, "heapdump")
        self.distribution_version = distribution_version
        self.benchmark_root = benchmark_root
        self.binary_path = os.path.join(node_root_dir, "install")
        # use a random subdirectory to isolate multiple runs because an external (non-root) user cannot clean it up.
        self.data_paths = [os.path.join(node_root_dir, "data", str(uuid.uuid4()))]


        provisioner_defaults = {
            "cluster_name": "benchmark-provisioned-cluster",
            "node_name": self.node_name,
            # we bind-mount the directories below on the host to these ones.
            "install_root_path": "/usr/share/opensearch",
            "data_paths": ["/usr/share/opensearch/data"],
            "log_path": "/var/log/opensearch",
            "heap_dump_path": "/usr/share/opensearch/heapdump",
            # Docker container needs to expose service on external interfaces
            "network_host": "0.0.0.0",
            "discovery_type": "single-node",
            "http_port": str(self.http_port),
            "transport_port": str(self.http_port + 100),
            "cluster_settings": {}
        }

        self.config_vars = {}
        self.config_vars.update(self.provision_config_instance.variables)
        self.config_vars.update(provisioner_defaults)

    def prepare(self, binary):
        # we need to allow other users to write to these directories due to Docker.
        #
        # Although os.mkdir passes 0o777 by default, mkdir(2) uses `mode & ~umask & 0777` to determine the final flags and
        # hence we need to modify the process' umask here. For details see https://linux.die.net/man/2/mkdir.
        previous_umask = os.umask(0)
        try:
            io.ensure_dir(self.binary_path)
            io.ensure_dir(self.node_log_dir)
            io.ensure_dir(self.heap_dump_dir)
            io.ensure_dir(self.data_paths[0])
        finally:
            os.umask(previous_umask)

        mounts = {}

        for provision_config_instance_config_path in self.provision_config_instance.config_paths:
            for root, _, files in os.walk(provision_config_instance_config_path):
                env = jinja2.Environment(loader=jinja2.FileSystemLoader(root), autoescape=select_autoescape(['html', 'xml']))

                relative_root = root[len(provision_config_instance_config_path) + 1:]
                absolute_target_root = os.path.join(self.binary_path, relative_root)
                io.ensure_dir(absolute_target_root)

                for name in files:
                    source_file = os.path.join(root, name)
                    target_file = os.path.join(absolute_target_root, name)
                    mounts[target_file] = os.path.join("/usr/share/opensearch", relative_root, name)
                    if plain_text(source_file):
                        self.logger.info("Reading config template file [%s] and writing to [%s].", source_file, target_file)
                        with open(target_file, mode="a", encoding="utf-8") as f:
                            f.write(_render_template(env, self.config_vars, source_file))
                    else:
                        self.logger.info("Treating [%s] as binary and copying as is to [%s].", source_file, target_file)
                        shutil.copy(source_file, target_file)

        docker_cfg = self._render_template_from_file(self.docker_vars(mounts))
        self.logger.info("Starting Docker container with configuration:\n%s", docker_cfg)

        with open(os.path.join(self.binary_path, "docker-compose.yml"), mode="wt", encoding="utf-8") as f:
            f.write(docker_cfg)

        return NodeConfiguration("docker", self.provision_config_instance.mandatory_var("runtime.jdk"),
                                 convert.to_bool(self.provision_config_instance.mandatory_var("runtime.jdk.bundled")), self.node_ip,
                                 self.node_name, self.node_root_dir, self.binary_path, self.data_paths)

    def docker_vars(self, mounts):
        v = {
            "os_version": self.distribution_version,
            "docker_image": self.provision_config_instance.mandatory_var("docker_image"),
            "http_port": self.http_port,
            "os_data_dir": self.data_paths[0],
            "os_log_dir": self.node_log_dir,
            "os_heap_dump_dir": self.heap_dump_dir,
            "mounts": mounts
        }
        self._add_if_defined_for_provision_config_instance(v, "docker_mem_limit")
        self._add_if_defined_for_provision_config_instance(v, "docker_cpu_count")
        return v

    def _add_if_defined_for_provision_config_instance(self, variables, key):
        if key in self.pci.variables:
            variables[key] = self.pci.variables[key]

    def _render_template(self, loader, template_name, variables):
        try:
            env = jinja2.Environment(loader=loader, autoescape=select_autoescape(['html', 'xml']))
            for k, v in variables.items():
                env.globals[k] = v
            template = env.get_template(template_name)

            return template.render()
        except jinja2.exceptions.TemplateSyntaxError as e:
            raise InvalidSyntax("%s in %s" % (str(e), template_name))
        except BaseException as e:
            raise SystemSetupError("%s in %s" % (str(e), template_name))

    def _render_template_from_file(self, variables):
        compose_file = os.path.join(self.benchmark_root, "resources", "docker-compose.yml.j2")
        return self._render_template(loader=jinja2.FileSystemLoader(io.dirname(compose_file)),
                                     template_name=io.basename(compose_file),
                                     variables=variables)

    def install(self, host, binaries):
        pass

    def cleanup(self, host, node_configurations):
        pass