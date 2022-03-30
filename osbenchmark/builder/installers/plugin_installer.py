import logging
import os

from osbenchmark.builder.installers.installer import Installer
from osbenchmark.builder.provision_config import BootstrapHookHandler
from osbenchmark.builder.utils.config_applier import ConfigApplier
from osbenchmark.builder.utils.path_manager import PathManager
from osbenchmark.builder.utils.template_renderer import TemplateRenderer


class PluginInstaller(Installer):
    def __init__(self, plugin, executor, hook_handler_class=BootstrapHookHandler):
        super().__init__(executor)
        self.logger = logging.getLogger(__name__)
        self.plugin = plugin
        self.hook_handler = hook_handler_class(self.plugin)
        if self.hook_handler.can_load():
            self.hook_handler.load()
        self.template_renderer = TemplateRenderer()
        self.path_manager = PathManager(executor)
        self.config_applier = ConfigApplier(executor, self.template_renderer, self.path_manager)

    def install(self, host, binaries, all_node_ips):
        install_cmd = self._get_install_command(host, binaries)
        self.executor.execute(host, install_cmd)

    def _get_install_command(self, host, binaries):
        installer_binary_path = os.path.join(host.node.binary_path, "bin", "opensearch-plugin")
        plugin_binary_path = binaries.get(self.plugin.name)

        if plugin_binary_path:
            self.logger.info("Installing [%s] into [%s] from [%s]", self.plugin.name, host.node.binary_path, plugin_binary_path)
            return '%s install --batch "%s"' % (installer_binary_path, plugin_binary_path)
        else:
            self.logger.info("Installing [%s] into [%s]", self.plugin.name, host.node.binary_path)
            return '%s install --batch "%s"' % (installer_binary_path, self.plugin.name)

    def get_config_vars(self):

    def invoke_install_hook(self, phase, variables, env):
        self.hook_handler.invoke(phase.name, variables=variables, env=env)

    def cleanup(self, host):
        pass
