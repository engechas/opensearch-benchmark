from osbenchmark.builder.installers.installer import Installer
from osbenchmark.exceptions import InstallError


class ExceptionHandlingInstaller(Installer):
    def __init__(self, installer, executor=None):
        super().__init__(executor)
        self.installer = installer

    def install(self, host, binaries, all_node_ips):
        try:
            return self.installer.install(host, binaries, all_node_ips)
        except Exception as e:
            raise InstallError("Installing node on host \"{}\" failed".format(host), e)

    def cleanup(self, host):
        try:
            return self.installer.cleanup(host)
        except Exception as e:
            raise InstallError("Cleaning up install data on host \"{}\" failed".format(host), e)
