import subprocess

from osbenchmark.builder.executors.shell_executor import ShellExecutor
from osbenchmark.exceptions import ExecutorError
from osbenchmark.utils import process


class LocalShellExecutor(ShellExecutor):
    # pylint: disable=arguments-differ
    def execute(self, host, command, output=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=None, detach=False, shell=False):
        if output:
            return process.run_subprocess_with_output(command, shell=shell)
        else:
            if process.run_subprocess_with_logging(command, stdout=stdout, stderr=stderr, env=env, detach=detach, shell=shell):
                raise ExecutorError("Command: \"{}\" returned a non-zero exit code".format(command))
