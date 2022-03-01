"""
Launchers are used to start and stop OpenSearch on the nodes in a self-managed cluster.
"""
class Launcher:
    def __init__(self, executor):
        self.executor = executor

    def start(self, host, node_configurations):
        """
        Starts the OpenSearch nodes on a given host

        ;param host: A Host object defining the host on which to start the nodes
        ;param node_configurations: A list of NodeConfiguration objects detailing the installation data of the nodes on the host
        ;return nodes: A list of Node objects defining the nodes running on a host
        """
        raise NotImplementedError

    def stop(self, host, nodes):
        """
        Stops the OpenSearch nodes on a given host

        ;param host: A Host object defining the host on which to stop the nodes
        ;param nodes: A list of Node objects defining the nodes running on a host
        ;return nodes: A list of Node objects representing OpenSearch nodes that were successfully stopped on the host
        """
        raise NotImplementedError
