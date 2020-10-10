from numpy import ndarray


class NetworkSimulationStrategy:
    def simulate(self) -> ndarray:
        raise NotImplementedError
