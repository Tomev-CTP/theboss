__author__ = "Tomasz Rybotycki"

from random import random
from numpy import ndarray
from src.simulation_strategies.GeneralizedCliffordsSimulationStrategy import GeneralizedCliffordsSimulationStrategy


class GeneralizedCliffordsUniformLossesSimulationStrategy(GeneralizedCliffordsSimulationStrategy):
    def __init__(self, interferometer_matrix: ndarray, uniform_losses: float = 0):
        self._uniform_losses = uniform_losses
        super().__init__(interferometer_matrix=interferometer_matrix)

    def _fill_r_sample(self) -> None:
        self.r_sample = [0 for _ in self.interferometer_matrix]
        self.current_key = tuple(self.r_sample)
        self.current_sample_probability = 1

        for i in range(self.number_of_input_photons):
            if random() <= self._uniform_losses:
                continue
            if self.current_key not in self.pmfs:
                self._calculate_new_layer_of_pmfs()
            self._sample_from_latest_pmf()
