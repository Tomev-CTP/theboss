import unittest

from numpy import asarray

from src.BosonSamplingSimulator import BosonSamplingSimulator
from src.LossyBosonSamplingExactDistributionCalculators import (
    BosonSamplingExperimentConfiguration)
from src.network_simulation_strategy.LossyNetworkSimulationStrategy import LossyNetworkSimulationStrategy
from src.Quantum_Computations_Utilities import generate_haar_random_unitary_matrix
from src.SimulationStrategyFactory import SimulationStrategyFactory


class TestBosonSamplingClassicalSimulationStrategies(unittest.TestCase):

    def setUp(self) -> None:
        self._probability_of_uniform_loss = 0.8
        self._initial_state = [1, 1, 1, 1, 0, 0, 0, 0]
        self._number_of_samples_for_experiments = 1000
        self._lossy_interferometer_matrix = self._probability_of_uniform_loss * generate_haar_random_unitary_matrix(
            len(self._initial_state))
        self._experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self._lossy_interferometer_matrix,
            initial_state=asarray(self._initial_state),
            initial_number_of_particles=sum(self._initial_state),
            number_of_modes=len(self._initial_state),
            number_of_particles_lost=0,  # Losses should only come from network.
            number_of_particles_left=sum(self._initial_state),
            probability_of_uniform_loss=self._probability_of_uniform_loss,
            network_simulation_strategy=LossyNetworkSimulationStrategy(self._lossy_interferometer_matrix)
        )
        self._strategy_factory = SimulationStrategyFactory(self._experiment_configuration)

    def test_lossy_network_simulation_number_of_particles(self) -> None:
        """
        This test checks if number of particles for lossy network simulator is lower than number of
        particles without the losses. Note that I can be pretty sure it will be lower in large losses
        regime, but in case of lower losses this test may not hold.
        """
        strategy = self._strategy_factory.generate_strategy()
        simulator = BosonSamplingSimulator(strategy)
        lossy_average_number_of_particles = 0

        for _ in range(self._number_of_samples_for_experiments):
            lossy_average_number_of_particles += sum(simulator.get_classical_simulation_results(self._initial_state))

        lossy_average_number_of_particles /= self._number_of_samples_for_experiments
        lossless_number_of_particles = sum(self._initial_state)
        self.assertLess(lossy_average_number_of_particles, lossless_number_of_particles)
