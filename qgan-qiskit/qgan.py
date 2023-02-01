#from qiskit import *
#from qiskit.visualization import *
import scipy
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.optimize import minimize

from qiskit import BasicAer, QuantumCircuit, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA,SLSQP,SPSA

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

simulator = BasicAer.get_backend("statevector_simulator")

class Discriminator:

    def __init__(self, feature_vector, d_params, num_ansatz_reps):

        self.feature_vector = feature_vector
        self.d_params = d_params
        self.num_ansatz_reps = num_ansatz_reps


    def d_distribution(self):

        num_qubits = len(self.feature_vector)
        qc = QuantumCircuit(num_qubits)

        #---------- QUANTUM FEATURE MAP ------------ #
        feature_vector = self.feature_vector
        for qubit, variable in enumerate(feature_vector):
            qc.ry(variable, qubit)

        # -------------- ANSATZ --------------- #
        qc.barrier()
        ansatz = RealAmplitudes(num_qubits=num_qubits,
                                reps=self.num_ansatz_reps,
                                ).assign_parameters(self.d_params)

        # --------------- MEASUREMENT ------------------ #
        qc.append(ansatz, list(range(num_qubits)))
        counts = execute(qc, backend=simulator).result().get_counts(qc)
        return counts

    def parity_function(self):

        P_d = self.d_distribution()

        strings=['0000','0001','0010','0011','0100','0101','0110','0111',
                '1000','1001','1010','1011','1100','1101','1110','1111']

        for string in strings:
            if string not in P_d:
                P_d[string] = 0

        # ----------- PARITY FUNCTION <ZIII...III> ----------- #
        exp_value = 0
        for key, value in P_d.items():
            if key[0]==0:
                exp_value += value
            else:
                exp_value -= value

        norm_exp_value = (exp_value + 1)/2
        return norm_exp_value


class Generator:

    def __init__(self, latent_vector, g_params, num_ansatz_reps):

        self.latent_vector = latent_vector
        self.g_params = g_params
        self.num_ansatz_reps = num_ansatz_reps

    def g_distribution(self):

        num_qubits = len(self.latent_vector)
        qc = QuantumCircuit(num_qubits)

        # ----------- QUANTUM FEATURE MAP ------------- #
        latent_vector = self.latent_vector
        for qubit, variable in enumerate(latent_vector):
            qc.ry(variable, qubit)

        # ---------------------- ANSATZ ----------------------- #
        qc.barrier()
        ansatz = RealAmplitudes(num_qubits=num_qubits,
                                reps=self.num_ansatz_reps,
                                ).assign_parameters(self.g_params)

        # ----------------- GENERATOR DISTRIBUTION ------------------- #
        qc.append(ansatz, list(range(num_qubits)))
        counts = execute(qc, backend=simulator).result().get_counts(qc)

        strings=['00','01','10','11']

        for string in strings:
            if string not in counts:
                counts[string]=0

        sorted_counts = {}
        for string in strings:
            sorted_counts[string] = counts[string]

        probs = []
        for key, value in sorted_counts.items():
            probs.append(value)

        return probs


def D_crossentropy(disc_params, data, num_samples, g_num_qubits, g_initial_guess, g_num_reps, d_num_reps):
    # --------------- PREPARE FAKE MINIBATCH  -----------------#
    fake_minibatch = []

    for m in range(num_samples):
        # -------------------- GENERATE LATENT VECTOR ---------------------- #
        latent_vec = np.random.uniform(0, 2*np.pi, g_num_qubits)

        # ---------------- PASS LATENT VECTOR ON GENERATOR ----------------- #
        generator = Generator(latent_vector = latent_vec,
                                g_params = g_initial_guess,
                                num_ansatz_reps = g_num_reps)

        # -------------------- GENERATE FAKE DISTRIBUTION ------------------ #
        fake_distr = generator.g_distribution()
        fake_minibatch.append(fake_distr)


    # ---------------- PREPARE REAL MINIBATCH  --------------- #
    real_minibatch = []

    for m in range(num_samples):
        # --------------------- SAMPLE FROM REAL DATA ---------------------- #
        real_vector = data.iloc[m, :-1]
        real_minibatch.append(real_vector)


    # -------------------- DISCRIMINATOR CROSS-ENTROPY --------------------- #
    d_loss = 0

    for fake_data, real_data in zip(fake_minibatch, real_minibatch):

        # ---------------- PASS FAKE DATA ON DISCRIMINATOR ----------------- #
        fake_y_estimator = Discriminator(feature_vector = fake_data,
                                        d_params = disc_params,
                                        num_ansatz_reps = d_num_reps
                                        ).parity_function()

        # ------------------ DISCRIMINATOR LOSS FOR FAKE ------------------- #
        d_loss_fake = -(np.log(1 - fake_y_estimator))

        # ---------------- PASS REAL DATA ON DISCRIMINATOR ----------------- #
        real_y_estimator = Discriminator(feature_vector = real_data,
                                         d_params = disc_params,
                                         num_ansatz_reps = d_num_reps
                                        ).parity_function()

        # ------------------ DISCRIMINATOR LOSS FOR REAL ------------------- #
        d_loss_real = -(np.log(real_y_estimator))

        # -----------------DISCRIMINATOR CROSS ENTROPY --------------------- #
        d_loss += d_loss_fake + d_loss_real


    print(" ----- Cross-entropy from discriminator :  ", d_loss/num_samples, " ---------")
    return d_loss/num_samples


def G_crossentropy(g_params, optimal_params_discriminator, num_samples, g_num_qubits, g_num_reps, d_num_reps):

    fake_minibatch = []

    for m in range(num_samples):
        # -------------------- GENERATE LATENT VECTOR ---------------------- #
        latent_vec = np.random.uniform(0, 2*np.pi, g_num_qubits)

        # ---------------- PASS LATENT VECTOR ON GENERATOR ----------------- #
        generator = Generator(latent_vector = latent_vec,
                                g_params = g_params,
                                num_ansatz_reps = g_num_reps)

        # -------------------- GENERATE FAKE DISTRIBUTION ------------------ #
        fake_distr = generator.g_distribution()
        fake_minibatch.append(fake_distr)


    # -------------------- GENERATOR CROSS-ENTROPY --------------------- #
    g_loss = 0

    for fake_data in fake_minibatch:

        # ---------------- PASS FAKE DATA ON DISCRIMINATOR ----------------- #
        fake_y_estimator = Discriminator(feature_vector = fake_data,
                                    d_params = optimal_params_discriminator,
                                    num_ansatz_reps = d_num_reps
                                    ).parity_function()

        # ------------------------ GENERATOR LOSS -------------------------- #
        g_loss -= (np.log2(fake_y_estimator))



    print(" ------- Cross-entropy from Generator :  ", g_loss, " -----------")
    return g_loss


def main():
    # --- Preparing Normalized Real Data ---
    print("Retrieving Data...")
    print("-----------------")
    path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'data'))
    filename = "irisvirginica.data"
    data = pd.read_csv(os.path.join(path,filename))
    label = data.iloc[:, -1]
    features = data.iloc[:, :-1]

    x = features.values
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 2*np.pi))
    x_scaled = min_max_scaler.fit_transform(x)
    features = pd.DataFrame(x_scaled)
    data = features.assign(labels = label)

    print(data.head())

    # --- Discriminator Cross-Entropy ---
    EPOCHS = 200; NUM_SAMPLES = data.shape[0]
    g_num_qubits = 2; d_num_qubits = 4; g_num_reps = 4; d_num_reps = 4
    g_initial_guess = np.random.uniform(0, np.pi, g_num_qubits + g_num_qubits*g_num_reps)
    d_initial_guess = np.random.uniform(0, np.pi, d_num_qubits + d_num_qubits*d_num_reps)

    # --- Training Discriminator ---
    print("Training Discriminator")
    print("-----------------")
    d_args = data, NUM_SAMPLES, g_num_qubits, g_initial_guess, g_num_reps, d_num_reps
    result_d = scipy.optimize.minimize(fun = D_crossentropy, x0 = d_initial_guess, args=d_args)
    optimal_params_discriminator = result_d[0]

    # --- Training Generator ---
    print("Training Generator")
    print("-----------------")
    g_args = optimal_params_discriminator, NUM_SAMPLES, g_num_qubits, g_num_reps, d_num_reps
    result_g = optimizer.optimize(num_vars=len(g_initial_guess),
                                objective_function=G_crossentropy,
                                initial_point=g_initial_guess)

    optimal_params_generator = result_g[0]

    # --- Fake Data Quantum Generator ---
    NUM_NEW_SAMPLES = 10

    new_samples = []

    for sample in range(NUM_NEW_SAMPLES):
        # ----------------------- NEW LATENT VECTOR ------------------------ #
        latent_vec = np.random.uniform(0, 2*np.pi, g_num_qubits)

        # ---------------- PASS LATENT VECTOR ON GENERATOR ----------------- #
        generator = Generator(latent_vector = latent_vec,
                                g_params = optimal_params_generator,
                                num_ansatz_reps = g_num_reps)

        # -------------------- GENERATE FAKE DISTRIBUTION ------------------ #
        fake_distr = generator.g_distribution()
        new_samples.append(fake_distr)

    print("NEW SAMPLES DERIVED FROM QUANTUM GENERATOR: \n ", new_samples)

if __name__ == "__main__":
    main()