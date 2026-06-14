"""
Drexel CNSG Locomotor Circuit Code for Asymmetry
Python interface for CPGNetworkSimulator
Forked from SimonDanner
- Defines simulator class


Routines
--------

"""

import numpy as np
import numpy.matlib as npml
from scipy.stats import circstd, circmean
from scoop import futures
import time
import functools
import CPGNetworkSimulator as cpgnsim


class simulator:
    stdp_limit = 0.005
    its_limit = 20
    initialized = False

    def __init__(self, **kwargs):
        self.neurons = kwargs.get("neurons", ["RGF_NaP_L_hind", "RGF_NaP_R_hind", "RGF_NaP_L_front", "RGF_NaP_R_front"])
        self.filename = kwargs.get("filename", "./models/MLR_45.txt")
        self.musclenames = kwargs.get("musclenames", ["a", "b"])
        self.dt = kwargs.get("dt", 0.001)
        self.duration = kwargs.get("duration", 10.0)
        self.phase_diffs = kwargs.get("phase_diffs", [(0, 1), (2, 3), (0, 2), (0, 3), (1, 3), (1, 2)])
        self.phase_diff_names = kwargs.get(
            "phase_diff_names", ["L-R hind", "L-R fore", "homolateral", "diagonal", "homolateral", "diagonal"]
        )
        self.alphainit = 0.0
        self.doPreRun = True
        self.variables = {}
        self.total_iterations = 0
        self.steps_per_report = 10
        self.print_progress = kwargs.get("print_progress", True)
        with open(self.filename) as fp:
            for line in fp:
                ln = line.split()
                if len(ln) > 0:
                    if "variable" in ln[0]:
                        if ln[1] not in self.variables.keys():
                            self.variables[ln[1]] = float(ln[2])

    def initialize_simulator(self):
        if not self.initialized:
            self.sim = cpgnsim.CPGNetworkSimulator(self.filename, self.musclenames, (self.neurons,))
            nn = self.sim.getNeuronNames()
            self.neuron_indices = {v: k for k, v in nn.items()}
            self.sim.setAlpha(self.alphainit)
            self.setDuration(self.duration)
            self.initialized = True
            self.eleak0 = np.array(self.sim.getEleak())
            if self.doPreRun:
                for i in range(10):
                    self.run_sim()
                self.IC = self.sim.getState()
            self.v0 = self.sim.setupVariableVector([k for k in self.variables.keys()])

    def reset(self):
        if self.initialized:
            self.sim.updateVariableVector(self.v0)
            self.sim.setState(self.IC)

    def updateVariable(self, name, value):
        if name == "alpha":
            self.sim.setAlpha(value)
        if name == "Eleak":
            # self.sim.setEleak(self.eleak0+value)
            self.sim.setEleak(self.eleak0 * (1.0 - value))
            print("updated Eleak by adding {:2.3f}".format(value))
        else:
            self.sim.updateVariable(name, value)
        # self.iteration_variable = (name, value)

    def getVariableValue(self, name):
        return self.sim.getVariableValue(name)

    def setDuration(self, dur):
        self.duration = dur
        self.time_vec = np.arange(0.0, self.duration, self.dt)

    def run_sim(self):
        out = np.zeros((len(self.time_vec), len(self.neurons)))
        for ind_t, t in enumerate(self.time_vec):
            self.sim.step(self.dt, 1e-6)
            act = self.sim.getAct()
            out[ind_t, :] = act[0]
        return out

    def run_step(self):
        self.sim.step(self.dt)
        act = self.sim.getAct()
        return act[0]

    def run_step_controlled(self):
        self.sim.controlled_step(self.dt)
        act = self.sim.getAct()
        return act[0]

    def run_step_dense(self):
        self.sim.dense_step(self.dt)
        act = self.sim.getAct()
        return act[0]

    @staticmethod
    def calc_on_offsets(time_vec, out, threshold=0.1):
        """Determines onsets and offsets of each neuron
        For each array of neuron output at each timestep, add an indicator where
            - 0, if activity is increasing through threshold (onset)
            - 1, if activity is decreasing through threshold (offset)
        and create a multi-dimension array of on/off transition time steps

        Parameters
        ----------
        time_vec : array-like
            time steps of simulation
        out : array-like
            output activity of each neuron [[neuron_0]...[neuron_n]]
        threshold : float
            threshold for determining bursts

        Returns
        -------
        neuron_on_offsets
            multi-dim array of on and off time steps for each neuron

        """
        # Truth arrays of where neuron output crosses threshold and whether the transition in activity state is
        # increasing at the crossing or decreasing
        is_out_off_to_on = (np.diff((out > threshold).astype(np.int64), axis=0) == 1).T
        is_out_on_to_off = (np.diff((out > threshold).astype(np.int64), axis=0) == -1).T

        # Get the timesteps at the transition points from on to off or vice versa
        on_transition_timesteps = npml.repmat(time_vec[:-1], out.shape[1], 1)[is_out_off_to_on]
        off_transition_timesteps = npml.repmat(time_vec[:-1], out.shape[1], 1)[is_out_on_to_off]

        # Get the corresponding neuron index at each transition point
        idx_on_neurons = (npml.repmat(np.arange(out.shape[1]), len(time_vec) - 1, 1).T)[is_out_off_to_on]
        idx_off_neurons = (npml.repmat(np.arange(out.shape[1]), len(time_vec) - 1, 1).T)[is_out_on_to_off]

        # Creates onsets and offset arrays that stack timesteps, neuron indices, and a generated index
        # Generated index maintains sequence relationship across neurons as output is combined and sorted
        # | on/off t step | neuron num | generated index for row |
        onsets = np.stack((on_transition_timesteps, idx_on_neurons, np.arange(len(idx_on_neurons))), 1)
        onsets = onsets[onsets[:, 0].argsort()]
        offsets = np.stack((off_transition_timesteps, idx_off_neurons, np.arange(len(idx_off_neurons))), 1)
        offsets = offsets[offsets[:, 0].argsort()]

        # build the final matrix that concatenates offsets and onsets along with indicator column where
        # 0 = onset, and 1 = offsets
        # | on/off t step | neuron num | gen. index | on=0,off=1 |
        neurons_on_offsets = np.concatenate((
            np.concatenate((onsets, np.ones((len(onsets), 1)) * 0.0), 1),
            np.concatenate((offsets, np.ones((len(offsets), 1)) * 1.0), 1),
        ))
        neurons_on_offsets = neurons_on_offsets[neurons_on_offsets[:, 0].argsort()]

        return neurons_on_offsets

    @staticmethod
    def calc_phase_mid2(time_vec, out, phase_diffs):
        times = simulator.calc_on_offsets(time_vec, out)
        times_ms_ = []
        for i in range(4):
            times_l = times[times[:, 1] == i]
            indices = np.where(np.diff(times_l[:, 3]) == -1)
            ms = ((times_l[[ind + 1 for ind in indices], 0] + times_l[indices, 0]) / 2).T
            times_ms_.append(
                np.hstack((ms, np.ones_like(ms) * i, npml.reshape(np.arange(len(ms)), ms.shape), np.ones_like(ms) * 2))
            )
        times_ms = np.concatenate(times_ms_)
        times_ms = times_ms[times_ms[:, 0].argsort()]

        ref_onsets = times[np.logical_and(times[:, 1] == 0, times[:, 3] == 0)][:, 0]
        phase_dur = np.append(ref_onsets[1:] - ref_onsets[:-1], np.nan)
        fl_durs = []
        ex_durs = []
        for leg in range(4):
            onset_ = times[np.logical_and(times[:, 1] == leg, times[:, 3] == 0)]
            pd_ = np.append(onset_[1:, 0] - onset_[:-1, 0], np.nan)
            p = times[times[:, 1] == leg]
            indices = np.where(np.diff(p[:, 3]) == 1)
            print(indices, len(p), len(onset_))
            fl_phase_dur = np.zeros((len(onset_)))
            fl_phase_dur[:] = np.nan
            fl_phase_dur[p[indices, 2].astype(int)] = p[[ind + 1 for ind in indices], 0] - p[indices, 0]
            ex_phase_dur = pd_ - fl_phase_dur
            fl_durs.append(fl_phase_dur)
            ex_durs.append(ex_phase_dur)

        M = np.zeros((len(ref_onsets), out.shape[1]))
        M[:] = np.nan
        M[:, 0] = ref_onsets

        for i in range(1, out.shape[1]):
            p = times[np.logical_and((times[:, 1] == 0) | (times[:, 1] == i), times[:, 3] == 0)]
            indices = np.where(np.diff(p[:, 1]) == i)
            M[p[indices, 2].astype(int), i] = p[[ind + 1 for ind in indices], 0]

        phases = np.zeros((len(ref_onsets), len(phase_diffs)))
        for i, (x, y) in enumerate(phase_diffs):
            phases[:, i] = ((M[:, y] - M[:, x]) / phase_dur) % 1.0

        if phases.shape[0] != 0:
            no_nan = ~np.isnan(np.concatenate((np.stack((phase_dur, fl_phase_dur, ex_phase_dur), 1), phases), 1)).any(
                axis=1
            )
            return (phase_dur[no_nan], fl_phase_dur[no_nan], ex_phase_dur[no_nan], phases[no_nan], ref_onsets[no_nan])
        else:
            return (phase_dur, fl_phase_dur, ex_phase_dur, phases, ref_onsets[:-1])

    @staticmethod
    def calc_phase_mid(time_vec, out, phase_diffs):
        times = simulator.calc_on_offsets(time_vec, out)

        times_ms_ = []
        for i in range(4):
            times_l = times[times[:, 1] == i]
            indices = np.where(np.diff(times_l[:, 3]) == -1)
            ms = ((times_l[[ind + 1 for ind in indices], 0] + times_l[indices, 0]) / 2).T
            times_ms_.append(
                np.hstack((ms, np.ones_like(ms) * i, npml.reshape(np.arange(len(ms)), ms.shape), np.ones_like(ms) * 2))
            )
        times = np.concatenate((times, np.concatenate(times_ms_)))
        times = times[times[:, 0].argsort()]

        ref_onsets = times[np.logical_and(times[:, 1] == 0, times[:, 3] == 2)][:, 0]
        phase_dur = np.append(ref_onsets[1:] - ref_onsets[:-1], np.nan)

        p = times[times[:, 1] == 0]
        p = p[p[:, 3] != 2]
        indices = np.where(np.diff(p[:, 3]) == 1)[0]
        indices = indices[: len(ref_onsets)]
        fl_phase_dur = np.zeros((len(ref_onsets)))

        fl_phase_dur[:] = np.nan
        fl_phase_dur[p[indices, 2].astype(int)] = p[[ind + 1 for ind in indices], 0] - p[indices, 0]
        ex_phase_dur = phase_dur - fl_phase_dur

        M = np.zeros((len(ref_onsets), out.shape[1]))
        M[:] = np.nan
        M[:, 0] = ref_onsets

        for i in range(1, out.shape[1]):
            p = times[np.logical_and((times[:, 1] == 0) | (times[:, 1] == i), times[:, 3] == 2)]
            indices = np.where(np.diff(p[:, 1]) == i)
            M[p[indices, 2].astype(int), i] = p[[ind + 1 for ind in indices], 0]

        phases = np.zeros((len(ref_onsets), len(phase_diffs)))
        for i, (x, y) in enumerate(phase_diffs):
            phases[:, i] = ((M[:, y] - M[:, x]) / phase_dur) % 1.0
        if phases.shape[0] != 0:
            no_nan = ~np.isnan(np.concatenate((np.stack((phase_dur, fl_phase_dur, ex_phase_dur), 1), phases), 1)).any(
                axis=1
            )
            return (phase_dur[no_nan], fl_phase_dur[no_nan], ex_phase_dur[no_nan], phases[no_nan], ref_onsets[no_nan])
        else:
            return (phase_dur, fl_phase_dur, ex_phase_dur, phases, ref_onsets)

    @staticmethod
    def calc_phase_measures(time_vec, out, limb_pairings, threshold=0.1, ref_limb=0):
        """Calculates phase durations and differences between limbs

        Parameters
        ==========
        time_vec : array-like
            vector of time steps
        out : array-like
            multi-dimension array of outputs for each neuron
        limb_pairings : array-like
            list of tuples, where each tuple is a pair of limb numbers
        threshold : float
            activity threshold to determine bursts
        ref_limb : int
            index of neuron to use as reference limb for phase differences


        """
        neurons_on_offsets = simulator.calc_on_offsets(time_vec, out, threshold=threshold)
        # | on/offset t step | neuron num | gen. index | onset=0,offset=1) |
        ref_limb_onsets = neurons_on_offsets[
            np.logical_and(neurons_on_offsets[:, 1] == ref_limb, neurons_on_offsets[:, 3] == 0)
        ][:, 0]
        # phase duration is measured from one onset to the next
        ref_limb_period = np.append(np.diff(ref_limb_onsets), np.nan)

        ref_limb_on_offsets = neurons_on_offsets[neurons_on_offsets[:, 1] == ref_limb]
        ref_limb_onset_indices = np.where(np.diff(ref_limb_on_offsets[:, 3]) == 1)
        # flexor phase duration is measured from onset to peak of output
        ref_limb_flexor_duration = np.empty((len(ref_limb_onsets)))
        ref_limb_flexor_duration[:] = np.nan
        ref_limb_flexor_duration[ref_limb_on_offsets[ref_limb_onset_indices, 2].astype(int)] = (
            ref_limb_on_offsets[[idx + 1 for idx in ref_limb_onset_indices], 0]
            - ref_limb_on_offsets[ref_limb_onset_indices, 0]
        )
        # remainder of phase is extensor phase
        ref_limb_extensor_phase_dur = ref_limb_period - ref_limb_flexor_duration

        # M is a matrix of neurons in cols and timestep
        M = np.empty((len(ref_limb_onsets), out.shape[1]))
        M[:] = np.nan
        M[:, 0] = ref_limb_onsets

        # Y = np.empty((len(ref_limb_onsets), out.shape[1]))
        # Y[:] = np.nan

        for neuron_idx in range(1, out.shape[1]):
            neuron_onsets = neurons_on_offsets[
                np.logical_and(
                    (neurons_on_offsets[:, 1] == ref_limb) | (neurons_on_offsets[:, 1] == neuron_idx),
                    neurons_on_offsets[:, 3] == 0,
                )
            ]
            # indices of current limb
            neuron_onset_indices = np.where(np.diff(neuron_onsets[:, 1]) == neuron_idx)
            # populates M with column of onsets for current limb
            M[neuron_onsets[neuron_onset_indices, 2].astype(int), neuron_idx] = neuron_onsets[
                [ind + 1 for ind in neuron_onset_indices], 0
            ]

        # calculates phase diffs for each limb
        phase_diffs = np.zeros((len(ref_limb_onsets), len(limb_pairings)))
        for pair_idx, (limb1, limb2) in enumerate(limb_pairings):
            phase_diffs[:, pair_idx] = ((M[:, limb2] - M[:, limb1]) / ref_limb_period) % 1.0

        if phase_diffs.shape[0] != 0:
            no_nan = ~np.isnan(
                np.concatenate(
                    (
                        np.stack((ref_limb_period, ref_limb_flexor_duration, ref_limb_extensor_phase_dur), 1),
                        phase_diffs,
                    ),
                    1,
                )
            ).any(axis=1)

            # print("-------- DEBUG: in calc_phase ----")
            # IPython.embed()

            return (
                ref_limb_period[no_nan],
                ref_limb_flexor_duration[no_nan],
                ref_limb_extensor_phase_dur[no_nan],
                phase_diffs[no_nan],
                ref_limb_onsets[no_nan],
            )
        else:
            return (
                ref_limb_period,
                ref_limb_flexor_duration,
                ref_limb_extensor_phase_dur,
                phase_diffs,
                ref_limb_onsets[:-1],
            )

    @staticmethod
    def calc_burst_durations(time_vec, neuron_outputs, method=1, threshold=0.1, ref_limb=0):
        """Calculates burst durations for each neuron output
        method = 1 bases calculations on intersection of flexor and extensor activity
        method = 2 uses onset/offset of neuron with activity threshold

        Parameters
        ----------
        time_vec : array-like
            time steps of sim duration
        out : array-like
            output activity of each neuron
        method : int
            determines method to use to calculate burst durations
        threshold : float
            activity threshold for determining burst duration when using method 2
        ref_limb:

        """
        # TODO: Consider refactor to have this function act on a single neuron or neuron pair
        # neuron_outputs structure:
        # # | activity at time step | neuron |
        burst_properties = []
        if method == 1:
            # Determines flexor on/offsets based on comparison with extensor activity
            # ASSUMES: Flexor neurons are the first four neurons passed into the simulation
            # TODO: Fix this fragile architecture to make this no longer rely on neuron order.
            # Onsets are the points when the flexor activity is increasing and becomes greater than extensor activity
            # Offsets are the points when flexor activity is decreasing and becomes less than extensor activity
            for i in range(4):
                flex_ext_diff = np.sign(np.subtract(neuron_outputs[:, i], neuron_outputs[:, i + 4]))
                onsets = np.logical_and(flex_ext_diff[:-1] == -1, flex_ext_diff[1:] == 1)
                offsets = np.logical_and(flex_ext_diff[:-1] == 1, flex_ext_diff[1:] == -1)

                burst_offset_times = time_vec[1:][offsets]
                burst_onset_times = time_vec[:-1][onsets]
                if burst_offset_times.size > 0 and burst_onset_times.size > 0:
                    while burst_offset_times[0] < burst_onset_times[0]:
                        burst_offset_times = burst_offset_times[1:]
                        burst_onset_times = burst_onset_times[:-1]
                    if len(burst_offset_times) < len(burst_onset_times):
                        burst_offset_times = np.pad(
                            burst_offset_times, (0, len(burst_onset_times) - len(burst_offset_times))
                        )
                    if len(burst_onset_times) < len(burst_offset_times):
                        burst_onset_times = np.pad(
                            burst_onset_times, (0, len(burst_offset_times) - len(burst_onset_times))
                        )
                    burst_dur = burst_offset_times - burst_onset_times
                burst_properties.append(np.stack((burst_dur, burst_onset_times, burst_offset_times), 1))
            # Determine extensor on/off sets using flexor activity and calculates burst durations
            # ASSUMES: The extensor neurons are the next four neurons
            for i in range(4, 8):
                burst_offset_times = burst_properties[i - 4][:, 1]
                burst_onset_times = burst_properties[i - 4][:, 2]
                while burst_offset_times[0] < burst_onset_times[0]:
                    burst_offset_times = burst_offset_times[1:]
                    burst_onset_times = burst_onset_times[:-1]
                burst_dur = burst_offset_times - burst_onset_times
                burst_properties.append(np.stack((burst_dur, burst_onset_times, burst_offset_times), 1))

            # Determines CINs and LPNs on/offsets and calculate their burst durations
            # This essentially uses method 2 because the CINs and LPNs do not have a paired activity to compare with
            on_off_times = simulator.calc_on_offsets(time_vec, neuron_outputs, threshold=threshold)
            # on_off_times structure:
            # # | on/off t step | neuron num | gen. index | onset=0,offset=1) |
            for i in range(8, neuron_outputs.shape[1]):
                on_off_times_neuron = on_off_times[on_off_times[:, 1] == i]
                is_burst_onset = np.logical_and(on_off_times_neuron[:-1, 3] == 0, on_off_times_neuron[1:, 3] == 1)
                burst_dur = (
                    on_off_times_neuron[1:, :][is_burst_onset, 0] - on_off_times_neuron[:-1, :][is_burst_onset, 0]
                )
                burst_onset_times = on_off_times_neuron[:-1, :][is_burst_onset, 0]
                burst_offset_times = on_off_times_neuron[1:, :][is_burst_onset, 0]
                burst_properties.append(np.stack((burst_dur, burst_onset_times, burst_offset_times), 1))

        if method == 2:
            on_off_times = simulator.calc_on_offsets(time_vec, neuron_outputs, threshold=threshold)
            # on_off_times structure:
            # # | on/off t step | neuron num | gen. index | onset=0,offset=1) |
            for i in range(neuron_outputs.shape[1]):
                on_off_times_neuron = on_off_times[on_off_times[:, 1] == i]
                # print(len(on_off_times_neuron))
                # compare on/off values of array trimed at end vs array trimmed at start
                # this shift allows detection of burst onset by pair of 0 and 1 between the arrays
                is_burst_onset = np.logical_and(on_off_times_neuron[:-1, 3] == 0, on_off_times_neuron[1:, 3] == 1)
                # is_burst_offset = np.logical_and(on_off_times_neuron[:-1,3]==1,on_off_times_neuron[1:,3]==0)
                # similar shifting, but this time to detect changing between on/off
                burst_dur = (
                    on_off_times_neuron[1:, :][is_burst_onset, 0] - on_off_times_neuron[:-1, :][is_burst_onset, 0]
                )
                burst_onset_times = on_off_times_neuron[:-1, :][is_burst_onset, 0]
                burst_offset_times = on_off_times_neuron[1:, :][is_burst_onset, 0]
                burst_properties.append(np.stack((burst_dur, burst_onset_times, burst_offset_times), 1))

        # print("-------- DEBUG: in calc_burst_durations ----")
        # IPython.embed()

        return burst_properties

    @staticmethod
    def classify_gait_simple(duty_factor, phases):
        lr_hind = phases[:, 0]
        homolateral = phases[:, 2]
        diagonal = phases[:, 3]

        lr_alt = np.logical_and(lr_hind >= 0.25, lr_hind <= 0.75)
        hl_alt = np.logical_and(homolateral >= 0.25, homolateral <= 0.75)
        di_alt = np.logical_and(diagonal >= 0.25, diagonal <= 0.75)
        diag_quarter = np.logical_or(
            np.logical_and(diagonal > 0.1, diagonal <= 0.4), np.logical_and(diagonal >= 0.6, diagonal < 0.9)
        )
        walk = np.logical_and(np.logical_and(duty_factor > 0.5, lr_alt), diag_quarter)
        trot = np.logical_and(lr_alt, np.logical_and(np.logical_or(diagonal <= 0.1, diagonal >= 0.9), hl_alt))
        gallop = np.logical_and(
            hl_alt,
            np.logical_or(
                np.logical_and(lr_hind > 0.025, lr_hind <= 0.25), np.logical_and(lr_hind >= 0.75, lr_hind < 0.975)
            ),
        )
        bound = np.logical_and(hl_alt, np.logical_or(lr_hind <= 0.025, lr_hind >= 0.975))
        if len(duty_factor) > 0:
            gaits = np.zeros((len(duty_factor),), dtype=int)
            gaits[walk] = 1
            gaits[trot] = 2
            gaits[gallop] = 3
            gaits[bound] = 4
        else:
            gaits = [np.nan]

        return gaits

    def do_iteration(self):
        mfq = np.nan
        mphases_ = []
        max_std_phases = 1.0
        its = 0
        while max_std_phases > self.stdp_limit and its < self.its_limit:
            out = self.run_sim()
            # phase_dur,fl_phase_dur,ex_phase_dur, phases, _ = self.calc_phase_mid(self.time_vec,out,self.phase_diffs)
            phase_dur, fl_phase_dur, ex_phase_dur, phases, _ = self.calc_phase_measures(
                self.time_vec, out, self.phase_diffs
            )
            if len(phase_dur) > 10:
                mphases_ = circmean(phases[-5:, :], 1.0, 0.0, 0)
                max_std_phases = np.max(circstd(phases[-5:, :], 1.0, 0.0, 0))
                mfq = 1.0 / np.nanmean(phase_dur[-5:])
            its += 1
            self.total_iterations += 1
        if its >= self.its_limit:
            mphases_ = [np.nan for m in mphases_]
            if self.print_progress:
                ostr = "max iterations reached"
                if hasattr(self, "iteration_variable"):
                    ostr += " - variable: {:5} {:4.3f}".format(self.iteration_variable[0], self.iteration_variable[1])
                ostr += " - max std phases: {:2.3f} > {:2.3f}".format(max_std_phases, self.stdp_limit)
                print(ostr)

        gaits = self.classify_gait_simple((ex_phase_dur / phase_dur)[-5:], phases[-5:, :])
        return (mfq, mphases_, np.nanmean(fl_phase_dur[-5:]), np.nanmean(ex_phase_dur[-5:]), gaits[-1])

    def do_1d_bifurcation(self, variable_name, range_, steps, updown=True):
        self.initialize_simulator()
        v = range_[0] + (np.arange(0, steps, 1)) / (steps - 1.0) * (range_[1] - range_[0])

        frequency = np.zeros((steps, 2)) * np.nan
        fl_dur = np.zeros((steps, 2)) * np.nan
        ex_dur = np.zeros((steps, 2)) * np.nan
        gait = np.zeros((steps, 2), dtype=int) * np.nan
        phases = np.zeros((steps, len(self.phase_diffs), 2)) * np.nan
        IChist = list()
        j_start_back = 0
        go_up_on_nan = True
        self.total_iterations = 0
        # self.sim.setState(IC)

        self.updateVariable(variable_name, v[0])
        tic = time.perf_counter()
        for j in range(0, steps):
            IChist.append(self.sim.getState())
            self.updateVariable(variable_name, v[j])

            fq, phases_, fl_dur_, ex_dur_, gait_ = self.do_iteration()
            if not np.isnan(fq):
                frequency[j, 0] = fq
                fl_dur[j, 0] = fl_dur_
                ex_dur[j, 0] = ex_dur_
                if isinstance(gait_, float):
                    gait[j, 0] = gait_
                phases[j, :, 0] = phases_
            j_start_back = j - 1
            # if np.isnan(fq) and not go_up_on_nan:
            #    j_start_back=j-2
            #    break
            # if not np.isnan(fq):
            #    go_up_on_nan=False
            if (j + 1) % self.steps_per_report == 0:
                toc = time.perf_counter()
                if self.print_progress:
                    print(f"Iter {j + 1} of {steps}*2; It/sec {self.steps_per_report / (toc - tic):.3f}")
                tic = time.perf_counter()

        if updown:
            self.sim.setState(IChist[j_start_back])
            for j in np.arange(j_start_back, -1, -1):
                self.updateVariable(variable_name, v[j])
                fq, phases_, fl_dur_, ex_dur_, gait_ = self.do_iteration()
                if not np.isnan(fq):
                    frequency[j, 1] = fq
                    fl_dur[j, 1] = fl_dur_
                    ex_dur[j, 1] = ex_dur_
                    if isinstance(gait_, float):
                        gait[j, 1] = gait_
                    phases[j, :, 1] = phases_
                n_steps_completed = j_start_back + j_start_back - j + 1
                if (n_steps_completed + 1) % self.steps_per_report == 0:
                    toc = time.perf_counter()
                    print(
                        f"Iter {n_steps_completed + 1} of {steps}*2; It/sec {self.steps_per_report / (toc - tic):.3f}"
                    )
                    tic = time.perf_counter()
        print("total sim time", self.total_iterations * self.duration)
        return (v, frequency, phases, fl_dur, ex_dur, gait)

    def do_1d_bifurcation_helper(self, at_value, bi_variable_name, bi_range, bi_steps, at_variable_name, updown=True):
        self.initialize_simulator()
        self.sim.setState(self.IC)
        self.its_limit = 5
        self.updateVariable(at_variable_name, at_value)
        return self.do_1d_bifurcation(bi_variable_name, bi_range, bi_steps, updown)

    def do_2d_bifurcation(self, variable_names, ranges, steps, updown=False):
        v0 = ranges[0][0] + (np.arange(0, steps[0], 1)) / (steps[0] - 1.0) * (ranges[0][1] - ranges[0][0])
        frequency = np.zeros(steps + (2,)) * np.nan
        gaits = np.zeros(steps + (2,)) * np.nan
        phases = np.zeros(steps + (len(self.phase_diffs), 2)) * np.nan
        paras = futures.map(
            functools.partial(
                self.do_1d_bifurcation_helper,
                bi_variable_name=variable_names[1],
                bi_range=ranges[1],
                bi_steps=steps[1],
                at_variable_name=variable_names[0],
                updown=updown,
            ),
            v0,
        )
        v1 = []
        for i, (v_, fq, ph, g) in enumerate(paras):
            frequency[i, :, :] = fq
            phases[i, :, :, :] = ph
            gaits[i, :, :] = g
            v1 = v_
        return (v0, v1, frequency, phases, gaits)

    def do_noise_iteration(self, value, variable_name, sigma, set_variables=list()):
        self.initialize_simulator()
        self.sim.updateParameter("sigmaNoise", sigma)
        self.updateVariable(variable_name, value)
        for name, value in set_variables:
            self.updateVariable(name, value)

        out = self.run_sim()
        phase_dur, fl_phase_dur, ex_phase_dur, phases, _ = self.calc_phase_measures(
            self.time_vec, out, self.phase_diffs
        )
        gaits = self.classify_gait_simple((ex_phase_dur / phase_dur), phases)
        return (1 / phase_dur, fl_phase_dur, ex_phase_dur, phases, gaits)

    def do_noise_simulation(self, variable_name, range_, steps, sigma, set_variables=list()):
        v = range_[0] + (np.arange(0, steps, 1)) / (steps - 1.0) * (range_[1] - range_[0])
        self.duration = 100.0
        self.doPreRun = False
        fqs = list()
        fl_phase_durs = list()
        ex_phase_durs = list()
        phases = list()
        gaits = list()
        paras = futures.map(
            functools.partial(
                self.do_noise_iteration, variable_name=variable_name, sigma=sigma, set_variables=set_variables
            ),
            v,
        )
        for i, (fq, fl, ex, ph, g) in enumerate(paras):
            fqs.append(fq)
            fl_phase_durs.append(fl)
            ex_phase_durs.append(ex)
            phases.append(ph)
            gaits.append(g)

        return (v, fqs, fl_phase_durs, ex_phase_durs, phases, gaits)
