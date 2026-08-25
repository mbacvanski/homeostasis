"""Mechanics tests for the homeostatic reservoir core.

Every dynamical rule from Falandays et al. (2024) eqs. 1-5 is exercised on
tiny hand-constructed networks with hand-computed expected values, so any
deviation from the paper's update rules fails loudly.
"""

import numpy as np
import pytest

from homeostasis import HomeostaticReservoir, ReservoirConfig


def make_manual_net(
    input_adj,
    adj,
    weights,
    out_adj,
    input_weight=1.0,
    leak=0.25,
    target_lr=0.01,
    clamp_negative_activations=False,
):
    """Build a reservoir and overwrite its random structure with given matrices."""
    input_adj = np.array(input_adj, dtype=bool)
    adj = np.array(adj, dtype=bool)
    out_adj = np.array(out_adj, dtype=bool)
    n_inputs, n_nodes = input_adj.shape
    cfg = ReservoirConfig(
        n_nodes=n_nodes,
        n_inputs=n_inputs,
        n_outputs=out_adj.shape[1],
        input_weight=input_weight,
        leak=leak,
        target_lr=target_lr,
        clamp_negative_activations=clamp_negative_activations,
        # allow manual matrices to include self-links if a test wants them
        allow_self_connections=True,
    )
    net = HomeostaticReservoir(cfg, seed=0)
    net.input_adjacency = input_adj
    net.input_weights = input_adj * input_weight
    net.adjacency = adj
    net.weights = np.where(adj, np.array(weights, dtype=float), 0.0)
    net.output_adjacency = out_adj
    net._rebuild_structure_caches()
    return net


def single_node_net(input_weight=1.0, leak=0.25):
    """One reservoir node, one input wired to it, one output wired to it."""
    return make_manual_net(
        input_adj=[[True]],
        adj=[[False]],
        weights=[[0.0]],
        out_adj=[[True]],
        input_weight=input_weight,
        leak=leak,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_shapes_and_seed_determinism(self):
        a = HomeostaticReservoir(ReservoirConfig(), seed=42)
        b = HomeostaticReservoir(ReservoirConfig(), seed=42)
        assert a.input_adjacency.shape == (62, 200)
        assert a.adjacency.shape == (200, 200)
        assert a.output_adjacency.shape == (200, 2)
        assert np.array_equal(a.adjacency, b.adjacency)
        assert np.array_equal(a.weights, b.weights)
        assert np.array_equal(a.input_adjacency, b.input_adjacency)
        assert np.array_equal(a.output_adjacency, b.output_adjacency)

    def test_different_seeds_differ(self):
        a = HomeostaticReservoir(ReservoirConfig(), seed=1)
        b = HomeostaticReservoir(ReservoirConfig(), seed=2)
        assert not np.array_equal(a.weights, b.weights)

    def test_no_self_connections_by_default(self):
        net = HomeostaticReservoir(ReservoirConfig(), seed=3)
        assert not np.any(np.diag(net.adjacency))

    def test_weights_zero_off_adjacency(self):
        net = HomeostaticReservoir(ReservoirConfig(), seed=4)
        assert np.all(net.weights[~net.adjacency] == 0.0)

    def test_input_weights_uniform(self):
        net = HomeostaticReservoir(ReservoirConfig(), seed=5)
        assert np.all(net.input_weights[net.input_adjacency] == 0.75)
        assert np.all(net.input_weights[~net.input_adjacency] == 0.0)

    def test_recurrent_weight_init_distribution(self):
        # Released code: rand(Normal(input_amp, .1)) on every link, i.e.
        # Normal(0.75, 0.1) — all-positive in practice (not the paper's N(0,1)).
        net = HomeostaticReservoir(ReservoirConfig(), seed=5)
        w = net.weights[net.adjacency]
        assert w.mean() == pytest.approx(0.75, abs=0.02)
        assert w.std() == pytest.approx(0.1, abs=0.02)
        assert np.all(w > 0)

    def test_initial_state(self):
        net = HomeostaticReservoir(ReservoirConfig(), seed=6)
        assert np.all(net.x == 0.0)
        assert np.all(net.targets == 1.0)
        assert np.all(net.thresholds == 2.0)
        assert not np.any(net.spiked)

    def test_link_density_near_p(self):
        net = HomeostaticReservoir(ReservoirConfig(), seed=7)
        assert net.adjacency.mean() == pytest.approx(0.1, abs=0.02)
        assert net.input_adjacency.mean() == pytest.approx(0.1, abs=0.02)


# ---------------------------------------------------------------------------
# Eq. 1: integration (leak, input, recurrent delivery)
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_leak_and_input(self):
        # x_t = x_{t-1} * (1 - l) + i @ W_in, with no spikes anywhere.
        net = single_node_net(input_weight=0.75)
        s1 = net.step(np.array([1.0]))
        assert s1.x[0] == pytest.approx(0.75)  # 0 * 0.75 + 1 * 0.75
        assert not s1.spiked[0]
        s2 = net.step(np.array([1.0]))
        assert s2.x[0] == pytest.approx(0.75 * 0.75 + 0.75)  # 1.3125

    def test_zero_input_decays(self):
        net = single_node_net()
        net.x = np.array([1.0])
        s = net.step(np.array([0.0]))
        assert s.x[0] == pytest.approx(0.75)

    def test_input_shape_validated(self):
        net = single_node_net()
        with pytest.raises(ValueError):
            net.step(np.zeros(3))

    def test_negative_activation_allowed_by_default(self):
        # Inhibitory delivery can push activation below zero; the released
        # runs disable the clamp (acts_neg switch).
        net = make_manual_net(
            input_adj=[[True, False]],
            adj=[[False, True], [False, False]],
            weights=[[0.0, -1.5], [0.0, 0.0]],
            out_adj=[[True], [False]],
        )
        net.step(np.array([3.0]))  # node 0 spikes
        s2 = net.step(np.array([0.0]))
        assert s2.x[1] == pytest.approx(-1.5)

    def test_negative_activation_clamped_when_configured(self):
        net = make_manual_net(
            input_adj=[[True, False]],
            adj=[[False, True], [False, False]],
            weights=[[0.0, -1.5], [0.0, 0.0]],
            out_adj=[[True], [False]],
            clamp_negative_activations=True,
        )
        net.step(np.array([3.0]))
        s2 = net.step(np.array([0.0]))
        assert s2.x[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Eq. 2: spiking
# ---------------------------------------------------------------------------


class TestSpiking:
    def test_paper_worked_example(self):
        # "if a node n has a current threshold T' = 2 and current activation
        # x = 2.5, it will spike and drop to an activation x' = 0.5"
        net = single_node_net()
        s = net.step(np.array([2.5]))
        assert s.spiked[0]
        assert s.x[0] == pytest.approx(0.5)

    def test_spike_at_exact_threshold(self):
        # The released code spikes at >= threshold (`acts .>= thresholds`).
        net = single_node_net()
        s = net.step(np.array([2.0]))  # exactly at threshold
        assert s.spiked[0]
        assert s.x[0] == pytest.approx(0.0)

    def test_single_spike_per_step_no_refractory(self):
        # x = 10 with T' = 2: one spike only (x' = 8), but the node may spike
        # again on the following step.
        net = single_node_net()
        s1 = net.step(np.array([10.0]))
        assert s1.spiked[0]
        assert s1.x[0] == pytest.approx(8.0)
        s2 = net.step(np.array([0.0]))  # x = 8 * 0.75 = 6 > threshold
        assert s2.spiked[0]

    def test_threshold_tracks_target(self):
        net = single_node_net()
        net.targets = np.array([1.5])
        assert net.thresholds[0] == pytest.approx(3.0)
        s = net.step(np.array([2.9]))  # below 3.0
        assert not s.spiked[0]


# ---------------------------------------------------------------------------
# Eqs. 3-4: error and target update
# ---------------------------------------------------------------------------


class TestTargetUpdate:
    def test_error_uses_post_spike_activation(self):
        net = single_node_net()
        s = net.step(np.array([2.5]))  # spikes, x' = 0.5
        assert s.error[0] == pytest.approx(0.5 - 1.0)

    def test_target_increases_when_above(self):
        net = single_node_net()
        s = net.step(np.array([1.8]))  # no spike, E = 0.8
        assert s.error[0] == pytest.approx(0.8)
        assert net.targets[0] == pytest.approx(1.0 + 0.01 * 0.8)

    def test_target_floor(self):
        net = single_node_net()
        s = net.step(np.array([0.0]))  # E = -1, update would go below floor
        assert s.error[0] == pytest.approx(-1.0)
        assert net.targets[0] == pytest.approx(1.0)

    def test_target_decreases_when_above_floor(self):
        net = single_node_net()
        net.targets = np.array([1.5])
        net.step(np.array([0.0]))  # x = 0, E = -1.5
        assert net.targets[0] == pytest.approx(1.5 - 0.015)

    def test_target_decrease_clipped_at_floor(self):
        net = single_node_net()
        net.targets = np.array([1.001])
        net.step(np.array([0.0]))  # E ~ -1.001 => raw update below 1
        assert net.targets[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Eq. 5: weight update
# ---------------------------------------------------------------------------


class TestWeightUpdate:
    def two_source_net(self):
        # Nodes 0 and 1 each get their own input; both project to node 2.
        return make_manual_net(
            input_adj=[[True, False, False], [False, True, False]],
            adj=[
                [False, False, True],
                [False, False, True],
                [False, False, False],
            ],
            weights=[
                [0.0, 0.0, 0.5],
                [0.0, 0.0, -0.3],
                [0.0, 0.0, 0.0],
            ],
            out_adj=[[False], [False], [True]],
        )

    def test_full_error_split_equally_opposite_sign(self):
        net = self.two_source_net()
        # Step 1: both sources spike (x = 3 > 2). Node 2 receives nothing yet.
        net.step(np.array([3.0, 3.0]))
        assert list(net.spiked) == [True, True, False]
        w_before = net.weights.copy()

        # Step 2: node 2 integrates 0.5 - 0.3 = 0.2, no spike, E2 = 0.2 - 1 = -0.8.
        # Its two incoming weights (both sources spiked at step 1) each move by
        # -E2 / 2 = +0.4. Source nodes have no in-links, so nothing else moves.
        s2 = net.step(np.array([0.0, 0.0]))
        assert s2.error[2] == pytest.approx(-0.8)
        assert net.weights[0, 2] == pytest.approx(0.5 + 0.4)
        assert net.weights[1, 2] == pytest.approx(-0.3 + 0.4)
        untouched = np.ones_like(net.weights, dtype=bool)
        untouched[0, 2] = untouched[1, 2] = False
        assert np.array_equal(net.weights[untouched], w_before[untouched])

    def test_positive_error_decreases_weights(self):
        net = self.two_source_net()
        net.weights[0, 2] = 2.0
        net.weights[1, 2] = 2.0
        net.step(np.array([3.0, 3.0]))  # both sources spike
        # Node 2 integrates 4.0, spikes (threshold 2), x' = 2, E2 = 1.0;
        # each incoming weight moves by -E2 / 2 = -0.5.
        s2 = net.step(np.array([0.0, 0.0]))
        assert s2.spiked[2]
        assert s2.error[2] == pytest.approx(1.0)
        assert net.weights[0, 2] == pytest.approx(1.5)
        assert net.weights[1, 2] == pytest.approx(1.5)

    def test_only_links_from_previous_spikers_update(self):
        net = self.two_source_net()
        net.step(np.array([3.0, 0.0]))  # only node 0 spikes
        assert list(net.spiked) == [True, False, False]
        # Node 2 integrates 0.5, E2 = -0.5; only the link from node 0 moves,
        # by -E2 / 1 = +0.5 (count excludes node 1, which did not spike).
        net.step(np.array([0.0, 0.0]))
        assert net.weights[0, 2] == pytest.approx(0.5 + 0.5)
        assert net.weights[1, 2] == pytest.approx(-0.3)

    def test_no_previous_spikers_no_update(self):
        net = self.two_source_net()
        w_before = net.weights.copy()
        net.step(np.array([0.5, 0.5]))  # nobody spikes
        net.step(np.array([0.0, 0.0]))
        assert np.array_equal(net.weights, w_before)

    def test_count_only_connected_spikers(self):
        # Node 1 spikes but has no link to node 2: the error must be divided
        # by 1 (only node 0 is a spiking in-neighbor), not 2.
        net = make_manual_net(
            input_adj=[[True, False, False], [False, True, False]],
            adj=[
                [False, False, True],
                [False, False, False],  # node 1 has no outgoing link
                [False, False, False],
            ],
            weights=np.zeros((3, 3)),
            out_adj=[[False], [False], [True]],
        )
        net.weights[0, 2] = 0.5
        net.step(np.array([3.0, 3.0]))  # both spike
        net.step(np.array([0.0, 0.0]))  # E2 = 0.5 - 1 = -0.5
        assert net.weights[0, 2] == pytest.approx(0.5 + 0.5)

    def test_weights_never_appear_off_adjacency(self):
        net = HomeostaticReservoir(ReservoirConfig(n_inputs=3), seed=11)
        rng = np.random.default_rng(0)
        for _ in range(300):
            net.step(rng.random(3) * 3.0)
        assert np.all(net.weights[~net.adjacency] == 0.0)


# ---------------------------------------------------------------------------
# Delivery timing: one-step delay, receipt-time weights
# ---------------------------------------------------------------------------


class TestDeliveryTiming:
    def chain_net(self):
        # Node 0 -> node 1 with weight 1.5; input drives node 0 only.
        return make_manual_net(
            input_adj=[[True, False]],
            adj=[[False, True], [False, False]],
            weights=[[0.0, 1.5], [0.0, 0.0]],
            out_adj=[[True], [False]],
        )

    def test_one_step_delay(self):
        net = self.chain_net()
        s1 = net.step(np.array([3.0]))  # node 0 spikes
        assert s1.spiked[0]
        assert s1.x[1] == pytest.approx(0.0)  # nothing received yet
        s2 = net.step(np.array([0.0]))  # spike arrives now
        assert s2.x[1] == pytest.approx(1.5)

    def test_delivery_uses_receipt_time_weights(self):
        """The weight 0 -> 1 changes *between* emission and receipt; per the
        released code (`get_acts` integrates spikes through the current wmat),
        the delivered amount is the value at receipt time.

        Timeline (input 3.0 every step, input weight 1, leak 0.25):
          step 1: node 0 spikes (x' = 1).  W[0,1] still 1.5.
          step 2: node 0 spikes again (x = 1*0.75 + 3 = 3.75, x' = 1.75).
                  node 1 receives 1.5, E1 = 0.5 -> T1 = 1.005, and W[0,1]
                  drops to 1.5 - 0.5 = 1.0 (node 0 spiked at step 1).
          step 3: node 1 integrates 1.5*0.75 + 1.0 = 2.125 through the
                  *updated* weight (NOT 1.125 + 1.5 = 2.625, which would be
                  emission-time). Threshold is 2*1.005 = 2.01, so it spikes:
                  x' = 2.125 - 2.01 = 0.115.
        """
        net = self.chain_net()
        net.step(np.array([3.0]))
        s2 = net.step(np.array([3.0]))
        assert s2.spiked[0]
        assert s2.error[1] == pytest.approx(0.5)
        assert net.targets[1] == pytest.approx(1.005)
        assert net.weights[0, 1] == pytest.approx(1.0)  # update did land
        s3 = net.step(np.array([3.0]))
        assert s3.spiked[1]
        assert s3.x[1] == pytest.approx(2.125 - 2.01)  # 0.115, receipt-time

    def test_no_spike_no_delivery(self):
        net = self.chain_net()
        net.step(np.array([1.5]))  # below threshold, no spike
        s2 = net.step(np.array([0.0]))
        assert s2.x[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Effector readout
# ---------------------------------------------------------------------------


class TestEffectors:
    def test_proportion_of_spiking_in_neighbors(self):
        # Output 0 listens to nodes 0-2; output 1 listens to node 2 only.
        net = make_manual_net(
            input_adj=np.eye(3, dtype=bool),
            adj=np.zeros((3, 3), dtype=bool),
            weights=np.zeros((3, 3)),
            out_adj=[[True, False], [True, False], [True, True]],
        )
        s = net.step(np.array([3.0, 3.0, 0.0]))  # nodes 0, 1 spike
        assert s.outputs[0] == pytest.approx(2.0 / 3.0)
        assert s.outputs[1] == pytest.approx(0.0)

    def test_same_step_readout(self):
        # Effectors reflect this step's spikes (no one-step delay).
        net = make_manual_net(
            input_adj=[[True]],
            adj=[[False]],
            weights=[[0.0]],
            out_adj=[[True]],
        )
        s = net.step(np.array([3.0]))
        assert s.spiked[0] and s.outputs[0] == pytest.approx(1.0)

    def test_zero_in_degree_output_is_zero(self):
        net = make_manual_net(
            input_adj=[[True]],
            adj=[[False]],
            weights=[[0.0]],
            out_adj=[[False]],  # output has no incoming links
        )
        s = net.step(np.array([3.0]))
        assert s.outputs[0] == 0.0


# ---------------------------------------------------------------------------
# Learning toggle and invariants
# ---------------------------------------------------------------------------


class TestLearningToggle:
    def test_frozen_weights_and_targets(self):
        net = HomeostaticReservoir(ReservoirConfig(n_inputs=4), seed=12)
        net.learning_enabled = False
        w0, t0 = net.weights.copy(), net.targets.copy()
        rng = np.random.default_rng(1)
        spiked_any = False
        for _ in range(200):
            s = net.step(rng.random(4) * 3.0)
            spiked_any = spiked_any or bool(s.spiked.any())
        assert spiked_any  # dynamics still run
        assert np.array_equal(net.weights, w0)
        assert np.array_equal(net.targets, t0)


class TestInvariants:
    def test_long_run_state_sane(self):
        net = HomeostaticReservoir(ReservoirConfig(), seed=13)
        rng = np.random.default_rng(2)
        for _ in range(500):
            s = net.step(rng.random(62))
            assert np.all(np.isfinite(s.x))
            assert 0.0 <= s.prop_spiked <= 1.0
            assert np.all(s.outputs >= 0.0) and np.all(s.outputs <= 1.0)
        assert np.all(net.targets >= 1.0)
        assert np.all(np.isfinite(net.weights))
