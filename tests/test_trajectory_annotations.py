import unittest


class TrajectoryAnnotationTests(unittest.TestCase):
    def test_repeated_state_failures_have_inclusive_observed_intervals(self):
        from safetyjev.trajectory_annotations import state_intervals
        log = [{"step": i, "ap": {"upright": value}} for i, value in
               enumerate([True, False, False, True, False])]
        self.assertEqual(state_intervals(log, "upright", False), [
            {"start_timestep": 1, "end_timestep": 2, "end_reason": "recovered"},
            {"start_timestep": 4, "end_timestep": 4, "end_reason": "trajectory_end"},
        ])

    def test_until_rejection_uses_history_and_release_at_current_step(self):
        from safetyjev.trajectory_annotations import until_first_rejection
        def log(pairs):
            return [{"step": i, "ap": {"support": a, "closed": b}}
                    for i, (a, b) in enumerate(pairs)]
        # Earlier closure discharges this clause; reopening later does not reset it.
        self.assertIsNone(until_first_rejection(log([(True, False), (True, True), (False, False)]),
                                               "support", "closed"))
        self.assertIsNone(until_first_rejection(log([(True, False), (False, True)]), "support", "closed"))
        self.assertEqual(until_first_rejection(log([(True, False), (False, False), (True, True)]),
                                              "support", "closed"), 1)

    def test_temporal_only_unsafe_trajectory_keeps_its_violation(self):
        from safetyjev.trajectory_annotations import annotate_trace
        log = [{"step": 0, "ap": {"upright": True, "support": True, "closed": False}},
               {"step": 1, "ap": {"upright": True, "support": False, "closed": False}}]
        trace = {"violated": True, "violation_step": 1, "log": log,
                 "constraints": [{"id": "upright", "ltl": "G(upright)"},
                                 {"id": "close_before_lift", "ltl": "support U closed"}]}
        rules = [{"type": "tilted", "kind": "state_interval", "constraint_id": "upright",
                  "ltl": "G(upright)", "ap": "upright", "violating_value": False},
                 {"type": "close_before_lift", "kind": "temporal_first_rejection",
                  "constraint_id": "close_before_lift", "ltl": "support U closed",
                  "maintain_ap": "support", "release_ap": "closed"}]
        events = annotate_trace(trace, rules)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "close_before_lift")
        self.assertEqual(events[0]["start_timestep"], 1)
        self.assertNotIn("end_timestep", events[0])
        self.assertEqual(events[0]["label_kind"], "temporal_first_rejection")
        trace["violation_step"] = 0
        with self.assertRaises(ValueError):
            annotate_trace(trace, rules)

    def test_unrecognized_formula_cannot_be_reinterpreted_by_rule_id(self):
        from safetyjev.trajectory_annotations import annotate_trace
        trace = {"violated": False, "violation_step": None,
                 "constraints": [{"id": "upright", "ltl": "F(upright)"}],
                 "log": [{"step": 0, "ap": {"upright": True}}]}
        with self.assertRaises(ValueError):
            annotate_trace(trace, [{"type": "tilted", "kind": "state_interval", "constraint_id": "upright",
                                    "ltl": "G(upright)", "ap": "upright", "violating_value": False}])

    def test_joint_check_records_repeated_failure_after_until_was_satisfied(self):
        from safetyjev.trajectory_annotations import derived_intervals, until_first_rejection
        log = [{"step": i, "ap": {"support": s, "closed": c}} for i, (s, c) in enumerate([
            (True, True), (False, False), (False, False), (False, True), (False, False)])]
        self.assertIsNone(until_first_rejection(log, "support", "closed"))
        events = derived_intervals(log, [{"type": "open_while_off_support",
                                          "all_of": {"support": False, "closed": False}}])
        self.assertEqual([(e["start_timestep"], e["end_timestep"]) for e in events], [(1, 2), (4, 4)])
        self.assertTrue(all(e["origin"] == "derived_current_check" for e in events))
        self.assertNotIn("constraint_id", events[0])


if __name__ == "__main__":
    unittest.main()
