* Added ``IsaacContrib-Hop-Stand-Flat-MicroDuck``, a single-leg stand on the right foot with the walking task's
  61-wide actor input, as the first stage towards in-place hopping, with the ``single_support``,
  ``lifted_foot_height`` and ``lifted_foot_contact`` MicroDuck reward terms.
* Added :mod:`isaaclab_tasks.contrib.microduck.hop_metrics` and ``scripts/reinforcement_learning/microduck/eval_hop.py``
  to evaluate exported MicroDuck single-leg policies on Newton MJWarp at fixed BAM supply voltages, reporting falls,
  hops, flight time, centre-of-mass rise, torque saturation and the test-plan verdict per voltage.
* Added microduck_rl's MicroDuck one-foot "flamingo" pose in :mod:`isaaclab_tasks.contrib.microduck.flamingo_pose`,
  the ``robot_com_pos_w`` and ``foot_site_pos_w`` MicroDuck measurements, and
  :mod:`isaaclab_tasks.contrib.microduck.flamingo_checks` with
  ``scripts/reinforcement_learning/microduck/check_flamingo_pose.py`` to check the pose's sole placement, centre-of-mass
  margin, foot clearance and joint-limit margins on the all-collisions robot with MuJoCo Warp contacts.
* Added ``IsaacContrib-Flamingo-Flat-MicroDuck``, a port of microduck_rl's stage-1 one-foot balance
  (``Mjlab-Flamingo-Flat-MicroDuck``) that spawns in the noised flamingo pose on the all-collisions robot with
  MuJoCo Warp contacts and no BAM current limit, with the swing-foot clearance target and the push and action-rate
  caps of microduck_rl's Cycle task, and with the ``com_over_foot``, ``foot_in_contact``,
  ``foot_site_height_gaussian``, ``projected_gravity_match``, ``lateral_tilt_cost``, ``joint_vel_gaussian``,
  ``pose_target_match``, ``joint_limit_proximity`` and ``joint_torque_rate_l2`` MicroDuck reward terms and the
  ``reset_flamingo_pose`` event.
* Added :mod:`isaaclab_tasks.contrib.microduck.flamingo_metrics` and
  ``scripts/reinforcement_learning/microduck/eval_flamingo.py`` to evaluate exported MicroDuck flamingo policies on
  Newton MJWarp with microduck_rl's stage-1 success metric (single support for 5 s without a fall) per push
  magnitude, reporting fall sides and swing-foot touchdowns.
* Added ``scripts/reinforcement_learning/microduck/record_flamingo.py`` to record MP4 videos of MicroDuck flamingo
  episodes under the evaluation conditions with Newton's Warp ray-tracing camera, keeping successful or failed
  episodes, and ``set_eval_conditions`` shared with the evaluation script.
* Added ``IsaacContrib-FlamingoCycle-Flat-MicroDuck``, a port of microduck_rl's controller-driven flamingo cycle
  (``Mjlab-FlamingoCycle-Flat-MicroDuck``) that stands, lifts either foot on command, holds and lowers with one
  policy, with the ``FlamingoCommand`` twist-slot command, the ``reset_flamingo_cycle_pose`` event and the
  ``cycle_*`` blended MicroDuck reward terms.
* Added :mod:`isaaclab_tasks.contrib.microduck.flamingo_eval` and
  ``scripts/reinforcement_learning/microduck/eval_flamingo_cycle.py`` to evaluate MicroDuck flamingo-cycle policies
  with a scripted stand, lift, hold and lower per stance side, :class:`~isaaclab_tasks.contrib.microduck.mdp.FlamingoCommand`
  ``set_command`` to script the command, and a ``--cycle`` mode of ``record_flamingo.py``.
* Added ``IsaacContrib-FlamingoHop-Flat-MicroDuck``, hopping in place on the right foot of the flamingo pose and
  holding still under pushes, fine-tuned from the flamingo stage-1 policy, with the ``HopCommand`` ``[0, 0, hop]``
  twist-slot command and the ``hop_air_time``, ``hop_ground_time_cost``, ``hop_drift_cost``,
  ``still_foot_in_contact`` and ``still_joint_vel_gaussian`` MicroDuck reward terms.
* Added ``scripts/reinforcement_learning/microduck/eval_flamingo_hop.py`` with ``HopEpisodeMetrics`` and the
  ``make_hop_env`` / ``run_hop_protocol`` rollout to evaluate MicroDuck flamingo-hop policies with a scripted still,
  hop, still protocol against the hop test plan, and a ``--hop`` mode of ``record_flamingo.py``.
