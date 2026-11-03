# Literature survey

Original synthesis for student review, 3 November 2026. Bracketed numbers match `REFERENCE_VERIFICATION.md` (the pasted 22-item list). These are findings from published work, not results of this project.

## Problem formulation and evidence quality

Traffic signal control can be formulated as repeated decisions made from observations of approaching traffic. Haydari and Yilmaz [1] organize deep reinforcement learning applications by control model, algorithm and experimental setting. Their survey makes clear that the representation of traffic and the simulator are part of the experiment, rather than neutral implementation details. Noaeen et al. [2] review network-level control and identify a gap between algorithm research and transport practice. Together, these surveys support a focused student project that documents its scenario inputs, assumptions and evaluation procedure instead of claiming a universally superior controller.

## Value-based and state-reward approaches

Liang et al. [4] encode traffic with a spatial grid and adjust signal durations with a deep value-based model. Their implementation combines several DQN refinements, so their findings cannot be attributed to an unmodified small feedforward DQN alone. Li et al. [6] study a cyclic keep-or-advance action in single- and multi-intersection settings. The latter includes shared/global information, which differs from independently deploying one local policy at each signal. Bouktif et al. [7] explicitly align state and reward design, evaluating a DDQN approach with prioritized replay. For this project, these studies motivate clear definitions of observation, action and waiting-time reward; they do not remove the need to validate the environment's minimum-green and yellow-interval constraints.

## Coordination and scalability

Chu et al. [3] use decentralized advantage actor-critic learning with information from neighboring agents, addressing partial observability and the large joint action space. Wu et al. [5] use recurrent multi-agent deterministic policy gradients with centralized critics and decentralized execution, incorporating pedestrians and bus priorities. Zhu et al. [9] replace deep network updates with a broad network and ridge regression and add a dynamic interaction mechanism. These approaches demonstrate that network control has coordination and training-cost concerns beyond single-junction control. Independent corridor deployment in this project is deliberately narrower: it does not implement communication or claim the benefits of a cooperative multi-agent method.

Gu et al. [16] partition a large road network into constrained regions and use an adaptive branching dueling Q-network; Li et al. [18] use dynamic zoning and spatio-temporal features. Liu and Ding [19] combine local learning with consensus updates. Federated PPO in Li et al. [17] shares model information across environments without collecting raw experience centrally. These methods are relevant future extensions, but they should not be described as implemented in the present repository.

## Objectives, deployment and limits

EMVLight [10] combines emergency vehicle routing with traffic signal pre-emption. Ibrokhimov et al. [13] preserve cyclic operation and explicit timing constraints using a pressure-based formulation; their experiments use CityFlow, illustrating why a blanket statement that every cited study uses SUMO is inaccurate. Guzmán et al. [15] use information from neighboring intersections in a distributed controller. Tan et al. [20] train with perturbed state observations to address sensing noise. Han et al. [11] emphasize the costs of online exploration and the dependence of offline policies on simulator fidelity, and identify insufficient separation of training and testing in published evaluation practice.

The resulting research question is therefore specific: on a documented Indian mixed-traffic scenario, does a traffic-trained DQN reduce waiting time relative to both a fixed program and SUMO's actuated program, while also improving trip duration, queue statistics and emissions? Test demand and seeds must be held fixed across controllers; unfinished trips and uninserted demand must be reported to expose selection bias. Emissions are reported separately from the waiting-time training reward. A simulator-only finding will not establish field safety or transfer to a physical junction.

## Source boundaries

The full audit contains primary links, DOI metadata and pending checks. Mao et al. [8], Zhang et al. [12], Yang et al. [21] and Wang et al. [22] remain useful leads but are excluded from detailed methodological claims here until the primary articles are accessible. Scopus/IEEE Xplore institutional checks and Turnitin similarity certification remain pending. No numerical performance gains from another paper are presented as measurements of this project.
