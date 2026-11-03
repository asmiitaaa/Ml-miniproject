# Submission review and demonstration evidence

## Student details

The supplied files name Asmita, Ananyaraj and Soham but do not give complete names, affiliation, roll numbers or registration numbers. Add those from the team's official records. They have not been guessed.

## Report review

1. Read `report/report.md` as the editable prose version. Review every implementation statement against the final code and every empirical statement against the saved run artifacts.
2. Compile `report/report.tex` with the institution's IEEEtran template. The ReportLab PDF is a readable two-column preview; confirm the institution's required title block, margins, reference format and page limit before submission.
3. The full 30-run fixed/actuated/DQN matrix is now included. Review its DQN collision flags, uncalibrated demand and unverified training/test seed separation before making final claims. Keep the test seed list, checkpoint selection procedure and manifests with the submission.
4. Review mean waiting and duration together with completed, unfinished and uninserted vehicles. Explain any controller's low completion rate, collision or teleport. Do not claim a performance improvement from incomplete or invalid runs.
5. Use `docs/REFERENCE_VERIFICATION.md` to finish institutional database checks. Retain the original citation export and access date; update the report's pending-check notes only when evidence exists.
6. Submit the final reviewed report through the institution's Turnitin process. Record the actual similarity score and review matching text and citations. Original drafting cannot guarantee the requested <=15% score. No Turnitin upload has been made by this assistant.

## Required SUMO-GUI recording

The document requests a side-by-side baseline-versus-learned demonstration, not merely a chart animation. The 20k SUMO checkpoint has been loaded, verified and evaluated. The completed three-panel native recording is in demonstrations/three_controller_final. Baseline-only recordings validate visualization and do not complete the learned-versus-baseline requirement.

For the final evidence:

- Choose one scenario and test seed already in the results table. Use exactly its saved route, vehicle and network files for fixed, actuated and learned replays.
- Record the config/input fingerprint, controller, seed, SUMO version and DQN checkpoint fingerprint in the video title card or accompanying manifest.
- Use the same camera, map zoom, playback speed and simulation time window. Show the simulation clock and relevant signals and queues.
- Capture fixed-versus-DQN and actuated-versus-DQN; place the two synchronized replays side by side. A third-panel comparison is also acceptable if readable.
- Run the real inference adapter during learned replay. Do not replace it with random actions, a mock policy or altered prerecorded vehicle paths.
- Explain what the video shows without using it as a substitute for measured seeded statistics. Include a representative run, not only a visually favorable segment.

## Questions Soham should be ready to answer

- Why do we pair seed and route inputs rather than compare unrelated random runs?
- What is the difference between waitingTime, departDelay, timeLoss and trip duration?
- Why is CO2 in the emission XML integrated over the simulator step?
- Why are per-seed sample standard deviations different from per-vehicle variation?
- What happens to percentage improvement when the baseline metric is zero?
- Why can completed-trip averages conceal congestion at the simulation cutoff?
- What does it mean if actuated does not outperform fixed under a specific scenario?
- Which results establish software integration, and which would establish traffic-controller performance?
