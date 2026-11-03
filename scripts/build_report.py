"""Build an honest two-column report preview and editable IEEEtran source.

Run from repository root: python scripts/build_report.py --results results/provisional
ReportLab is required for the PDF preview; an installed IEEEtran LaTeX toolchain
can separately compile report/report.tex. No results are invented when absent.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
from html import escape

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Dynamic Traffic Congestion Control and Signal Optimization Using Deep Reinforcement Learning"
ABSTRACT = (
    "This project specifies a SUMO-based experiment for adaptive traffic signal control under heterogeneous traffic. "
    "A DQN is intended to retain or advance each signal phase from queue, count and waiting-time observations, "
    "subject to minimum-green and yellow intervals. The evaluation layer compares fixed-time, actuated and learned "
    "control using identical seeded scenario inputs and records waiting time, trip duration, queue statistics and CO2. "
    "This report documents the implemented evaluation and reporting workflow and separates genuine validation "
    "runs from the final research experiment. A refreshed teammate branch supplies real-SUMO training code, a "
    "one-hour demand file and 10,000- and 20,000-step SUMO-labelled checkpoints. The 20,000-step checkpoint was "
    "evaluated in 30 one-hour runs covering three controllers, two synthetic demand levels and five paired seeds. "
    "Two peak-demand DQN runs recorded collision-related teleports, which limit interpretation of speed improvements. "
    "Independent evaluation does not establish training convergence or scenario calibration. Final conclusions "
    "require reviewed peak/off-peak demand, compatible policy deployment and complete paired experiments."
)
SECTIONS = [
    ("I. Introduction", [
        "Signal timing allocates scarce junction capacity between competing approaches. A fixed program is easy to reproduce, "
        "but cannot change its split in response to an unusual queue. An actuated program uses observed arrivals to adjust green "
        "durations. Reinforcement learning offers a third approach: a policy can learn the long-term consequences of its decisions "
        "from repeated interactions with a traffic simulator. The practical question is whether that policy improves a fair baseline "
        "comparison on a specified demand distribution, rather than whether a reward curve rises during training.",
        "The study targets a mixed-traffic Indian corridor. Its planned contribution is a reproducible comparison of efficiency "
        "and emissions, including the stronger actuated baseline. Current engineering deliverables support that comparison; "
        "the final empirical claim depends on the simulation and learned-policy inputs described below."
    ]),
    ("II. Related Work", [
        "Haydari and Yilmaz [1] survey deep RL transport applications by problem formulation and algorithm. Noaeen et al. [2] "
        "review network-level traffic signal research and identify a gap between research systems and transport practice. These "
        "reviews motivate explicit scenario definitions and reproducibility rather than broad claims about universal superiority.",
        "Liang et al. [4] represent traffic spatially and combine several value-learning refinements to adjust cycle timings. "
        "Their architecture differs from the small feedforward DQN planned here. Bouktif et al. [5] study consistency between "
        "state and reward using DDQN and prioritized replay, supporting a clear connection between the waiting-time reward "
        "and the delay measures used in evaluation.",
        "Chu et al. [3] address large networks using decentralized advantage actor-critic agents with neighboring information. "
        "Wu et al. [7] use recurrent multi-agent deterministic policy gradients, centralized critics and decentralized execution. "
        "Zhu et al. [8] use broad networks with ridge regression and an interaction mechanism to reduce training cost. "
        "These are coordination methods; independently deploying the same DQN at several signals does not implement their communication schemes.",
        "Mao et al. [6] provide a useful algorithm-comparison reference, but its primary article remains to be checked through "
        "institutional IEEE access. It is retained as a bibliographic lead without importing detailed outcome claims. Han et al. [9] "
        "highlight training/testing separation and simulator fidelity as deployment concerns. Tan et al. [10] address noisy "
        "traffic observations through perturbed-state training. The present evaluation does not establish robustness to sensor failure or field transfer."
    ]),
    ("III. Research Question and Scope", [
        "The primary question is whether a traffic-trained DQN lowers completed-trip waiting time relative to fixed-time and "
        "actuated control on paired peak and off-peak scenarios. Secondary outcomes are completed-trip duration, mean and "
        "maximum approach queue, and total simulated CO2. Completion, pending insertions, teleports and collisions are quality "
        "indicators needed to interpret those outcomes.",
        "Heterogeneity is a modeling assumption until calibrated against traffic observations. The specified vehicle proportions "
        "are 50% two-wheelers, 15% auto-rickshaws, 30% cars and 5% buses, with 0.4 m lateral resolution. The supplied vehicle "
        "file also defines bicycles; defining that type does not mean it should appear in the four-class experimental mix. "
        "The supplied network has six signal programs, whereas the planning document proposes three to five intersections. "
        "The team must document a reviewed subset or justify the expanded corridor before final evaluation."
    ]),
    ("IV. Methodology", [
        "A. Network and demand. The road network is imported from OpenStreetMap and reviewed for lane connectivity, "
        "permitted turns, signal programs and vehicle-class access. Peak and off-peak routes are frozen once per scenario "
        "and reused without modification across all controllers. Input hashes and the simulator version are recorded with each "
        "run. RandomTrips-based demand generation and field calibration remain simulation-phase responsibilities; a provisional "
        "test route set is not calibrated field evidence.",
        "B. Learning environment. The specified Gymnasium environment communicates through TraCI. Its observation includes "
        "per-approach counts, halting queues, cumulative waiting and phase identity. Every five simulated seconds the action "
        "retains the current phase or advances in its prescribed cycle. Minimum green and yellow transitions are enforced by "
        "the environment. Reward is the negative total approach waiting time. The refreshed teammate branch provides real "
        "TraCI reset, stepping and observation methods and a SUMO-mode training configuration. The model observes one "
        "selected incoming lane from each of four approaches at a single junction. Other deployment layouts require explicit "
        "lane-order and phase-map validation; the policy cannot be assumed to generalize to every signal in the corridor.",
        "C. Training protocol. The planned Stable-Baselines3 DQN uses PyTorch, experience replay, a periodically updated "
        "target network and epsilon-greedy exploration. Starting settings are approximately 0.001 learning rate, 50,000 replay "
        "transitions and one-hour episodes. Hyperparameters and checkpoint selection use training/validation seeds only; held-out "
        "test seeds must not be used to select the best model. Independent per-signal deployment must match the observation "
        "encoding used at training time. The available checkpoint filenames indicate 10,000 and 20,000 training steps; "
        "training curves, convergence evidence and checkpoint-selection records are not supplied. Training uses 0.5 m lateral "
        "resolution, whereas this evaluation uses the documented 0.4 m. This is a distribution shift that must be reported "
        "and addressed before a final controlled experiment. Older mock checkpoints remain excluded from traffic claims.",
        "D. Comparative experiment. For each demand scenario, run all three controllers on five to ten identical test seeds. "
        "Fixed and actuated programs share network geometry, vehicle inputs, run horizon, lateral resolution and simulator "
        "settings. An external DQN runner must emit the same output contract. Raw XML, queue telemetry, command logs and "
        "provenance are retained. The actuated-versus-fixed check is an empirical diagnostic: a failure triggers investigation "
        "of detection coverage and timings, not editing measurements to force the expected ranking."
    ]),
    ("V. Metrics and Statistical Reporting", [
        "Waiting time is SUMO tripinfo waitingTime averaged over completed, non-vaporized trips [11]; duration is arrival minus "
        "actual departure as recorded by tripinfo. DepartDelay is a separate insertion measure and must not be mistaken for "
        "junction waiting. Unfinished trips and vehicles unable to insert are reported because completed-trip means can improve "
        "artificially when the longest trips have not finished.",
        "Queue telemetry samples the number of halting vehicles, including zero counts, at each unique incoming approach "
        "at every simulator step. The mean is over approach-time samples and the maximum is over those samples. Queue "
        "counts are not queue distances. CO2 rates in emission output are integrated over the simulation step and converted "
        "from milligrams to kilograms [12], including vehicles still present at the run cutoff. Emission classes and simulator "
        "versions must be held fixed and reported; modeled emissions are not physical measurements.",
        "For each controller and scenario the report gives the mean and sample standard deviation across seeds, not across "
        "individual vehicles. For each matched seed, percentage improvement is 100*(baseline - controller)/baseline; these "
        "paired percentages are then summarized. A zero baseline yields an undefined percentage and is counted explicitly "
        "rather than replaced with zero. Positive improvement means the lower-is-better metric decreased. Five seeds provide "
        "an exploratory comparison, not proof of statistical significance."
    ]),
    ("VI. Implementation and Evidence Status", [
        "The evaluation package parses tripinfo and emission XML, records live approach queues through TraCI, validates "
        "paired demand fingerprints, and writes run-level, summary and paired-improvement CSV files with diagnostic flags. "
        "Matplotlib charts and a Streamlit dashboard expose scenario, controller and seed comparisons alongside completion "
        "quality. A report builder creates this editable report from the available measured summaries.",
        "The original clone supplied road geometry, vehicle definitions and notes. Refreshing the teammate branch exposed "
        "additional real-SUMO environment code, a one-hour route file and SUMO-labelled model checkpoints alongside older "
        "mock checkpoints. The 20,000-step SUMO checkpoint was selected before inspecting evaluation scores because it "
        "was the latest available model. All twelve loaded tensors match the original stored values; buffering tensor "
        "archive streams resolves a Windows reader incompatibility without changing model weights or checkpoint bytes. "
        "Thirty full-hour three-controller runs, tables, charts, dashboard views and a matched 90-second sampled GUI "
        "comparison are supplied. Each controller acts at the trained junction; the other five signals retain fixed plans. "
        "Training convergence, calibrated field demand and transfer to every corridor junction remain unverified."
    ]),
    ("VII. Limitations and Completion Plan", [
        "Before final submission, calibrate and review peak/off-peak inputs, resolve the observed DQN collision events, "
        "verify minimum-green and network clearance behavior, and document disjoint training and test seeds. Inspect every "
        "quality warning, explain the actuated baseline outcome and retain failed runs. Matched baseline-versus-DQN "
        "SUMO-GUI captures record scenario, seed, checkpoint and camera settings. Report simulation-only limits "
        "and avoid claims of deployment readiness.",
        "All 22 supplied literature entries have an audit trail; corrected issue years, pages and article identifiers are "
        "recorded separately. Several primary IEEE/Elsevier checks and institutional Scopus verification remain pending. "
        "The report uses newly written prose, but the requested Turnitin similarity threshold of at most 15% requires an "
        "actual institutional pre-check. No similarity score is asserted. Add the full names, roll/registration numbers and "
        "institutional affiliation, review the final IEEEtran compilation and update the results section only from genuine outputs."
    ]),
]

REFERENCES = [
    ('haydari', 'A. Haydari and Y. Yilmaz, "Deep reinforcement learning for intelligent transportation systems: A survey," IEEE Trans. Intell. Transp. Syst., vol. 23, no. 1, pp. 11-32, 2022.', '10.1109/TITS.2020.3008612'),
    ('noaeen', 'M. Noaeen et al., "Reinforcement learning in urban network traffic signal control: A systematic literature review," Expert Syst. Appl., vol. 199, Art. no. 116830, 2022.', '10.1016/j.eswa.2022.116830'),
    ('chu', 'T. Chu, J. Wang, L. Codeca and Z. Li, "Multi-agent deep reinforcement learning for large-scale traffic signal control," IEEE Trans. Intell. Transp. Syst., vol. 21, no. 3, pp. 1086-1095, 2020.', '10.1109/TITS.2019.2901791'),
    ('liang', 'X. Liang, X. Du, G. Wang and Z. Han, "A deep reinforcement learning network for traffic light cycle control," IEEE Trans. Veh. Technol., vol. 68, no. 2, pp. 1243-1253, 2019.', '10.1109/TVT.2018.2890726'),
    ('bouktif', 'S. Bouktif, A. Cheniki, A. Ouni and H. El-Sayed, "Deep reinforcement learning for traffic signal control with consistent state and reward design approach," Knowl.-Based Syst., vol. 267, Art. no. 110440, 2023.', '10.1016/j.knosys.2023.110440'),
    ('mao', 'F. Mao, Z. Li and L. Li, "A comparison of deep reinforcement learning models for isolated traffic signal control," IEEE Intell. Transp. Syst. Mag., vol. 15, no. 1, pp. 160-180, 2023. [Primary article check pending.]', '10.1109/MITS.2022.3144797'),
    ('wu', 'T. Wu et al., "Multi-agent deep reinforcement learning for urban traffic light control in vehicular networks," IEEE Trans. Veh. Technol., vol. 69, no. 8, pp. 8243-8256, 2020.', '10.1109/TVT.2020.2997896'),
    ('zhu', 'R. Zhu, L. Li, S. Wu, P. Lv, Y. Li and M. Xu, "Multi-agent broad reinforcement learning for intelligent traffic light control," Inf. Sci., vol. 619, pp. 509-525, 2023.', '10.1016/j.ins.2022.11.062'),
    ('han', 'Y. Han, M. Wang and L. Leclercq, "Leveraging reinforcement learning for dynamic traffic control: A survey and challenges for field implementation," Commun. Transp. Res., vol. 3, Art. no. 100104, 2023.', '10.1016/j.commtr.2023.100104'),
    ('tan', 'K. L. Tan, A. Sharma and S. Sarkar, "Robust deep reinforcement learning for traffic signal control," J. Big Data Anal. Transp., vol. 2, pp. 263-274, 2020.', '10.1007/s42421-020-00029-6'),
    ('tripinfo', 'Eclipse SUMO, "TripInfo," SUMO Documentation, accessed Nov. 3, 2026.', 'https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html'),
    ('emissions', 'Eclipse SUMO, "Emission Output," SUMO Documentation, accessed Nov. 3, 2026.', 'https://sumo.dlr.de/docs/Simulation/Output/EmissionOutput.html'),
]

def tex_escape(s):
    return ''.join({'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}.get(c,c) for c in s)

def read_results(path):
    if not path:
        return [], ["No results directory supplied; numerical evaluation is pending."]
    summary = path/'summary.csv'
    if not summary.is_file():
        return [], [f"No measured summary.csv found in {path.name}; numerical evaluation is pending."]
    with summary.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    quality_path = path/'quality_checks.json'
    quality = json.loads(quality_path.read_text(encoding='utf-8')) if quality_path.exists() else {}
    warnings = quality.get('warnings', ["No quality-check artifact is available."])
    evidence=[]
    runs_path=path/'runs.csv'
    if runs_path.is_file():
        with runs_path.open(encoding='utf-8', newline='') as f: runs=list(csv.DictReader(f))
        seeds=sorted({int(r['seed']) for r in runs})
        n_scenarios=len({r['scenario'] for r in runs})
        counts=', '.join(sorted({str(r['n_seeds']) for r in rows}))
        controllers=sorted({r['controller'] for r in runs})
        evidence.append(f"The measured dataset contains {len(runs)} real-SUMO runs across {n_scenarios} demand settings and seeds {', '.join(map(str,seeds))}. Controllers: {', '.join(controllers)}. Each table reports the seed mean (sample standard deviation), with seed counts {counts} per controller/scenario.")
        if 'dqn' not in controllers:
            evidence.append("DQN cells remain pending in this dataset. SUMO-labelled checkpoints are available in the refreshed teammate branch, but were not executed as part of these recorded baseline runs.")
        if runs and all('completion_rate' in r for r in runs):
            completion=[float(r['completion_rate']) for r in runs]
            completion_note = "All requested vehicles completed within the horizon." if min(completion)==1 else "Incomplete trips censor the delay means."
            evidence.append(f"Completed-trip fractions range from {100*min(completion):.2f}% to {100*max(completion):.2f}%. {completion_note} The runs record {sum(int(r.get('teleports',0)) for r in runs)} teleport starts and {sum(int(r.get('collisions',0)) for r in runs)} colliding-vehicle detections. The latter counts involved vehicles, not distinct collision events.")
            unsafe=[r for r in runs if int(r.get('collisions',0)) or int(r.get('teleports',0))]
            if unsafe:
                labels=', '.join(f"{r['scenario']}/{r['controller']}/seed {r['seed']}" for r in unsafe)
                evidence.append(f"Safety flags occur in {labels}. Collision-related teleports alter journeys, so speed and emissions improvements in the affected scenario are diagnostic measurements, not evidence of a safe policy benefit. All seeds remain in the tables; no failed seed was discarded.")
        manifests=list((path/'raw').rglob('manifest.json'))
        if manifests:
            one=json.loads(manifests[0].read_text(encoding='utf-8'))
            evidence.append(f"The recorded simulator is {one.get('sumo_version',['','unknown'])[-1]}, with a {one.get('duration_s','unknown')}-second horizon and {one.get('step_length_s','unknown')}-second step. Demand remains synthetic and uncalibrated; all controllers share the trained-junction scope.")
    checks_path=path/'baseline_checks.csv'
    if checks_path.is_file():
        with checks_path.open(encoding='utf-8', newline='') as f: checks=list(csv.DictReader(f))
        for r in checks:
            if r['metric']=='avg_waiting_time_s':
                evidence.append(f"For {r['scenario']}, the paired actuated-versus-fixed waiting-time reduction is {float(r['improvement_pct_mean']):.2f}% (sample SD {float(r['improvement_pct_std']):.2f} percentage points; {r['n_pairs']} pairs); the diagnostic status is {r['status']}. This is a baseline integration result, not a DQN result or evidence of calibrated field demand.")
    paired_path=path/'paired_improvements.csv'
    if paired_path.is_file():
        with paired_path.open(encoding='utf-8',newline='') as f: paired=list(csv.DictReader(f))
        for r in paired:
            if r['controller']=='dqn' and r['metric']=='avg_waiting_time_s':
                evidence.append(f"For {r['scenario']}, measured DQN waiting-time reduction relative to {r['baseline']} is {float(r['improvement_pct_mean']):.2f}% (sample SD {float(r['improvement_pct_std']):.2f} percentage points; {r['n_pairs']} pairs). These values must be interpreted together with the safety and calibration limitations, especially the flagged peak runs.")
    return rows, ["Measured validation results only. These values are not a final calibrated corridor performance claim.", *evidence, *warnings]

def value(row, metric):
    try:
        mean, std = float(row[metric+'_mean']), float(row[metric+'_std'])
        import math
        if not math.isfinite(mean): return 'undefined'
        return f'{mean:.2f} ({std:.2f})' if math.isfinite(std) else f'{mean:.2f} (SD n/a)'
    except (ValueError, KeyError, TypeError):
        return 'unavailable'

def result_tables(rows):
    tables=[]
    for scenario in sorted({r['scenario'] for r in rows}):
        part={r['controller']:r for r in rows if r['scenario']==scenario}
        controllers=['fixed','actuated','dqn']
        data=[['Metric', *controllers]]
        for key,label in [('avg_waiting_time_s','Waiting (s)'),('avg_trip_duration_s','Duration (s)'),('avg_queue_length','Mean queue (veh)'),('max_queue_length','Max queue (veh)'),('co2_kg','CO2 (kg)')]:
            data.append([label,*[value(part[c],key) if c in part else 'pending' for c in controllers]])
        tables.append((scenario,data,'Mean (sample SD). '+'; '.join(f"{c}: n={part[c]['n_seeds']}" if c in part else f'{c}: pending' for c in controllers)))
    return tables

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',type=Path); ap.add_argument('--output',type=Path,default=ROOT/'report')
    args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    rows,warnings=read_results(args.results); tables=result_tables(rows)
    status="IMPLEMENTATION REPORT - FINAL CALIBRATED EVALUATION PENDING"
    sections=SECTIONS.copy()
    sections.insert(6,('VII. Measured Validation Results', warnings))
    sections[-1]=('VIII. Limitations and Completion Plan',sections[-1][1])
    md=[f'# {TITLE}',f'Asmita | Ananyaraj | Soham\n\n3 November 2026\n\n**{status}**', '## Abstract', ABSTRACT,
        'Keywords: Deep reinforcement learning; DQN; traffic signal control; SUMO; reproducible evaluation.']
    tex=[r'\documentclass[conference]{IEEEtran}',r'\usepackage[T1]{fontenc}',r'\usepackage{url}',r'\usepackage{booktabs}',r'\begin{document}',
         r'\title{'+tex_escape(TITLE)+'}',r'\author{\IEEEauthorblockN{Asmita, Ananyaraj, and Soham}\IEEEauthorblockA{Institution and student identifiers to be added after student review}}',r'\maketitle',
         r'\begin{abstract}'+tex_escape(ABSTRACT)+r'\end{abstract}',r'\begin{IEEEkeywords}Deep reinforcement learning, DQN, traffic signal control, SUMO, reproducible evaluation.\end{IEEEkeywords}',
         r'\noindent\textbf{Status: '+tex_escape(status)+r'.}']
    for title,paras in sections:
        md.extend(['## '+title,*paras]); tex.append(r'\section{'+tex_escape(re.sub(r'^[IVX]+\. ','',title))+'}')
        for p in paras: tex.append(tex_escape(p)+'\n')
        if 'Measured Validation' in title:
            for scenario,data,note in tables:
                md.extend(['### '+scenario,'| '+' | '.join(data[0])+' |','|'+'|'.join(['---']*len(data[0]))+'|',*['| '+' | '.join(r)+' |' for r in data[1:]],note])
                tex.extend([r'\begin{table}[ht]\centering\scriptsize',r'\caption{'+tex_escape(scenario)+r': measured validation only}',r'\begin{tabular}{l'+'c'*(len(data[0])-1)+'}',r'\toprule',*[' & '.join(map(tex_escape,r))+r' \\' for r in data],r'\bottomrule\end{tabular}',r'\par '+tex_escape(note),r'\end{table}'])
    md.append('## References'); tex.append(r'\begin{thebibliography}{10}')
    for i,(key,citation,doi) in enumerate(REFERENCES,1):
        url=doi if doi.startswith('https:') else 'https://doi.org/'+doi
        md.append(f'{i}. {citation} {url}')
        tex.append(r'\bibitem{'+key+'} '+tex_escape(citation)+r' \url{'+url+'}.')
    tex.extend([r'\end{thebibliography}',r'\end{document}'])
    (out/'report.md').write_text('\n\n'.join(md),encoding='utf-8'); (out/'report.tex').write_text('\n'.join(tex),encoding='utf-8')
    build_pdf(out/'report.pdf',sections,tables,status)
    (out/'BUILD_NOTES.md').write_text('This PDF is a two-column IEEE-style report generated with ReportLab because a LaTeX compiler was not available. Compile report.tex with IEEEtran for the final institutional template. The report contains all 30 three-controller one-hour validation runs. DQN peak collision flags, uncalibrated demand, single-junction scope, missing convergence/seed provenance, institutional Scopus/IEEE checks and Turnitin pre-check remain open. Add student identifiers and affiliation before submission.\n',encoding='utf-8')
    print(f'Report outputs: {out}')

def build_pdf(path,sections,tables,status):
    from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,KeepTogether,NextPageTemplate
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    w,h=A4; margin=42; gap=17; cw=(w-2*margin-gap)/2
    styles={
        'body':ParagraphStyle('body',fontName='Times-Roman',fontSize=10,leading=11.6,alignment=TA_JUSTIFY,spaceAfter=5.3),
        'head':ParagraphStyle('head',fontName='Times-Bold',fontSize=10.5,leading=12.5,alignment=TA_CENTER,spaceBefore=9,spaceAfter=6,keepWithNext=True),
        'ref':ParagraphStyle('ref',fontName='Times-Roman',fontSize=8,leading=9.6,spaceAfter=5.5),
        'cell':ParagraphStyle('cell',fontName='Times-Roman',fontSize=8,leading=9.5),
        'small':ParagraphStyle('small',fontName='Times-Italic',fontSize=8,leading=9.5,spaceAfter=5),
    }
    def p(text,style='body'): return Paragraph(escape(text),styles[style])
    def footer(c,doc):
        c.setFont('Times-Roman',8);c.setFillColor(colors.HexColor('#555555'));c.drawCentredString(w/2,24,f'Soham - implementation and validation report | 3 November 2026 | {doc.page}')
    def first(c,doc):
        footer(c,doc)
        title=Paragraph(escape(TITLE),ParagraphStyle('title',fontName='Times-Roman',fontSize=18,leading=21,alignment=TA_CENTER))
        _,th=title.wrap(w-2*margin,80);title.drawOn(c,margin,h-margin-th)
        c.setFont('Times-Roman',10);c.setFillColor(colors.black);c.drawCentredString(w/2,h-margin-th-18,'Asmita | Ananyaraj | Soham')
        c.setFont('Times-Italic',8);c.drawCentredString(w/2,h-margin-th-32,'Institution and student identifiers require student review')
        c.setFont('Times-Bold',8);c.setFillColor(colors.HexColor('#8b4513'));c.drawCentredString(w/2,h-margin-th-49,status)
    doc=BaseDocTemplate(str(path),pagesize=A4,leftMargin=margin,rightMargin=margin,topMargin=margin,bottomMargin=38)
    frames=lambda top:[Frame(margin,38,cw,top-38,id='left'),Frame(margin+cw+gap,38,cw,top-38,id='right')]
    doc.addPageTemplates([PageTemplate(id='First',frames=frames(h-165),onPage=first),PageTemplate(id='Body',frames=frames(h-margin),onPage=footer)])
    story=[NextPageTemplate('Body'),p('Abstract','head'),p(ABSTRACT),p('Index Terms - Deep reinforcement learning, DQN, SUMO, traffic signal control, reproducible evaluation.','small')]
    for title,paras in sections:
        story.append(p(title,'head'));story.extend(p(x) for x in paras)
        if 'Measured Validation' in title:
            for scenario,data,note in tables:
                colwidths=[cw*.34]+[cw*.66/(len(data[0])-1)]*(len(data[0])-1)
                t=Table([[p(cell,'cell') for cell in row] for row in data],colWidths=colwidths,hAlign='CENTER',repeatRows=1)
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9edf2')),('LINEABOVE',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.5,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
                story.append(KeepTogether([p('Measured validation: '+scenario,'small'),t,Spacer(1,4),p(note,'small')]))
    story.append(p('References','head'))
    for i,(_,citation,doi) in enumerate(REFERENCES,1):
        url=doi if doi.startswith('https:') else 'https://doi.org/'+doi
        label=doi.replace('https://sumo.dlr.de/docs/','sumo.dlr.de/docs/')
        story.append(Paragraph(escape(f'[{i}] {citation} ')+f'<link href="{url}" color="#234c76">'+escape(label)+'</link>.',styles['ref']))
    doc.build(story)

if __name__=='__main__': main()
