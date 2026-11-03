"""Scientific integrity checks using independent, hand-computable fixtures."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pandas as pd

from traffic_eval.metrics import parse_tripinfo, parse_emissions, parse_queues
from traffic_eval.reporting import summarize, validate_runs
from traffic_eval.runner import load_config, fingerprint, check_inputs, run_one, _start_sumo


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)

    def tearDown(self): self.tmp.cleanup()

    def test_trip_means_exclude_unfinished_and_teleported(self):
        file=self.root/'tripinfo.xml'
        file.write_text('<tripinfos><tripinfo arrival="10" waitingTime="4" duration="10"/><tripinfo arrival="20" waitingTime="8" duration="14"/><tripinfo arrival="-1" waitingTime="90" duration="100"/><tripinfo arrival="15" waitingTime="80" duration="90" vaporized="teleport"/></tripinfos>')
        got=parse_tripinfo(file)
        self.assertEqual(got['completed_trips'],2); self.assertEqual(got['unfinished_trips'],2)
        self.assertEqual(got['avg_waiting_time_s'],6); self.assertEqual(got['avg_trip_duration_s'],12)

    def test_emission_rate_conversion_and_nonunit_step(self):
        file=self.root/'emissions.xml'
        file.write_text('<emission-export><timestep time="0"><vehicle CO2="1000000"/><vehicle CO2="2000000"/></timestep><timestep time="0.5"><vehicle CO2="1000000"/></timestep></emission-export>')
        self.assertEqual(parse_emissions(file,.5)['co2_kg'],2)

    def test_queue_time_and_approach_average(self):
        file=self.root/'queues.csv'
        file.write_text('time,tls_id,approach_id,queue_length\n1,a,n,0\n1,a,s,4\n2,a,n,2\n2,a,s,6\n')
        got=parse_queues(file)
        self.assertEqual(got['avg_queue_length'],3); self.assertEqual(got['max_queue_length'],6)
        self.assertEqual(got['avg_total_queue_length'],6)

    def test_queue_missing_zero_approach_is_rejected(self):
        file=self.root/'queues.csv'; file.write_text('time,tls_id,approach_id,queue_length\n1,a,n,0\n1,a,s,4\n2,a,s,6\n')
        with self.assertRaisesRegex(ValueError,'all approaches'): parse_queues(file)


class PairingTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(scenario='peak',controller=c,seed=s,demand_hash=f'h{s}',avg_waiting_time_s=v,
             avg_trip_duration_s=v,avg_queue_length=v,max_queue_length=v,co2_kg=v)
             for c,s,v in [('fixed',1,10),('actuated',1,5),('fixed',2,100),('actuated',2,90)]])

    def test_improvement_is_seed_paired_not_ratio_of_means(self):
        summary,paired=summarize(self.frame())
        row=paired[(paired.controller=='actuated') & (paired.metric=='avg_waiting_time_s')].iloc[0]
        self.assertEqual(row.improvement_pct_mean,30) # (50% + 10%)/2
        self.assertAlmostEqual(row.improvement_pct_std,28.2842712475)
        self.assertEqual(summary.n_seeds.tolist(),[2,2])

    def test_duplicate_rows_rejected(self):
        frame=self.frame(); frame=pd.concat([frame,frame.iloc[:1]])
        with self.assertRaisesRegex(ValueError,'Duplicate'): summarize(frame)

    def test_mismatched_demand_rejected(self):
        frame=self.frame(); frame.loc[1,'demand_hash']='other'
        with self.assertRaisesRegex(ValueError,'fingerprints differ'): summarize(frame)

    def test_unpaired_seed_sets_rejected(self):
        frame=self.frame(); frame.loc[1,'seed']=3
        with self.assertRaisesRegex(ValueError,'different seed sets'): summarize(frame)

    def test_empty_fingerprint_rejected(self):
        frame=self.frame(); frame.loc[0,'demand_hash']=''
        with self.assertRaisesRegex(ValueError,'nonempty'): summarize(frame)

    def test_zero_baseline_does_not_claim_infinite_improvement(self):
        frame=self.frame(); frame.loc[0,'avg_waiting_time_s']=0
        _,paired=summarize(frame)
        row=paired[(paired.controller=='actuated') & (paired.metric=='avg_waiting_time_s')].iloc[0]
        self.assertEqual(row.n_pairs,1); self.assertEqual(row.improvement_pct_mean,10)


class ProvenanceTests(unittest.TestCase):
    def test_hash_changes_when_demand_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'net.xml').write_text('<net/>'); (root/'routes.xml').write_text('<routes/>')
            (root/'run.sumocfg').write_text('<configuration><input><net-file value="net.xml"/><route-files value="routes.xml"/></input></configuration>')
            cfg={'_root':tmp}; scenario={'sumocfg':'run.sumocfg'}
            before=fingerprint(cfg,scenario,11)[0]
            (root/'routes.xml').write_text('<routes><vehicle/></routes>')
            self.assertNotEqual(before,fingerprint(cfg,scenario,11)[0])

    def test_duplicate_seed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'; path.write_text(json.dumps({'seeds':[1,1],'scenarios':{'peak':{}}}))
            with self.assertRaisesRegex(ValueError,'distinct'): load_config(path)


class StartupTests(unittest.TestCase):
    def _config(self,root):
        (root/'net.xml').write_text('<net><tlLogic id="t" type="static"><phase state="Gr" duration="30"/></tlLogic></net>')
        (root/'run.sumocfg').write_text('<configuration><input><net-file value="net.xml"/></input><random_number><random value="true"/></random_number></configuration>')
        return {'_root':str(root),'_config_path':str(root/'run.sumocfg'),'scenarios':{'peak':{'sumocfg':'run.sumocfg'}}}

    def test_config_random_true_is_overridden_on_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=self._config(root)
            with patch.dict('sys.modules',{'traci':SimpleNamespace()}), patch('traffic_eval.runner.find_sumo',return_value='sumo'), patch('traffic_eval.runner._start_sumo',side_effect=RuntimeError('stop after command construction')) as start:
                with self.assertRaisesRegex(RuntimeError,'stop after'): run_one(cfg,'peak','fixed',11,root/'out')
                command=start.call_args.args[0]
            self.assertEqual(command[command.index('--random')+1],'false')
            self.assertEqual(command[command.index('--seed')+1],'11')

    def test_extra_random_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=self._config(root); cfg['sumo_extra_args']=['--random','true']
            with patch.dict('sys.modules',{'traci':SimpleNamespace()}), patch('traffic_eval.runner.find_sumo',return_value='sumo'):
                with self.assertRaisesRegex(ValueError,'cannot override'): run_one(cfg,'peak','fixed',11,root/'out')

    def test_failed_handshake_terminates_owned_process_and_bounds_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            process=MagicMock(); process.poll.return_value=None
            traci=SimpleNamespace(connect=MagicMock(side_effect=OSError(10013,'Permission denied')))
            sock=MagicMock(); sock.__enter__.return_value.getsockname.return_value=('127.0.0.1',12345)
            with patch('traffic_eval.runner.socket.socket',return_value=sock), patch('traffic_eval.runner.subprocess.Popen',return_value=process):
                with self.assertRaisesRegex(RuntimeError,'10013'): _start_sumo(['sumo'],Path(tmp),traci)
            self.assertEqual(traci.connect.call_args.kwargs['numRetries'],5)
            process.terminate.assert_called_once(); process.wait.assert_called_once_with(timeout=3)


if __name__=='__main__': unittest.main()
