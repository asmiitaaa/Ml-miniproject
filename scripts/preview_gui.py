"""Replay actual controllers in SUMO-GUI; optionally coordinate native captures.

Native capture avoids SUMO's failing OpenGL screenshot path on this computer.
With --external-capture, Codex's Windows capture helper supplies genuine PNGs
and acknowledges each file; no traffic graphics are synthesized by this script.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from traffic_eval.runner import load_config,check_inputs,input_files,prepare_network,find_sumo,fingerprint,resolve,sha256
from traffic_eval.collisions import collision_args


def atomic_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    # Windows can briefly deny replacement while the capture helper reads
    # the previous request. Retry the atomic replacement instead of losing
    # an otherwise valid replay to a transient sharing violation.
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39:raise
            time.sleep(.05)


def main():
    import traci
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/evaluation.model_validation.json')
    parser.add_argument('--scenario',default='peak')
    parser.add_argument('--controllers',nargs='+',choices=['fixed','actuated','dqn'],default=['dqn'])
    parser.add_argument('--seconds',type=int,default=120)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--every',type=int,default=15)
    parser.add_argument('--delay-ms',type=int,default=100)
    parser.add_argument('--pace-ms',type=int,default=0,help='Wall-clock pacing between steps, independent of SUMO GUI delay')
    parser.add_argument('--output',default=None)
    parser.add_argument('--external-capture',action='store_true')
    args=parser.parse_args()
    if min(args.seconds,args.every)<=0 or args.seconds%args.every:parser.error('seconds must be a positive multiple of every')
    cfg=load_config(args.config)
    check_inputs(dict(cfg,scenarios={args.scenario:cfg['scenarios'][args.scenario]}),args.controllers)
    out=Path(args.output or ('demonstrations/preview_'+time.strftime('%Y%m%d_%H%M%S'))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    digest,hashes=fingerprint(cfg,cfg['scenarios'][args.scenario],args.seed)
    sumocfg,network,_=input_files(cfg,cfg['scenarios'][args.scenario])
    records=[]
    try:
        for controller in args.controllers:
            folder=out/controller;folder.mkdir();(folder/'frames').mkdir()
            options=dict(cfg['controllers'][controller])
            netcopy=folder/'controlled.net.xml';prepare_network(network,netcopy,controller,options)
            with socket.socket() as port_socket:
                port_socket.bind(('127.0.0.1',0));port=port_socket.getsockname()[1]
            command=[find_sumo(cfg,True),'-c',str(sumocfg),'--net-file',str(netcopy),'--remote-port',str(port),
                     '--seed',str(args.seed),'--random','false','--begin','0','--end',str(args.seconds+2),
                     '--step-length','1','--lateral-resolution',str(cfg.get('lateral_resolution_m',.4)),
                     '--start','true','--quit-on-end','true','--delay',str(args.delay_ms),
                     '--breakpoints',str(args.seconds+10),'--window-size','720,760','--window-pos','20,20',
                     '--no-step-log','true','--duration-log.disable','true','--error-log',str(folder/'errors.log')]
            command+=collision_args(cfg,folder)
            connection=policy=process=None
            record={'controller':controller,'scenario':args.scenario,'seed':args.seed,'demand_hash':digest,'frames':[],'command':command,'status':'running'}
            try:
                with (folder/'process.log').open('w',encoding='utf-8') as log:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    connection=traci.connect(port=port,host='127.0.0.1',numRetries=5,proc=process)
                    view=connection.gui.getIDList()[0]
                    tls=options.get('tls_id') or next(iter(options.get('control_tls_ids',[])),None)
                    junction=next((n for n in ET.parse(network).findall('junction') if n.get('id')==tls),None)
                    if junction is not None:
                        x,y=float(junction.get('x')),float(junction.get('y'))
                        boundary=[x-150,y-150,x+150,y+150]
                    else:boundary=[v for corner in connection.simulation.getNetBoundary() for v in corner]
                    connection.gui.setBoundary(view,*boundary);connection.gui.setAngle(view,0)
                    record['camera_boundary']=boundary
                    if controller=='dqn':
                        module,factory=options['factory'].split(':',1)
                        checkpoint=resolve(cfg,options['checkpoint'])
                        policy=getattr(importlib.import_module(module),factory)(connection=connection,checkpoint=str(checkpoint),options={**options,'seed':args.seed,'_step_length_s':1,'_run_dir':str(folder),'_lateral_resolution_m':cfg.get('lateral_resolution_m',.4)})
                        record['checkpoint_sha256']=sha256(checkpoint)
                        record['controller_metadata']=policy.metadata
                    for tick in range(1,args.seconds+1):
                        if policy is not None:policy.step(connection.simulation.getTime())
                        connection.simulationStep()
                        if args.pace_ms>0:time.sleep(args.pace_ms/1000)
                        if args.external_capture and tick%args.every==0:
                            index=tick//args.every-1
                            image=folder/'frames'/f'frame_{index:05d}.png'
                            ack=image.with_suffix('.ack.json')
                            request={'status':'capture','controller':controller,'clock_s':tick,'index':index,
                                     'image_file':str(image),'ack_file':str(ack),'process_id':process.pid,'demand_hash':digest}
                            atomic_json(out/'capture_request.json',request)
                            print(f'WAITING FOR NATIVE CAPTURE {controller} at {tick}s',flush=True)
                            deadline=time.monotonic()+180
                            while not ack.is_file():
                                if time.monotonic()>deadline:raise TimeoutError('Native screenshot helper did not acknowledge frame')
                                time.sleep(.1)
                            ack_data=json.loads(ack.read_text(encoding='utf-8'))
                            if ack_data['sha256']!=sha256(image):raise ValueError('Native image integrity mismatch')
                            record['frames'].append({'file':str(image.relative_to(out)),'simulation_clock_s':tick,'sha256':sha256(image)})
                            atomic_json(folder/'manifest.json',record)
                    record['status']='complete'
                    print(f'GUI REPLAY COMPLETED {controller}: {args.seconds}s',flush=True)
            finally:
                if policy is not None:policy.close()
                if connection is not None:
                    try:connection.close(wait=False)
                    except Exception:pass
                if process is not None:
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                atomic_json(folder/'manifest.json',record)
            records.append(record)
        if args.external_capture:
            from record_demo import compose
            compose_args=argparse.Namespace(width=720,height=760,scenario=args.scenario,seed=args.seed,no_video=False,format='mp4',fps=1)
            video,_=compose(out,records,compose_args,'SUMO-GUI: matched three-controller replay | uncalibrated demand')
            manifest={'status':'complete','full_demonstration_complete':True,'capture_method':'Windows native window capture via Codex Computer Use',
                      'controllers':args.controllers,'scenario':args.scenario,'seed':args.seed,'demand_hash':digest,'input_sha256':hashes,
                      'seconds':args.seconds,'capture_interval_s':args.every,'video':video.name,'video_sha256':sha256(video),
                      'note':'Actual SUMO-GUI windows, same frozen demand/camera/seed/clock. Native capture avoids the local SUMO snapshot-renderer hang. This is a sampled-time demonstration, not continuous real-time screen recording.'}
            atomic_json(out/'recording_manifest.json',manifest)
        atomic_json(out/'capture_request.json',{'status':'complete'})
    except BaseException as exc:
        atomic_json(out/'capture_request.json',{'status':'failed','error':str(exc)})
        raise


if __name__=='__main__':main()
