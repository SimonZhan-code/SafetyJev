import numpy as np

def episode_fixture(steps=16,violation_step=6):
    proposals=[];execution=[]
    for start in range(0,steps,8):
        commands=(np.arange(64,dtype=np.float32).reshape(8,8)/100+start).tolist()
        pid=f'p{start}'
        proposals.append({'proposal_id':pid,'start_step':start,'planned_commands':commands,
                          'raw_actions':commands+commands,'planned_steps':8})
        for offset in range(min(8,steps-start)):
            execution.append({'step':start+offset,'proposal_id':pid,'offset':offset,
                              'command':commands[offset].copy()})
    return {'result':{'status':'completed'},'episode_id':'episode','group_id':'jar/task_0000','action_dt_s':.05,
            'state_features':[f'arm_q_{i}' for i in range(7)]+[f'arm_qdot_{i}' for i in range(7)]+['gripper_q','gripper_qdot'],
            'constraints':[{'id':'upright','question':'Will the jar tilt beyond 30 degrees?', 'requires_history':False}],
            'observations':[{'step':t,'robot_state':[float(t)]*16,
                'images':{'overview':f'observations/{t}-overview.png','wrist':f'observations/{t}-wrist.png'}} for t in range(steps+1)],
            'proposals':proposals,'execution':execution,
            'oracle':[{'step':t,'valid':True,'ap':{'upright':violation_step is None or t<violation_step},
                'violated':{'upright':violation_step is not None and t>=violation_step}} for t in range(steps+1)]}
