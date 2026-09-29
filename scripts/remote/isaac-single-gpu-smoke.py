import json
from pathlib import Path
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"multi_gpu":False,"active_gpu":0,"physics_gpu":0,"width":256,"height":256,"renderer":"RaytracedLighting"})
for _ in range(10):
    app.update()
Path("/workspace/SafetyJev/artifacts/isaac-single-gpu-startup.json").write_text(json.dumps({"isaac_sim":"4.5.0","headless_startup":True,"updates":10,"maniguard_rollout":False})+"\n")
print("SAFETYJEV_ISAAC_STARTUP_OK",flush=True)
app.close()
