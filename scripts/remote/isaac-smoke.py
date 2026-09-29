import json
from pathlib import Path
from isaacsim import SimulationApp
app=SimulationApp({"headless":True})
for _ in range(10):
    app.update()
Path("/workspace/SafetyJev/artifacts/isaac-startup.json").write_text(json.dumps({"isaac_sim":"4.5.0","headless_startup":True,"updates":10,"maniguard_rollout":False})+"\n")
print("SAFETYJEV_ISAAC_STARTUP_OK",flush=True)
app.close()
