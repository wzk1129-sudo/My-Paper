import json
from anatomy_controlnet_v1.runtime import construct
m,a=construct("C1_CAPACITY","cuda:0")
print(json.dumps(a,indent=2),flush=True)
