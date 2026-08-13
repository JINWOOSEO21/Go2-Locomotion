"""검증 사전 점검: 의존성 존재 + parkour 패키지가 어느 경로에서 import 되는지."""
import importlib.util as u

for m in ["cupy", "ruamel.yaml", "simple_parsing", "shapely", "scipy", "torch"]:
    try:
        print(m, bool(u.find_spec(m)))
    except ModuleNotFoundError:
        print(m, False)
try:
    import parkour_isaaclab
    print("parkour_isaaclab:", parkour_isaaclab.__file__)
except Exception as e:
    print("parkour_isaaclab: IMPORT FAIL", e)
try:
    import parkour_tasks
    print("parkour_tasks:", parkour_tasks.__file__)
except Exception as e:
    print("parkour_tasks: IMPORT FAIL", e)
