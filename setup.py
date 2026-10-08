from setuptools import find_namespace_packages, setup

# locomotion_isaaclab 에는 최상위 __init__.py 가 없어 namespace package 다.
# find_packages() 는 이런 패키지를 못 찾아서 scripts.* 만 설치되고
# import locomotion_isaaclab 이 실패한다. find_namespace_packages 로 잡아준다.
setup(
    name="go2_locomotion",
    version="0.1",
    packages=find_namespace_packages(include=["locomotion_isaaclab*", "scripts*"]),
)
