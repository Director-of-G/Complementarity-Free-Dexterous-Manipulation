import os
import signal
import subprocess
import xml.etree.ElementTree as ET

from examples.mpc.allegro.ball.params import ExplicitMPCParams


def change_sphere_radius_in_mjcf(xml_path, radius):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 2. 查找并修改 geom 标签的 size 属性
    for geom in root.iter('geom'):
        if 'name' in geom.attrib and geom.attrib['name'] == 'object_geom':  # 假设 geom 名字是 'object_geom'
            geom.set('size', str(radius))  # 修改为新的 size 值
            break

    # 3. 将修改后的 XML 写回文件
    tree.write(xml_path)

param_list = [
    0.05, 0.052, 0.054, 0.056, 0.058,
    # 0.06,
    0.062, 0.064
]

mjcf_path = ExplicitMPCParams().model_path_

pwd1 = '~/research/inhand_manipulation/dex_playground/ros2_ws/src/complementarity_free_control/src/Complementarity-Free-Dexterous-Manipulation'
setup_cmd1 = "source /home/jyp/.bashrc"

session_name = "sweep_params_and_test_system"

for param in param_list:
    print(f"\n=== Testing with param: {param} ===")

    run_cmd1 = "python ./examples/mpc/allegro/ball/test.py --result_folder sweep_radius_" + str(param).replace('.', '_')

    change_sphere_radius_in_mjcf(mjcf_path, param)

    # 启动子进程（独立进程组，便于 kill）
    p = subprocess.Popen(
        f"cd {pwd1} && {setup_cmd1} && {run_cmd1}",
        shell=True,
        executable="/bin/bash",
        preexec_fn=os.setsid  # 创建独立进程组
    )

    try:
        # 等待退出（可设置 timeout）
        p.wait(timeout=7200)  # 2 hours
        print("Process finished normally with code:", p.returncode)

    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        print("Process timeout, killing...")

        # 向整个进程组发送SIGINT（等同于 Ctrl-C）
        os.killpg(os.getpgid(p.pid), signal.SIGINT)

        # 再等它一会儿（如果还没死，可以 SIGKILL）
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("Force killing...")
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)

print("=== All tests completed. ===")
