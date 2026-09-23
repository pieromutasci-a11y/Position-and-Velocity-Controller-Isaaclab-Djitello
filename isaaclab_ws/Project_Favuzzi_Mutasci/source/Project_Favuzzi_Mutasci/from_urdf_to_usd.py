import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Converti URDF Tello in USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

def main():
    urdf_cfg = UrdfConverterCfg(
        asset_path="tello.urdf",
        usd_dir="./usd_out",
        usd_file_name="tello.usd",
        fix_base=False,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        joint_drive=None,
    )
    converter = UrdfConverter(urdf_cfg)
    print("USD generato in:", converter.usd_path)

if __name__ == "__main__":
    main()
    simulation_app.close()
