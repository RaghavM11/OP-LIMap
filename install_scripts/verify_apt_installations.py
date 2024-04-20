import subprocess


def main(dep_list):
    installed_packages = subprocess.check_output(["apt", "list",
                                                  "--installed"]).decode("utf-8").splitlines()
    installed_packages = [line.split("/")[0] for line in installed_packages]

    for package in dep_list:
        if package not in installed_packages:
            raise Exception(f"Package {package} is not installed.")

    print("All apt dependencies are installed.")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('dep_list', nargs='+',
                        help='List of dependencies to check')  #, required=True)
    args = parser.parse_args()
    main(args.dep_list)
