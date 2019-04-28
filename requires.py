import pip
import subprocess
import sys

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])

if __name__ == '__main__':
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    packets = ['crawler','python-pixabay','numpy','scipy','matplotlib','sklearn', 'keras', 'tensorflow']
    for packet in packets:
        if packet not in installed_packages:
            print("Installing packet {}".format(packet))
            install(packet)
