import os
import platform
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "."):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError("CMake is required to build the RaiSim environments") from exc
        for extension in self.extensions:
            self.build_extension(extension)

    def build_extension(self, extension):
        output_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(extension.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        build_args = ["--config", "Release"]
        if platform.system() != "Windows":
            build_args += ["--", "-j2"]

        os.makedirs(self.build_temp, exist_ok=True)
        environment = os.environ.copy()
        version = self.distribution.get_version()
        environment["CXXFLAGS"] = f'{environment.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{version}\\"'
        subprocess.check_call(["cmake", extension.sourcedir, *cmake_args], cwd=self.build_temp, env=environment)
        subprocess.check_call(["cmake", "--build", ".", *build_args], cwd=self.build_temp)


subpackages = find_packages()

setup(
    name="dexgrasp_zero",
    version="1.0.0",
    description="DexGrasp-Zero RaiSim environments",
    packages=["dexgrasp_zero"] + [f"dexgrasp_zero.{package}" for package in subpackages],
    package_dir={"dexgrasp_zero": "."},
    ext_modules=[CMakeExtension("_raisim_gym")],
    cmdclass={"build_ext": CMakeBuild},
    include_package_data=True,
    zip_safe=False,
)
