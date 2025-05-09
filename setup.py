from setuptools import find_packages, setup

package_name = "ros2_depthanything"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thomasluk",
    maintainer_email="thomasluk624@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "depth_to_pointcloud = ros2_depthanything.depth_to_pointcloud:main",
        ],
    },
)
