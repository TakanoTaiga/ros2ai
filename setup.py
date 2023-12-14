from setuptools import find_packages, setup

package_name = 'ros2ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='taiga',
    maintainer_email='ttttghghnb554z@outlook.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'blip_image_captioning_base = ros2ai.blip_image_captioning_base:main',
            'blip_image_captioning_large = ros2ai.blip_image_captioning_large:main',
            'pix2struct_textcaps_base = ros2ai.pix2struct_textcaps_base:main',
            'vit_gpt2_image_captioning = ros2ai.vit_gpt2_image_captioning:main',
            'resnet_50 = ros2ai.resnet_50:main',
            'vilt_b32_finetuned_vqa = ros2ai.vilt_b32_finetuned_vqa:main',
            'owlvit_base_patch16 = ros2ai.owlvit_base_patch16:main',
            'owlvit_base_patch32 = ros2ai.owlvit_base_patch32:main',
        ],
    },
)
