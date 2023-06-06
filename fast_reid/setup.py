# -*- coding: utf-8 -*-
from setuptools import setup
import os
import importlib
from typing import Union
import glob
import warnings


def get_cuda_home() -> Union[str, None]:
    """Get cuda home path if cuda runtime toolkit is installed.

    Normally cuda is installed at /usr/local/cuda.
    Please set CUDA_HOME environment variable if it is not installed at default path.

    Returns
    -------
    Union[str, None]
        Returns the root path of cuda toolkit.
    """
    cuda_home_env = os.getenv("CUDA_HOME")
    if cuda_home_env is None:
        cuda_path = os.popen("whereis cuda").read()[6:-1]
        if len(cuda_path) < 4:
            cuda_path = "/usr/local/cuda"
    else:
        cuda_path = cuda_home_env

    return cuda_path


def get_cuda_version() -> Union[str, None]:
    """Get cuda version if it is installed.

    Normally cuda is installed at /usr/local/cuda. Details can be found here:
    https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#ubuntu-x86_64-run

    Returns
    -------
    Union[str, None]
        Returns version of cuda(eg, 9.0, 10.2) if cuda is found.
    """
    # specify the cuda version manually.
    manual_cuda_version = os.getenv("CUDA_RUNTIME_VERSION", None)
    if manual_cuda_version:
        return manual_cuda_version

    cuda_path = get_cuda_home()

    cuda_exists = os.path.exists(cuda_path)

    cuda_runtime_file = os.path.join(cuda_path, "lib64/libcudart.so.*")

    if cuda_exists:
        # check if libcudart.so* exists
        cuda_runtime_file_getter = glob.glob(cuda_runtime_file)

        if not cuda_runtime_file_getter:
            warnings.warn(
                "CUDA libcudart.so* not found at CUDA_HOME, fall back to system path"
            )
            cuda_runtime_file = "/usr/lib/x86_64-linux-gnu/libcudart.so*"
            cuda_runtime_file_getter = glob.glob(cuda_runtime_file)

            if not cuda_runtime_file_getter:
                raise Exception(
                    "libcudart.so* could not be found, please report a bug to https://github.com/AnyVisionltd/ai-module-builder/issues"
                )

        ver_string = os.popen(
            f'ls {cuda_runtime_file} | sort | tac | head -1 | rev | cut -d"." -f -3 | rev | cut -f1,2 -d"."'
        ).read()
        print("CUDA version :" + ver_string)
        return ver_string.strip()
    else:
        return None


packages = [
    "fastreid",
    "fastreid.config",
    "fastreid.data",
    "fastreid.data.datasets",
    "fastreid.data.samplers",
    "fastreid.data.transforms",
    "fastreid.engine",
    "fastreid.evaluation",
    "fastreid.evaluation.rank_cylib",
    "fastreid.layers",
    "fastreid.modeling",
    "fastreid.modeling.backbones",
    "fastreid.modeling.backbones.regnet",
    "fastreid.modeling.heads",
    "fastreid.modeling.losses",
    "fastreid.modeling.meta_arch",
    "fastreid.solver",
    "fastreid.solver.optim",
    "fastreid.utils",
]

package_data = {
    "": ["*"],
    "fastreid.modeling.backbones.regnet": ["effnet/*", "regnetx/*", "regnety/*"],
}

install_requires = [
    "scikit-learn>=1.2.1,<2.0.0",
    "tabulate>=0.9.0,<0.10.0",
    "termcolor>=2.2.0,<3.0.0",
    "yacs>=0.1.8,<0.2.0",
]

cuda_version = get_cuda_version()

if cuda_version is not None:
    install_requires.append("faiss-gpu>=1.7.2,<2.0.0")
else:
    install_requires.append("faiss-cpu>=1.7.2,<2.0.0")


pytorch_spec = importlib.util.find_spec("torch")

# prefer not updating pytorch
if pytorch_spec is None:
    install_requires.append("torch==1.13.1")
    install_requires.append("torchvision==0.14.1")

setup_kwargs = {
    "name": "fastreid",
    "version": "1.4.100",
    "description": "SOTA Re-identification Methods and Toolbox re-wrapped by Oosto", 
    "long_description": '<img src=".github/FastReID-Logo.png" width="300" >\n\n[![Gitter](https://badges.gitter.im/fast-reid/community.svg)](https://gitter.im/fast-reid/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)\n\nGitter: [fast-reid/community](https://gitter.im/fast-reid/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link)\n\nWechat: \n\n<img src=".github/wechat_group.png" width="150" >\n\n\nFastReID is a research platform that implements state-of-the-art re-identification algorithms. It is a ground-up rewrite of the previous version, [reid strong baseline](https://github.com/michuanhaohao/reid-strong-baseline).\n\n## What\'s New\n\n- [Sep 2021] [DG-ReID](https://github.com/xiaomingzhid/sskd) is updated, you can check the [paper](https://arxiv.org/pdf/2108.05045.pdf).\n- [June 2021] [Contiguous parameters](https://github.com/PhilJd/contiguous_pytorch_params) is supported, now it can\n  accelerate ~20%.\n- [May 2021] Vision Transformer backbone supported, see `configs/Market1501/bagtricks_vit.yml`.\n- [Apr 2021] Partial FC supported in [FastFace](projects/FastFace)!\n- [Jan 2021] TRT network definition APIs in [FastRT](projects/FastRT) has been released! \nThanks for [Darren](https://github.com/TCHeish)\'s contribution.\n- [Jan 2021] NAIC20(reid track) [1-st solution](projects/NAIC20) based on fastreid has been released！\n- [Jan 2021] FastReID V1.0 has been released！🎉\n  Support many tasks beyond reid, such image retrieval and face recognition. See [release notes](https://github.com/JDAI-CV/fast-reid/releases/tag/v1.0.0).\n- [Oct 2020] Added the [Hyper-Parameter Optimization](projects/FastTune) based on fastreid. See `projects/FastTune`.\n- [Sep 2020] Added the [person attribute recognition](projects/FastAttr) based on fastreid. See `projects/FastAttr`.\n- [Sep 2020] Automatic Mixed Precision training is supported with `apex`. Set `cfg.SOLVER.FP16_ENABLED=True` to switch it on.\n- [Aug 2020] [Model Distillation](projects/FastDistill) is supported, thanks for [guan\'an wang](https://github.com/wangguanan)\'s contribution.\n- [Aug 2020] ONNX/TensorRT converter is supported.\n- [Jul 2020] Distributed training with multiple GPUs, it trains much faster.\n- Includes more features such as circle loss, abundant visualization methods and evaluation metrics, SoTA results on conventional, cross-domain, partial and vehicle re-id, testing on multi-datasets simultaneously, etc.\n- Can be used as a library to support [different projects](projects) on top of it. We\'ll open source more research projects in this way.\n- Remove [ignite](https://github.com/pytorch/ignite)(a high-level library) dependency and powered by [PyTorch](https://pytorch.org/).\n\nWe write a [fastreid intro](https://l1aoxingyu.github.io/blogpages/reid/fastreid/2020/05/29/fastreid.html) \nand [fastreid v1.0](https://l1aoxingyu.github.io/blogpages/reid/fastreid/2021/04/28/fastreid-v1.html) about this toolbox.\n\n## Changelog\n\nPlease refer to [changelog.md](CHANGELOG.md) for details and release history.\n\n## Installation\n\nSee [INSTALL.md](INSTALL.md).\n\n## Quick Start\n\nThe designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder\'s purpose by yourself.\n\nSee [GETTING_STARTED.md](GETTING_STARTED.md).\n\nLearn more at out [documentation](https://fast-reid.readthedocs.io/). And see [projects/](projects) for some projects that are build on top of fastreid.\n\n## Model Zoo and Baselines\n\nWe provide a large set of baseline results and trained models available for download in the [Fastreid Model Zoo](MODEL_ZOO.md).\n\n## Deployment\n\nWe provide some examples and scripts to convert fastreid model to Caffe, ONNX and TensorRT format in [Fastreid deploy](tools/deploy).\n\n## License\n\nFastreid is released under the [Apache 2.0 license](LICENSE).\n\n## Citing FastReID\n\nIf you use FastReID in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.\n\n```BibTeX\n@article{he2020fastreid,\n  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},\n  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},\n  journal={arXiv preprint arXiv:2006.02631},\n  year={2020}\n}\n```\n',
    "author": "He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao",
    "author_email": "None",
    "maintainer": "None",
    "maintainer_email": "None",
    "url": "https://github.com/JDAI-CV/fast-reid",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.8,<4.0",
}


setup(**setup_kwargs)
