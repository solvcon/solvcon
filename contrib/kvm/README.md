# KVM-based Ubuntu Environment of SOLVCON
To automate Ubuntu installlation to run [SOLVCON](https://github.com/solvcon/solvcon).

# Getting Started

## Prerequisites
An Ubuntu Xenial desktop with the following files/tools/scripts of the packages installed:
- *libvirt.pc* (provided by *libvirt-dev*)
- *Python.h* (provided by *libpython3.5-dev*)
- *cloud-localds* (provided by *cloud-image-utils*)
- *libvirt-python* (you may install it in your virtual env by *pip install libvirt-python*
- optional: You may want to have *virt-manager* or *virt-viewer* in your system to see the provisioned system.

### Install the Prerequisites
#### Debian Packages

    sudo apt-get install libvirt-dev
    sudo apt-get install libpython3.5-dev
    sudo apt-get install cloud-image-utils

#### libvirt-python

If your system does not provide libvirt-python package, you may want to install it by

    virtualenv -p python3 venv
    source venv/bin/activate
    pip install libvirt-python

#### virt-manager

    sudo apt-get install virt-manager

For most normal users, you may want to add yourself into the libvirt group. Acheive it by

    sudo adduser `id -un` libvirt

## Installing Source

Fetch the source. That's it!

Remember to activate the virtual environment if you have not done it

    virtualenv -p python3 venv
    source venv/bin/activate

If you want to connect the KVM system later over SSH, you need to paste your public ssh key [user-data](https://github.com/solvcon/solvcon/tree/master/contrib/kvm/data/user-data)

    ssh_authorized_keys:
     - put_your_ssh_public_key replace_the_string

For example, it could be

    ssh_authorized_keys:
     - ssh-rsa AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA you@yourhost

# Run
Go to the root path of the source code branch, and run

    ./bin/solvcon-kvm

You may need to
- Provide your sudo password to execute some commands of the scripts.
- Make sure you are in the pre-requisites-ready working environment, e.g. a virtual Python environment installed necessary packages.

This executable, [ubuntu-kvm](https://github.com/solvcon/solvcon/tree/master/contrib/ubuntu-kvm), is a wrapper of several scripts to
- Download an Ubuntu image file.
- Create a KVM domain
- Install the Ubuntu image in the KVM domain
- Install SOLVCON in the Ubuntu system installed in the KVM domain

Once the installation completes, it will pop up IP information to access the KVM instance. Log in the instance by

    ssh ubuntu@<The IP shown on stdout>

with password *passw0rd*, or your private key if you have pasted your public key in [user-data](https://github.com/solvcon/solvcon/tree/master/contrib/kvm/data/user-data).

After provisioning, you will get the files and folders shown below:

| File/Folder | Description |
| ----------- | ----------- |
| solvcon | SOLVCON source code. It is pulled by git. |
| miniconda | Miniconda environment. It is ready to run SOLVCON and enabled when you log in the guest system. |
| cloud-init-solvcon-output.log | cloud-init output log when installing SOLVCON. If the installation completes, the log should show the unit test results. |
