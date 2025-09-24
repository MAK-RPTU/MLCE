Link: https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874

To create environment:

Method 1: Using Anaconda Navigator

Method 2: Type the following commands in the sequence below:

>conda create -n MLC2023 //create an environment name "MLC2023" in env folder of anaconda
>conda activate MLC2023 //Activate the environment

The below commands are executed after activation for each environment so its not efficient.
>conda install numpy
>conda install matplotlib
>conda install jupyter

After installing the above packages for environment.

>jupyter notebook //If we dont install above packages for environment it will not work

Ctrl+C to terminate in command window

In Jupyter although the kernel is default python3 but we can test our environment by using
following code:

# To verify the environment
import os
print (os.environ['CONDA_DEFAULT_ENV'])
