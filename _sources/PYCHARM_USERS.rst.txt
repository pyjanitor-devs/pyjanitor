
PyCharm Users
===============

Currently, PyCharm doesn't support the generation of Conda environments via a
YAML file as prescribed above. To get around this issue, you need to setup
your environment as described above, and then within PyCharm point your interpreter
to the predefined conda environment.

1. Complete steps 1-3 under the Getting Started section.
2. Determine the location of the newly-created conda environment::

    conda info --env

3. Open up the location of the cloned pyjanitor directory in PyCharm.
4. Navigate to the Preferences location.

    .. image:: /images/preferences.png

5. Navigate to the Project Interpreter tab.

    .. image:: /images/project_interpreter.png

6. Click the cog at the top right and select Add.

    .. image:: /images/click_add.png

7. Select Conda Environment on the left and then select existing environment. Click
on the three dots, copy the location of your newly-created conda environment,
and append bin/python to the end of the path.

    .. image:: /images/add_env.png

Click OK and you should be good to go!
