==========================================
Quick Start: A User Guide
==========================================

1. Install the Plugin
------------------------------------------

All methods described below require that you have a Conda environment with Napari installed on your machine.
If this is not yet the case, you can follow the instructions in the `Miniconda documentation <https://docs.anaconda.com/free/miniconda/index.html#latest-miniconda-installer-links>`_ to install Miniconda.

Once Miniconda is installed, use the following commands in your terminal to create a new environment and install Napari:

.. code-block:: bash

    # Create a new Conda environment
    conda create -n mice-env -y python=3.9

    # Activate the new environment
    conda activate mice-env

    # Install Napari with all recommended dependencies
    pip install 'napari[all]'

    # Install additional required libraries from conda-forge
    conda install -c conda-forge libstdcxx-ng

To launch Napari, use the following command in your terminal after activating the correct environment:

.. code-block:: bash

    napari

+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Method                | Instructions                                                                                                                                                             |
+=======================+==========================================================================================================================================================================+
| With pip              | Launch your Conda environment, and type :code:`pip install mouse-in-box`. The plugin should then appear in your plugins list.                                            |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Local install         | Navigate to the directory containing the plugin, enter it in your terminal, and run :code:`pip install -e .`.                                                            |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+



2. Before Starting
------------------------------------------

- Ensure that your videos are in a compatible format. AVI is recommended, but MP4 is acceptable in some cases.
- Obtain a decent computer. A GPU is not required as it won't be used, but at least 8GB of RAM and a decent CPU are recommended.
- Ensure there is ample free disk space. For a 500MB video, approximately 50MB of temporary files are created.
- Before proceeding, open Napari and activate the plugin by clicking on the "Plugins" menu (in the top bar), then selecting "Mouse in Box".


3. Usage
------------------------------------------

a. Video Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- This section enables you to open a video, view its properties, and navigate through it.
- If you already have something opened in Napari, or if you have just analyzed a video, you can reset the plugin by clicking on the :code:`✨ Clear state` button in the plugin's interface.
- Click on the :code:`📂 Select file` button and navigate to your video file.
- The properties of the video, such as name, duration, size, FPS, etc., should appear.
- Use the navigation tools described below to navigate through the video.

+-------------------------+-------------------------------------------------------------------------------------------+
| Name                    | Description                                                                               |
+=========================+===========================================================================================+
| Backward                | Jumps back 25 frames from the current position.                                           |
+-------------------------+-------------------------------------------------------------------------------------------+
| Forward                 | Jumps forward 25 frames from the current position.                                        |
+-------------------------+-------------------------------------------------------------------------------------------+
| Time slider             | Allows free navigation through the video.                                                 |
+-------------------------+-------------------------------------------------------------------------------------------+
| Frame input             | Allows navigation to a precise frame index within the video.                              |
+-------------------------+-------------------------------------------------------------------------------------------+

b. Experiment Duration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Use the minute and second inputs to set the duration of the experiment.
- This duration applies individually to each box, from the "start" frame that we will define later.

c. Box Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- In this section, you will define the box(es) in which the mouse will be tracked.
- A box is defined exclusively by the area within which a mouse is considered visible.
- Start by using the left-click to pan and the mouse-wheel to zoom to position yourself correctly. You must see all the boxes as you won't be able to move the viewer again until the starting frame of each box is set.
- Navigate in the video to a point where nothing obstructs the view of the boxes.
- For each box:
    - Click on the :code:`🔵 Add box` button.
    - A new line should appear in the table below.
    - Pick the color for the box, and rename it if you wish.
    - In the left panel, choose the kind of polygon you want and draw the area where the mouse will be visible.
- Once all the boxes are drawn, navigate through the video to find, for each box, the first frame at which the mouse is in the box and nothing obstructs the view anymore. Click in the "Start" column once you find it.
- When each box has its starting frame, you can move the viewer again.

d. Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Now, we will set the calibration to export distances in physical units rather than pixels.
- Before continuing, measure something in the scene (e.g., the width of a box).
- Then, go to the left panel and add a new shapes layer. The button looks like a little polygon.
- Choose the "line" tool and draw a line over the distance you measured.
- In the "Calibration" section, enter the measured distance and the unit.
- You can now click the :code:`📏 Apply calibration` button.
- At this point, due to the change from pixels to units, your image may appear either huge or extremely small. To center the view around your image, click the home button (the button with the little house, in the lower-left corner).

e. Tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Start by navigating in the video to find a frame where an entire mouse is visible.
- As you did for calibration, add a new shapes layer but this time, use a polygon instead of a line.
- Draw a polygon over the mouse to represent the smallest area considered visible. Once complete, click the :code:`📏 Set area` button.
- You can now click the :code:`♻️ Clear background` button. Wait a few seconds. This operation will create a new hidden layer containing the reference image of the background.
- The button below is for launching tracking. Before clicking on it, you may adjust the threshold using the number input to its right. Tracking takes a couple of minutes.
- You can then click on the :code:`📐 Measure` button to start processing visibility, position, distance, session duration, etc. This is the longest operation.
    - Now, the background of a box is gray if the experiment has not started or is done, blue if the mouse is visible, and red if the mouse is hidden.
    - A red line indicates the mouse's path during this session.
- Finally, you can click on the :code:`📤 Create results tables` button to generate two results tables for this experiment.
    - The first table contains visibility data for the mouse in each box for every frame.
    - The second table (described more precisely in the next section) includes data for each session: duration, distance, visibility, and position.

f. The Results Table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The sessions results table contains seven columns for each box.
- The first three columns cover sessions where the mouse is visible: "[V]".
- The next three columns cover sessions where the mouse is hidden: "[H]".
- The last column counts the number of times the mouse passed through the door during the entire experiment. Divide this number by 2 to get the number of times the mouse exited the box.
- To understand a mouse's behavior, alternate between the first three columns and the next three as we alternate from visible to hidden sessions.
- If you combine the durations from both sets of columns, you should match the theoretical experiment duration specified in the plugin.
- The exported CSV uses semicolons (;) as separators.
