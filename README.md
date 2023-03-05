# Surface reconstruction from point clouds

## Part of Intelligent Visual Computing Course Projects, UMass Amherst

Constructing surface from point clouds using the Marching cubes algorithm with naive implicit functions and Moving least squares.

&nbsp;

Use python version < 3.11

Install the requirements using the following command:
```
pip install -r /path/to/requirements.txt
```

&nbsp;

To generate the surface from the point clouds, use the following commands:

Using naive method:
```
python3 basicReconstruction.py --file /path/to/.pts_file --method naive
```

Using Moving least squares method:
```
python3 basicReconstruction.py --file /path/to/.pts_file --method mls
```
