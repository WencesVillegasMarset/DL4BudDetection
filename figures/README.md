# Plot Generation

* Run the Plot Geneneration script via Docker 

```console
bash:~$ docker run -v path_to_figures_dir:/figures/ wencesvillegas/budplots:2.1 # or you tagged image
```

This runs the generate_plot.py script and saves plots under the /plots directory

In caso of edits made to the script, build with:

```console
bash:~$ docker build --tag your_tag .
```