# czi2jpg

Conversion from Zeiss microscopes CZI files to jpg, with downscaling
and annotation.

```
./czi2jpg.py -a poi.csv -d 4 -o output.jpg tiled.czi
```

* `-a CSV` gives a list of points that should be annotated
* `-d N` gives the downsampling factor (in both dimensions)
* `-o`

The CSV file has the format:

* First line is ignored
* Subsequent lines have:
  * First column is `i` for circles, `r` for squares, anything else ignored
  * Second column is the x co-ordinate
  * Third column is the y co-ordinate
  * Fourth column is ignored
  * Fifth column is a string containing the number that will be printed beside the POI mark

# Building the wheel

```sh
pipenv lock -r > requirements.txt
pip wheel . -r requirements.txt
```
