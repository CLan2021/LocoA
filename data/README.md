### Example data

##### Meta dataset
One_meta2.csv is a tab delimited file that includes information on each monitoring experiment and all the beetles used.

##### Device output file
Whenever the monitors are connected, the system automatically creates .txt output files. 
The monitor data file format is as follows:

TriKinetics MonitorNN.txt data files are tab-delimited text files, organized into 42 columns per row, as follows:
- 1 Index (Incremented with each reading, 1 at program launch)
- 2 Date (DD MMM YY)
- 3 Time (HH:MM:SS)
- 4 Monitor status (1 = valid data, 51 = no data received)
- 5 Extras (Number of extra readings consolidated by Filescan)
- 6 Monitor number (1-120)
- 7 Tube number (1-32, 0 if monitor row)
- 8 Data type (MT, M1F, CT, C12FP, D3, Pn, Rt, TA, etc)
- 9 unused
- 10 Light sensor (1 = On, 0 = Off, not present in all units)
- 11-42 Data columns (1 per tube if monitor row)

In our case, we set the index to be 1 minute per increment (row) and the data type is counts.
