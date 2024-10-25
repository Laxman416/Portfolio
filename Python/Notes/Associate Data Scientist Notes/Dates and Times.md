# Dates and Times

**Course on DataCamp:**
`Working with Dates and Times in Python`

- [Dates and Times](#dates-and-times)
  - [Dates and Calenders](#dates-and-calenders)
  - [Combining Dates and Time](#combining-dates-and-time)
  - [Time Zones](#time-zones)
  - [Datetime in pandas](#datetime-in-pandas)

## Dates and Calenders

**Dates in Python**

`date` class:
```python
from datetime import date

date(year, month, day)

```
*Attributes:*
- `.day`
- `.month`
- `.year`

*Functions:*
-`.weekday()` -> 0 monday 

**Math with Dates**

```python

from datetime import timedelta

td = timedeleta(days = 29)
print(t1 + td)
```
**Turning Dates into Strings**

```python

d.isoformat()
d.strftime("%Y/%m/%d") -> prints date
```
*Functions:*
-`.isoformat()` -> prints date in isoformat string
- `.strftime()` -> prints in custom format
## Combining Dates and Time

```python 
from datetime import datetime

dt = datetime(year = , month = , days = , hours = , minutes =, seconds = , microsecond = )
```

**Printing and parsing datetime**

```python

dt.strftime("%Y-%m-%d %H:%M:%S")

dt = datetime.strptime("", "format-argument")


```
*Functions:*
- `datetime.strptime('' ,'format')` -> parsing dates
- `datetime.fromtimestamp(time)`
 
**Duration**

*Functions:*
- `datetime.total_seconds()` 

```python
from datetime import timedelta

delta = timedelta(seconds = 1)

```

## Time Zones

```python
from datetime import datetime, timedelta, timezone

ET = timezone(timedelta(hours=-5))

dt.astimezone(ET)

dt.replace(tzinfo=timezone.utc) # doesnt change time
dt.astimezone(timezone.utc) # changes time

```
*Functions:*
- `dt.astimezone(timezone(...))` 

**Time Zone database**
```python
from dateutil import tz

ET = tz.gettz('America/New York')
```
*Functions:*
- `tz.gettz('')` 

**DST**
dateutil will figure out

Ending DST

`tz.datetime_ambiguous()` -> if ambigious which dst
`tz.enfold(datetime)`

## Datetime in pandas

`pd.to_datetime(df[''], format = '')`
-> Timestamp same as datetime

converting/parsing:

`.dt.total_seconds()`

summarising:

group by for time: e.g. each month
`df.resample('M', on 'column_name')[].mean()`

**Additional Methods:**
`.dt.tz_localize()`
- parameters:
  - ambiguous = 'NaT' 

`dt.year`
`dt.day_name`

shift column up
`df[''].shift(1)`

Timezones in Pandas:
```python
df[''].dt.tz_localize('America/New York', ambiguous = 'NaT')
#.min will skip over ambiguous
```

**Plotting**

```python
# Resample rides to monthly, take the size, plot the results
rides.resample('M', on = 'Start date')\
  .size()\
  .plot(ylim = [0, 150])

```