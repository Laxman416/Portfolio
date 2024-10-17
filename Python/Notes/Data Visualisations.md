# Data Visualisations <!-- omit in toc -->

**Course on Datacamp:**<br> 
`'Introduction to Data Visualisation with Matplotlib'`<br> 
`'Introduction to Data Visualisation with Seaborn'`<br> 

- [Introduction to Data Visualisation with Matplotlib](#introduction-to-data-visualisation-with-matplotlib)
  - [Introduction to Matplotlib](#introduction-to-matplotlib)
    - [Introduction](#introduction)
    - [Customising plots](#customising-plots)
    - [Small Multiples](#small-multiples)
  - [Plotting Time-series](#plotting-time-series)
    - [Plotting Time-series Data](#plotting-time-series-data)
    - [Plotting Time-series Data with Different Variables](#plotting-time-series-data-with-different-variables)
    - [Annotating Time-series Data](#annotating-time-series-data)
  - [Quantitative Comparisons and Statistical Visualisations](#quantitative-comparisons-and-statistical-visualisations)
    - [Bar-charts](#bar-charts)
    - [Histograms](#histograms)
    - [Statistical Plotting](#statistical-plotting)
    - [Scatter Plots](#scatter-plots)
  - [Sharing Visualisations with Others](#sharing-visualisations-with-others)
    - [Preparing Figures to Share](#preparing-figures-to-share)
    - [Saving Visualisations](#saving-visualisations)
    - [Automating Figures from Data](#automating-figures-from-data)
- [Introduction to Data Visualisation with Seaborn](#introduction-to-data-visualisation-with-seaborn)
  - [Introduction to Seaborn](#introduction-to-seaborn)
    - [Introduction](#introduction-1)
    - [Using Pandas with Seaborn](#using-pandas-with-seaborn)
    - [Adding a Third Variable with Hue](#adding-a-third-variable-with-hue)
  - [Visualising Two Quantitative Variables](#visualising-two-quantitative-variables)
    - [Introduction to relational plots and subplots](#introduction-to-relational-plots-and-subplots)
    - [Customising Scatter Plots](#customising-scatter-plots)
    - [Introduction to Line Plots](#introduction-to-line-plots)
  - [Visualising a Categorical and Quantitative Variable](#visualising-a-categorical-and-quantitative-variable)
    - [Count Plots and Bar Plots](#count-plots-and-bar-plots)
    - [Box Plots](#box-plots)
    - [Point Plots](#point-plots)
  - [Customising Seaborn Plots](#customising-seaborn-plots)
    - [Changing plot style and colour](#changing-plot-style-and-colour)
    - [Adding Titles and Labels: I](#adding-titles-and-labels-i)
    - [Adding Titles and Labels: II](#adding-titles-and-labels-ii)


# Introduction to Data Visualisation with Matplotlib

## Introduction to Matplotlib
 
### Introduction

```python
fig, ax = plt.subplots()
# fig is a container
# ax holds data: canvas

ax.plot(df['column1'],df['column2'])
ax.plot(df['column4'],df['column3'])
# plots 2 graphs
```

### Customising plots

paramters of plot:
- add markers: `markers` = 'o'
- `linestyle` = '--'/ 'None'/ ...
- `color`

Setters
- `ax.set_xlabel('')`
- `ax.set_ylabel('')`
- `ax.set_title('')`

### Small Multiples

```python

fig, ax = plt.subplots(3,2)

ax[0,0]

fig, ax = plt.subplots(2,1, sharey =True)

ax[0]
ax[1]
```

## Plotting Time-series

### Plotting Time-series Data

```python

ax.plot(df.index, df[column])
# if index is date

```
### Plotting Time-series Data with Different Variables

**`ax2 = ax.twinx()`** -> creates seperate y axis
```python
fig,ax = plt.subplots()

ax.plot(..., color - ...)
ax2 = ax.twinx()
ax2.plot(...)
```
color y_axis and y_axis_ticks

`ax.tick_params('y', colors = 'blue')`

Create Fn to colour and plot and label
```python
def plot_timeseries(axes,x,y,color,xlabel,ylabel):
  axes.plot(...)
  axes.set_xlabel()
  axes.set_ylabel()
  axes.tick_params()
```

### Annotating Time-series Data

Annotating:
```python
ax.annotate(
  "",
  xy = (x,y), 
  xytext = (x,y),
  arrowprops = {"arrowstyle":"->", "colour":"gray"})
```

## Quantitative Comparisons and Statistical Visualisations

### Bar-charts

- stacked bar chart requires bottom arguement
- label parameter for legend
- rotation partater in set_xlabel
  
```python
ax.bar(df.index,column)
ax.bar(df.index,column2, bottom = colum1, label = '')

ax.set_xlabel(df.index, rotation = 90)


```
### Histograms

`histtype = 'step'` -> to not fill hist

```python
fig,ax = plt.subplots()
ax.hist(df.column, histtype = "step")
ax.hist(df.column2)

```
### Statistical Plotting

adding errorbars: 
- hist: parameters `xerr` and `yerr` df['column'].std() 
- line: ax.errorbar(x,y,yerr) - vertical markers added
- boxplots: 
```python
ax.boxplots([x,y])
ax.set_xticklabels(['',''])
```
### Scatter Plots

saving 3rd variable by colour
```python

ax.scatterplots(x,y, c = column)
```

## Sharing Visualisations with Others

### Preparing Figures to Share

`plt.style.use("ggplot")`
`plt.style.use("seaborn-colorblind")`

### Saving Visualisations
`fig.set_size_inches([width, height])` -> set size
`plt.savefig("path", quality = num)` -> quality high -> smallest file
`plt.savefig("path.svf", quality = num, dpi = 300)` -> .svg to edit later

### Automating Figures from Data

```python

for sport in sports:
  slice
  plot bar
plt.show
# plots bar chart for each sport 
```

# Introduction to Data Visualisation with Seaborn

Course on Datacamp 'Introduction to Data Visualisation with Seaborn'

## Introduction to Seaborn

### Introduction

- easy to create most common types
- pandas

scatterplot: `sns.scatterplot(x)`
countplot: `sns.countplot(x)`
hist: `sns.histplot()`
### Using Pandas with Seaborn

scatterplot: `sns.scatterplot(x,y,data = df)`
countplot: `sns.countplot(x,data=df)`

### Adding a Third Variable with Hue

parameter: `hue = 'column', hue_order = ['',''], palette`
```
hue_colors = {'Yes': "black",
              'No': "red"}
```

scatterplot: `sns.scatterplot(x,y,data = df, hue = 'column')`


## Visualising Two Quantitative Variables

parameter: `kind`, `row/col` similiar to hue but plots on different plot
`col_wrap` -> how many plots per row
`col_order = list` -> order of plot output

### Introduction to relational plots and subplots
**parameter:**
relationship plots using subplots between subgroups

```python
sns.replot(x,y,data,kind = 'scatter',col/row = 'column' )
```

### Customising Scatter Plots

- point size and style
- point transparency

**parameter:**
- `size = column` : use along with `hue = column`
- `style = column`: plots subgroups with diff points style
- `aplha = float`
  
### Introduction to Line Plots

`sns.lineplot()`
over time
**parameter:**
- `kind = 'line'`
- `marker = True` -> displays marker for each point
- `dashes = False` -> plots different lines with same linestyle

**Multiple observation per x value**

scatterplot plots all
lineplot displays mean with confidence level - 95%

**parameter:**
`ci = "sd"` -> confidence interval as std

## Visualising a Categorical and Quantitative Variable

### Count Plots and Bar Plots

Categorical Plots: Count and Bar plots
`catplot()` -> same as replot: ca use subplots with `col = ` and `row =`

**paramters:**
- `order = list`
- `kind = 'count'`

Change orientation by switching x and y

### Box Plots

**paramters:**
- `order = list`
- `kind = 'box'`
- `sym = ""` -> emit outliers
- `whis = 2.0 or [5,95]` -> extendes whiskers to 2IQR or 5 to 95 percentiles

### Point Plots

show mean of quantitative plots.
point vs line: point uses catergorical in one axis
point plot stacks above each other compared to bar which is side by side. easier to compare

**paramters:**
- `kind = 'point'`
- `join = False` -> removes line between points
- `estimator = median` -> displays median instead of mean
- `capsize = 0.2`
- `ci = 'None'/'sd'`


## Customising Seaborn Plots

### Changing plot style and colour

`sns.set_style('white')`
`sns.set_style('ticks')`
`sns.set_style('whitegrid')` -> better for specific values

`sns.set_pallet(RdBu)`
- Diverging  pallete: `RdBu`
- Sequential pallete: continous scale: `Blyes`

or insert list of colours -> custom pallete

`sns.set_context('paper'/'talk')` -> talk increase size of font etc

### Adding Titles and Labels: I

FacetGrid: replot, catplot -> creates subplots
AxesSubplots: single plot

FacetGrid: parameter = `g.fig.suptitle()`
```python
g= sns.catplot(...)
g.fig.suptitle("",y= 1) -> y defines where the title is
```

### Adding Titles and Labels: II

AxesSubplots: 
parameter:
- `g.fig.set_title()`

```python
g= sns.boxplot(...)
g.fig.set_title("",y= 1) -> y defines where the title is
```
AxesSubplots and FacedGrid:
**parameters:**
- `g.set(xlabel = , ylabel = )`

Rotate x-axis labels with mpl
`plt.xticks(rotation)`